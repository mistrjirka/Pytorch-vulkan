"""Match CUDA fused SDPA and math SDPA against the Vulkan implementation."""
import argparse,json,statistics,time,contextlib
import torch
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--backend',choices=['cuda','vulkan'],required=True)
p.add_argument('--impl',choices=['auto','math'],default='auto')
p.add_argument('--batch',type=int,default=16)
p.add_argument('--heads',type=int,default=8)
p.add_argument('--seq',type=int,default=128)
p.add_argument('--dim',type=int,default=32)
p.add_argument('--warmup',type=int,default=12)
p.add_argument('--repeats',type=int,default=50)
p.add_argument('--out',required=True)
a=p.parse_args()
torch.set_num_threads(4);torch.manual_seed(811)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
if a.backend=='vulkan':
 import torch_vulkan
 synchronize=torch_vulkan.synchronize;device=torch.device('vulkan')
 assert torch_vulkan.is_hardware_accelerated()
else:
 device=torch.device('cuda');synchronize=torch.cuda.synchronize
 assert torch.cuda.get_device_name(0)=='NVIDIA GeForce RTX 2080 Ti'
from torch.nn.attention import sdpa_kernel,SDPBackend
@contextlib.contextmanager
def context():
 if a.impl=='math' and a.backend=='cuda':
  with sdpa_kernel(SDPBackend.MATH):yield
 else:yield
shape=(a.batch,a.heads,a.seq,a.dim)
data=[torch.randn(shape,device='cpu')*.13 for _ in range(3)]
inputs=[x.to(device).requires_grad_() for x in data]
gradient=torch.randn(shape,device='cpu')*.1
grad=gradient.to(device)
def step():
 for t in inputs:t.grad=None
 with context():
  y=torch.nn.functional.scaled_dot_product_attention(*inputs,dropout_p=0.0)
 y.backward(grad)
 return y
cpu=[x.detach().requires_grad_() for x in data]
cy=torch.nn.functional.scaled_dot_product_attention(*cpu,dropout_p=0.0);cy.backward(gradient)
y=step();synchronize()
torch.testing.assert_close(y.detach().cpu(),cy.detach(),rtol=2e-3,atol=2e-4)
for x,c in zip(inputs,cpu):torch.testing.assert_close(x.grad.cpu(),c.grad,rtol=3e-3,atol=3e-4)
for _ in range(a.warmup):step()
synchronize()
if a.backend=='vulkan':torch_vulkan.reset_execution_stats()
ms=[]
for _ in range(a.repeats):
 synchronize();start=time.perf_counter();step();synchronize()
 ms.append((time.perf_counter()-start)*1000)
report={'backend':a.backend,'implementation':a.impl,'shape':shape,'dtype':'float32','tf32':False,'warmup':a.warmup,
'repeats':a.repeats,'median_ms':statistics.median(ms),'p10_ms':sorted(ms)[int(.1*(len(ms)-1))],
'p90_ms':sorted(ms)[int(.9*(len(ms)-1))],'samples_ms':ms,'cpu_parity_passed':True,
'cpu_fallback_calls':torch_vulkan.execution_stats()['cpu_fallback_calls'] if a.backend=='vulkan' else None}
assert report['cpu_fallback_calls'] in (None,0)
dest=Path(a.out);dest.parent.mkdir(parents=True,exist_ok=True)
dest.write_text(json.dumps(report,indent=2)+'\n')
print('RESULT',a.backend,a.impl,a.batch,round(report['median_ms'],3),'ms',flush=True)
