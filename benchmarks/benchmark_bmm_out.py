"""Matched bmm.out A/B with reused outputs, same GPU work and semantics."""
import argparse,json,os,statistics,time
from pathlib import Path
import torch

arg=argparse.ArgumentParser()
arg.add_argument('--output',required=True)
arg.add_argument('--backend',choices=['vulkan','cuda'],default='vulkan')
arg.add_argument('--kind',choices=['bmm','mm'],default='bmm')
arg.add_argument('--warmup',type=int,default=12)
arg.add_argument('--steps',type=int,default=80)
arg.add_argument('--outputs',type=int,default=128)
a=arg.parse_args()
torch.set_num_threads(4)
if a.backend=='vulkan':
 import torch_vulkan
 device=torch.device('vulkan')
 sync=torch_vulkan.synchronize
 device_name=torch_vulkan.device_name()
else:
 device=torch.device('cuda')
 sync=torch.cuda.synchronize
 device_name=torch.cuda.get_device_name(0)
torch.manual_seed(1643)
x=torch.randn((3,32,48) if a.kind=='bmm' else (32,48)).to(device)
y=torch.randn((3,48,24) if a.kind=='bmm' else (48,24)).to(device)
outs=[torch.empty((3,32,24) if a.kind=='bmm' else (32,24),device=device) for _ in range(a.outputs)]
fn=torch.bmm if a.kind=='bmm' else torch.mm
ref=fn(x.cpu(),y.cpu())
for z in (outs[0],outs[-1]):
    fn(x,y,out=z)
    torch.testing.assert_close(z.cpu(),ref,rtol=2e-4,atol=2e-4)
def work():
    for z in outs: fn(x,y,out=z)
for _ in range(a.warmup):work()
sync()
before=torch_vulkan.queue_stats() if a.backend=='vulkan' else {}
ms=[]
for _ in range(a.steps):
    start=time.perf_counter();work();sync()
    ms.append((time.perf_counter()-start)*1000)
after=torch_vulkan.queue_stats() if a.backend=='vulkan' else {}
r={'mode':'native-cuda' if a.backend=='cuda' else ('legacy-functional-copy' if os.environ.get(('TORCH_VULKAN_DISABLE_DIRECT_BMM_OUT' if a.kind=='bmm' else 'TORCH_VULKAN_DISABLE_DIRECT_MM_OUT'))=='1' else 'direct-'+a.kind+'-out'),
    'backend':a.backend,'kind':a.kind,
    'device':device_name,'warmup':a.warmup,'steps':a.steps,'outputs':a.outputs,
    'median_ms':statistics.median(ms),
    'times_ms':ms,'dispatches_per_iteration':(after['recorded_ops']-before['recorded_ops'])/a.steps if a.backend=='vulkan' else None,
    'cpu_fallback':torch_vulkan.execution_stats()['cpu_fallback_calls'] if a.backend=='vulkan' else None}
p=Path(a.output);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(r,indent=2)+'\n')
print('RESULT',r['device'],r['mode'],round(r['median_ms'],4),'ms',r['dispatches_per_iteration'],'GPU dispatches',flush=True)
