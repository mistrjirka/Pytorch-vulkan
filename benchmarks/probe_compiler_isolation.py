"""Verify importing the Vulkan compiler does not replace CPU/CUDA codegen."""
import argparse,json
from pathlib import Path
import torch
import torch_vulkan
from torch._inductor import metrics
p=argparse.ArgumentParser();p.add_argument('--device',choices=['cpu','cuda'],required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
torch.manual_seed(71);torch.backends.cuda.matmul.allow_tf32=False
x=torch.randn(7,9,requires_grad=True);w=torch.randn(9,5,requires_grad=True)
y=x.detach().to(a.device).requires_grad_();v=w.detach().to(a.device).requires_grad_()
fn=lambda x,w:(x.sin()@w).relu().square().mean()
expected=fn(x,w);expected.backward();metrics.reset();compiled=torch.compile(fn,fullgraph=True);actual=compiled(y,v);actual.backward()
torch.testing.assert_close(actual.cpu(),expected,rtol=2e-4,atol=2e-5)
for got,ref in ((y,x),(v,w)):torch.testing.assert_close(got.grad.cpu(),ref.grad,rtol=4e-4,atol=4e-5)
assert metrics.generated_kernel_count>0
result=dict(device=a.device,torch=torch.__version__,status='ok',generated_kernel_count=metrics.generated_kernel_count)
a.out.write_text(json.dumps(result,indent=2)+'\n');print(result)
