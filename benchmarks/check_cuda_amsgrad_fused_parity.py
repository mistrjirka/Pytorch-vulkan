"""Numeric parity of CUDA fused and unfused AMSGrad AdamW on identical tensors."""
import json,torch
from pathlib import Path
torch.manual_seed(4217);torch.set_num_threads(4)
assert torch.cuda.get_device_name(0)=='NVIDIA GeForce RTX 2080 Ti'
shapes=[(32,32),(256,),(64,128),(3,3,64,64),(8,1,5,7)]
base=[torch.randn(shape,dtype=torch.float32)*.2 for shape in shapes]
grad=[torch.randn_like(x)*.001 for x in base]
data={}
allparam={}
for fused in (False,True):
 p=[torch.nn.Parameter(x.cuda().clone()) for x in base]
 opt=torch.optim.AdamW(p,lr=1e-4,weight_decay=1e-5,amsgrad=True,fused=fused,foreach=False)
 for iteration in range(40):
  for x,g in zip(p,grad):x.grad=g.cuda()
  opt.step()
 torch.cuda.synchronize()
 allparam[fused]=(p,opt)
ref,ropt=allparam[False];fused_params,fopt=allparam[True]
mx=0
for x,y in zip(ref,fused_params):
 torch.testing.assert_close(x,y,atol=2e-6,rtol=1e-5)
 mx=max(mx,float((x-y).abs().max()))
 for field in ('exp_avg','exp_avg_sq','max_exp_avg_sq'):
  s=ropt.state[x][field];t=fopt.state[y][field]
  torch.testing.assert_close(s,t,atol=2e-6,rtol=1e-5)
  mx=max(mx,float((s-t).abs().max()))
data=dict(steps=40,shapes=shapes,max_abs_difference=mx,optimizer='torch.optim.AdamW(amsgrad=True,foreach=False) fused=False versus fused=True',passed=True)
out=Path('benchmarks/results/optimizer-amsgrad-ab-2026-10-20/cuda-fused-parity.json')
out.write_text(json.dumps(data,indent=2)+'\n')
print('AMSGRAD_PARITY',json.dumps(data),flush=True)
