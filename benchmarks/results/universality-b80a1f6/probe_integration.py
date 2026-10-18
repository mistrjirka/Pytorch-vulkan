import argparse,json,sys,copy
from pathlib import Path
import torch
from torch.nn import functional as F
p=argparse.ArgumentParser();p.add_argument('--backend',required=True);p.add_argument('--out',required=True);a=p.parse_args();torch.set_num_threads(4);torch.manual_seed(99)
if a.backend=='vulkan':import torch_vulkan as mod
else:mod=torch.cuda
d={'backend':a.backend,'torch':torch.__version__,'cases':{}}
def run(name,fn):
 print('BEGIN',name,file=sys.stderr,flush=True)
 try:d['cases'][name]={'status':'ok','result':fn()};mod.synchronize()
 except Exception as e:d['cases'][name]={'status':'error','type':type(e).__name__,'error':str(e)[:1800]}
 Path(a.out).write_text(json.dumps(d,indent=2)+'\n');print('END',name,d['cases'][name]['status'],file=sys.stderr,flush=True)
def layout():
 x=torch.empty((2,3,5,7),device=a.backend,memory_format=torch.channels_last)
 return {'strides':x.stride(),'channels_last':x.is_contiguous(memory_format=torch.channels_last)}
def device_index():return str(torch.empty(3,device=a.backend+':17').device)
def view_version():
 x=torch.randn(4,4).to(a.backend).requires_grad_();base=x*2;v=base.view(16);loss=(v*v).sum();v0=[base._version,v._version];base.add_(1);v1=[base._version,v._version]
 try:loss.backward();rejected=False
 except RuntimeError as e:
  if 'modified by an inplace' not in str(e):raise
  rejected=True
 return {'versions_before':v0,'versions_after':v1,'modification_rejected':rejected}
def view_bounds():
 x=torch.empty(4,device=a.backend);torch.as_strided(x,(100,),(1,));return 'accepted out-of-bounds view without reading it'
def embedding_flags():
 w=torch.randn(7,4,requires_grad=True);v=w.detach().to(a.backend).requires_grad_();i=torch.tensor([0,1,1,3]);x=F.embedding(i,w,padding_idx=0,scale_grad_by_freq=True);y=F.embedding(i.to(a.backend),v,padding_idx=0,scale_grad_by_freq=True);x.sum().backward();y.backward(torch.ones_like(y));torch.testing.assert_close(y.cpu(),x);torch.testing.assert_close(v.grad.cpu(),w.grad);return 'padding/frequency forward and gradients match CPU'
def embedding_index_precision():
 n=2**24+2;w=torch.zeros(n,1);w[2**24]=456;w[2**24+1]=123;i=torch.tensor([2**24+1]);y=F.embedding(i.to(a.backend),w.to(a.backend)).cpu();return {'expected':123.,'actual':y.item(),'matches':y.item()==123.}
def mixed_mm():
 x=torch.arange(12,dtype=torch.float32).reshape(3,4).to(a.backend);y=torch.arange(8,dtype=torch.float64).reshape(4,2).to(a.backend);z=x@y;return {'accepted_mismatched_dtypes':True,'output':z.cpu().tolist()}
def stream():return str(torch.Stream(device=a.backend))
def event():return str(torch.Event(device=a.backend))
def compilation():
 def fn(x):return x.sin()+x
 x=torch.randn(4,7).to(a.backend);compiled=torch.compile(fn,fullgraph=True);y=compiled(x);torch.testing.assert_close(y.cpu(),fn(x.cpu()),rtol=1e-4,atol=1e-5);return 'default Inductor fullgraph output matches CPU'
for n,f in [('channels_last_factory',layout),('invalid_device_index',device_index),('view_version',view_version),('view_bounds',view_bounds),('embedding_flags',embedding_flags),('embedding_large_index',embedding_index_precision),('mixed_dtype_mm',mixed_mm),('stream',stream),('event',event),('compile',compilation)]:run(n,f)
print(json.dumps(d,indent=2))
