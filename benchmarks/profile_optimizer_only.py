"""Optimizer-only comparison using exactly the same heterogeneous MyNet tensors.

This is a measurement-only harness, not a replacement for torch.optim.
Standard Adam, standard AdamW and opt-in native Vulkan AdamW have distinct
semantics. Compare same-optimizer variants for fair speedup; never compare
Adam versus AdamW as a numerical equivalence test.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import statistics
import sys
import time
from pathlib import Path
import torch

parser=argparse.ArgumentParser()
parser.add_argument('--backend',choices=['vulkan','cuda'],required=True)
parser.add_argument('--optimizer',choices=['adam','adamw','native_adamw'],required=True)
parser.add_argument('--foreach',action='store_true',help='Use standard torch.optim foreach implementation')
parser.add_argument('--fused',action='store_true',help='Request native PyTorch fused=True implementation')
parser.add_argument('--warmup',type=int,default=12)
parser.add_argument('--repeats',type=int,default=100)
parser.add_argument('--profile-count',type=int,default=5)
parser.add_argument('--out',required=True)
args=parser.parse_args()
torch.set_num_threads(4)
torch.manual_seed(20261019)
src=Path(os.environ.get('NIS_LAB02_SOURCE',
    str(Path.home()/'programovani/magistr/nis/lab02_cnn')))
if not (src/'models.py').is_file():
 src=Path(__file__).resolve().parent/'nis-lab02-reference'
if not (src/'models.py').is_file():
 raise FileNotFoundError('Set NIS_LAB02_SOURCE to a local directory containing Lab02 models.py')
sys.path.insert(0,str(src))
from models import MyNet
if args.backend=='vulkan':
 import torch_vulkan
 device=torch.device('vulkan')
 sync=torch_vulkan.synchronize
 devname=torch_vulkan.device_name()
 assert torch_vulkan.is_hardware_accelerated()
else:
 device=torch.device('cuda')
 sync=torch.cuda.synchronize
 devname=torch.cuda.get_device_name(0)
 torch.backends.cudnn.allow_tf32=False
 torch.backends.cuda.matmul.allow_tf32=False

model=MyNet(3,3,5)
params=[p for p in model.parameters() if p.requires_grad]
digest=hashlib.sha256()
for p in params:
 digest.update(p.detach().contiguous().numpy().tobytes())
digest=digest.hexdigest()
params=[torch.nn.Parameter(p.detach().to(device)) for p in params]
grad_gen=torch.Generator().manual_seed(2903)
grad_cpu=[torch.randn(p.shape,generator=grad_gen,dtype=torch.float32)*0.001 for p in params]
grads=[g.to(device) for g in grad_cpu]
for p,g in zip(params,grads):p.grad=g
if args.optimizer=='adam':
 opt=torch.optim.Adam(params,lr=1e-4,weight_decay=1e-5,foreach=(None if args.fused else args.foreach),fused=(True if args.fused else None))
elif args.optimizer=='adamw':
 opt=torch.optim.AdamW(params,lr=1e-4,weight_decay=1e-5,foreach=(None if args.fused else args.foreach),fused=(True if args.fused else None))
else:
 assert args.backend=='vulkan'
 from torch_vulkan.optim import AdamW as NativeAdamW
 opt=NativeAdamW(params,lr=1e-4,weight_decay=1e-5)

def step():
 opt.step()

for _ in range(args.warmup):step()
sync()
times=[]
for _ in range(args.repeats):
 start=time.perf_counter()
 step();sync()
 times.append((time.perf_counter()-start)*1000)
ordered=sorted(times)
def pct(p):
 t=(len(ordered)-1)*p
 lo=int(t);hi=min(lo+1,len(ordered)-1)
 return ordered[lo]*(hi-t)+ordered[hi]*(t-lo)
r=dict(device=devname,backend=args.backend,optimizer=args.optimizer,foreach=args.foreach,fused=args.fused,
    parameter_count=len(params),parameter_numel=sum(p.numel() for p in params),
    tensor_shapes=[list(p.shape) for p in params],params_sha256=digest,
    grad_sha256=hashlib.sha256(b''.join(x.numpy().tobytes() for x in grad_cpu)).hexdigest(),
    torch=torch.__version__,warmup=args.warmup,repeats=args.repeats,
    median_ms=statistics.median(times),p10_ms=pct(.1),p90_ms=pct(.9),ms=times)
if args.backend=='vulkan':
 r['execution_stats']=torch_vulkan.execution_stats()
 r['cache_stats']=torch_vulkan.cache_stats()
 # Profiling may be off; separate diagnostic with GPU timestamps enabled
 # lets us measure true shader time without biasing the wall-time series.
 try:
  before=torch_vulkan.gpu_timing_stats()
  for _ in range(args.profile_count):step();sync()
  after=torch_vulkan.gpu_timing_stats()
  gpu={}
  for key,after_item in after.items():
   prior=before.get(key,{})
   diffs={k:float(v)-float(prior.get(k,0)) for k,v in after_item.items() if isinstance(v,(float,int))}
   if diffs.get('samples',0)>0:gpu[key]=diffs
  r['gpu_timings']=gpu
 except Exception as e:r['gpu_timing_error']=str(e)
else:
 try:
  from torch.profiler import profile,ProfilerActivity
  with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA]) as p:
   for _ in range(args.profile_count):step()
   sync()
  r['profiler_ops']=[dict(name=e.key,calls=e.count,self_cpu_ms=e.self_cpu_time_total/1000,
      self_gpu_ms=getattr(e,'self_device_time_total',0)/1000)
    for e in sorted(p.key_averages(),key=lambda e:e.self_cpu_time_total,reverse=True)[:35]]
 except Exception as e:r['profile_error']=str(e)
Path(args.out).parent.mkdir(parents=True,exist_ok=True)
Path(args.out).write_text(json.dumps(r,indent=2,default=str)+'\n')
print('RESULT',args.backend,args.optimizer,'params',len(params),'elements',sum(p.numel() for p in params),
      'median_ms',round(r['median_ms'],5),'p10_ms',round(r['p10_ms'],5),'p90_ms',round(r['p90_ms'],5))
print('TOP GPU',[(k,round(v.get('shader_ms',0)/max(1,args.profile_count),4),v.get('samples',0)/max(1,args.profile_count))
       for k,v in sorted(r.get('gpu_timings',{}).items(),key=lambda x:x[1].get('shader_ms',0),reverse=True)[:10]])
