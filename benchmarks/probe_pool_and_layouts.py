"""Matched pooling, concatenation and broadcast-layout benchmarks with parity checks."""
import argparse
import json
import os
from pathlib import Path
import statistics
import time
import torch
import torch.nn.functional as F

p=argparse.ArgumentParser()
p.add_argument('--backend',choices=['cuda','vulkan'],required=True)
p.add_argument('--out',type=Path,required=True)
a=p.parse_args()
torch.set_num_threads(8)
torch.manual_seed(2026)
if a.backend=='vulkan':
 import torch_vulkan
 sync=torch_vulkan.synchronize;gpu=torch_vulkan.device_name()
else:
 sync=torch.cuda.synchronize;gpu=torch.cuda.get_device_name()
rows=[]
def measure(name,shape,fn,reference):
 result=fn();values=result if isinstance(result,tuple) else (result,)
 refs=reference if isinstance(reference,tuple) else (reference,)
 errors=[]
 for got,ref in zip(values,refs):
  cpu=got.cpu();torch.testing.assert_close(cpu,ref,rtol=2e-6,atol=2e-6)
  errors.append((cpu-ref).abs().max().item() if cpu.numel() else 0)
 del result,values
 for _ in range(5):fn();sync()
 before=torch_vulkan.gpu_timing_stats() if a.backend=='vulkan' else {}
 samples=[]
 for _ in range(15):
  sync();start=time.perf_counter();fn();sync();samples.append((time.perf_counter()-start)*1000)
 after=torch_vulkan.gpu_timing_stats() if a.backend=='vulkan' else {}
 timings={k:{m:(v-before.get(k,{}).get(m,0))/15 for m,v in data.items()} for k,data in after.items()}
 activities=[torch.profiler.ProfilerActivity.CPU]
 if a.backend=='cuda':activities.append(torch.profiler.ProfilerActivity.CUDA)
 with torch.profiler.profile(activities=activities) as prof:
  for _ in range(3):fn();sync()
 events=[dict(name=e.key,calls=e.count/3,self_cpu_ms=e.self_cpu_time_total/3000,device_ms=e.device_time_total/3000) for e in prof.key_averages()]
 rows.append(dict(name=name,shape=shape,median_ms=statistics.median(samples),range_ms=[min(samples),max(samples)],
                  max_abs_errors=errors,gpu_timing_per_call=timings,operators=events))
for shape,k,s,pad,dil,ceil in [
 ((1,32,64,64),[2,2],[2,2],[0,0],[1,1],False),
 ((2,64,64,64),[3,3],[1,1],[1,1],[1,1],False),
 ((2,7,19,23),[3,5],[2,3],[1,2],[2,1],True),
 ((3,7,9),[2,3],[2,2],[0,1],[1,1],True),
]:
 x=torch.randn(shape);y,indices=F.max_pool2d(x,k,s,pad,dil,ceil,return_indices=True);g=torch.randn(y.shape)
 xv=x.to(a.backend);gv=g.to(a.backend);iv=indices.to(a.backend)
 measure('max_pool2d_forward',shape,lambda:torch.ops.aten.max_pool2d_with_indices(xv,k,s,pad,dil,ceil),(y,indices))
 ref=torch.ops.aten.max_pool2d_with_indices_backward(g,x,k,s,pad,dil,ceil,indices)
 measure('max_pool2d_backward',shape,lambda:torch.ops.aten.max_pool2d_with_indices_backward(gv,xv,k,s,pad,dil,ceil,iv),ref)
for shape in [(1,64,32,32),(2,32,64,64)]:
 x=torch.randn(shape);y=torch.randn(shape)
 xv=x.to(a.backend);yv=y.to(a.backend)
 measure('cat_channels',shape,lambda:torch.cat([xv,yv],1),torch.cat([x,y],1))
 # Slice channels from a larger allocation: a common concat-backward view.
 storage=torch.randn(shape[0],shape[1]*2,shape[2],shape[3]);x=storage[:,shape[1]:]
 xs=storage.to(a.backend);xv=xs[:,shape[1]:]
 b=torch.randn(1,shape[1],1,1);bv=b.to(a.backend)
 measure('add_offset_broadcast',shape,lambda:torch.add(xv,bv,alpha=.25),x+b*.25)
report=dict(backend=a.backend,gpu=gpu,torch=torch.__version__,dtype='float32',warmup=5,repeats=15,
 env={k:v for k,v in os.environ.items() if k.startswith('TORCH_VULKAN_')},results=rows)
a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2)+'\n')
for r in rows:print(r['name'],r['shape'],round(r['median_ms'],4))
