"""Isolate buffer/descriptor reuse from GPU arithmetic, without changing models."""
import argparse,json,os,time,statistics,torch
import torch_vulkan
a=argparse.ArgumentParser()
a.add_argument('--mode',choices=['fresh','preallocated'],required=True)
a.add_argument('--ops',type=int,default=400)
a.add_argument('--n',type=int,default=40)
a.add_argument('--size',type=int,default=512)
a.add_argument('--output',required=True)
args=a.parse_args()
torch.set_num_threads(4)
seed=torch.ones(args.size,device='vulkan')
add=torch.ones(args.size,device='vulkan')
outputs=[torch.empty_like(seed) for _ in range(args.ops)] if args.mode=='preallocated' else []
def step():
 x=seed
 for i in range(args.ops):
  if args.mode=='preallocated':
   torch.add(x,add,out=outputs[i])
   x=outputs[i]
  else: x=torch.add(x,add)
 return x
for j in range(5):z=step();torch_vulkan.synchronize()
assert abs(float(z[0].cpu())-(args.ops+1))<0.002
stats_start=torch_vulkan.cache_stats()
ms=[]
for j in range(args.n):
 st=time.perf_counter()
 z=step();torch_vulkan.synchronize()
 ms.append((time.perf_counter()-st)*1000)
assert abs(float(z[0].cpu())-(args.ops+1))<0.002
stats=torch_vulkan.cache_stats()
delta={k:stats[k]-stats_start.get(k,0) for k in stats if isinstance(stats[k],int)}
r=dict(mode=args.mode,ops=args.ops,n=args.n,size=args.size,median_ms=statistics.median(ms),
   p10_ms=sorted(ms)[int(.1*len(ms))],p90_ms=sorted(ms)[int(.9*len(ms))-1],
   samples_ms=ms,cache_deltas=delta,execution=torch_vulkan.execution_stats(),
   device=torch_vulkan.device_name())
import pathlib
p=pathlib.Path(args.output);p.parent.mkdir(parents=True,exist_ok=True)
p.write_text(json.dumps(r,indent=2)+'\n')
print('RESULT',args.mode,'size',args.size,'median',round(r['median_ms'],3),'p10/p90',round(r['p10_ms'],3),round(r['p90_ms'],3),'cached',delta,'fallback',r['execution']['cpu_fallback_calls'])
