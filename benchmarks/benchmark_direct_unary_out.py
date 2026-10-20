"""Compare old functional+copy and generic direct-output unary ATen paths.

TORCH_VULKAN_DISABLE_DIRECT_UNARY_OUT=1 selects the pre-existing route.
Both benchmark variants execute the *same* torch.out calls and reference
the same heterogeneous unary shapes. No generated model-specific shaders.
"""
from __future__ import annotations
import argparse
import json
import statistics
import os
from pathlib import Path
import time
import torch

p=argparse.ArgumentParser()
p.add_argument('--warmup',type=int,default=10)
p.add_argument('--steps',type=int,default=70)
p.add_argument('--outputs',type=int,default=200)
p.add_argument('--elements',type=int,default=4096)
p.add_argument('--out',required=True)
p.add_argument('--family',choices=['diffusion','math'],default='diffusion')
p.add_argument('--backend',choices=['vulkan','cuda'],default='vulkan')
a=p.parse_args()
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
gen=torch.Generator().manual_seed(4283)
x=(torch.randn(a.elements,generator=gen)*.1+.5).to(device)
outs=[torch.empty_like(x) for _ in range(a.outputs)]
ops=(['sin','cos','exp','sqrt','rsqrt','tanh','neg','silu']
     if a.family=='diffusion' else
     ['reciprocal','log','log1p','expm1','floor','ceil','trunc','round','sign',
      'log10','log2','exp2','sinh','cosh','tan','asin','acos','atan',
      'asinh','acosh','atanh','erf','erfc'])
fn=[]
for i in range(a.outputs):
 op=ops[i%len(ops)]
 f=(lambda input,out,op=op: torch.ops.aten.silu.out(input,out=out)) if op=='silu' else \
   (lambda input,out,op=op: getattr(torch,op)(input,out=out))
 fn.append(f)
# sqrt/rsqrt are defined on positive input; negatives yield NaNs and are
# compared with equal_nan=True against the exact CPU behavior.
xp=x.cpu()
for i,f in enumerate(fn):
 if i%max(1,a.outputs//9)==0:
  try:
   ref=f(xp,torch.empty_like(xp))
   actual=f(x,outs[i])
   torch.testing.assert_close(actual.cpu(),ref,equal_nan=True,rtol=6e-5,atol=6e-5)
  except Exception as exc:
   raise RuntimeError(f'Numerical check failed for output {i}: {exc}') from exc

def run():
 for i,f in enumerate(fn):
  f(x,outs[i])

for _ in range(a.warmup):run()
sync()
base=torch_vulkan.queue_stats() if a.backend=='vulkan' else {}
cache_before=torch_vulkan.cache_stats() if a.backend=='vulkan' else {}
timings=[]
for _ in range(a.steps):
 start=time.perf_counter()
 run()
 sync()
 timings.append((time.perf_counter()-start)*1000)
after=torch_vulkan.queue_stats() if a.backend=='vulkan' else {}
cache_after=torch_vulkan.cache_stats() if a.backend=='vulkan' else {}
percentiles=sorted(timings)
r={
 'device':device_name,'backend':a.backend,
 'family':a.family,
 'mode':'native-cuda' if a.backend=='cuda' else ('legacy-functional-copy' if os.environ.get('TORCH_VULKAN_DISABLE_DIRECT_UNARY_OUT')=='1' else 'direct-output'),
 'output_operations':a.outputs,
 'elements':a.elements,
 'warmup':a.warmup,'steps':a.steps,
 'median_ms':statistics.median(timings),
 'p10_ms':percentiles[int(.1*(a.steps-1))],
 'p90_ms':percentiles[int(.9*(a.steps-1))],
 'times_ms':timings,
 'shader_dispatches_per_iteration':(after['recorded_ops']-base['recorded_ops'])/a.steps if a.backend=='vulkan' else None,
 'cache_deltas':{k:cache_after[k]-cache_before[k] for k in ['algo_hits','algo_misses','algo_evictions']} if a.backend=='vulkan' else {},
 'cpu_fallback_calls':torch_vulkan.execution_stats()['cpu_fallback_calls'] if a.backend=='vulkan' else None,
 'operations':ops
}
path=Path(a.out);path.parent.mkdir(parents=True,exist_ok=True)
path.write_text(json.dumps(r,indent=2)+'\n')
print('RESULT',r['mode'],r['device'],round(r['median_ms'],4),'ms',r['shader_dispatches_per_iteration'],'dispatches/step',r['cpu_fallback_calls'],'CPU fallbacks',flush=True)
