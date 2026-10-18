"""Matched native BatchNorm FP32 forward/backward, including CUDA kernel names."""
import argparse
import json
import os
from pathlib import Path
import statistics
import time
import torch

p=argparse.ArgumentParser()
p.add_argument('--backend',choices=['cuda','vulkan'],required=True)
p.add_argument('--out',type=Path,required=True)
a=p.parse_args()
torch.set_num_threads(8)
if a.backend=='vulkan':
    import torch_vulkan
    sync=torch_vulkan.synchronize
    gpu=torch_vulkan.device_name()
else:
    sync=torch.cuda.synchronize
    gpu=torch.cuda.get_device_name()
torch.manual_seed(2026)
rows=[]
for shape in [(32,128),(2,32,33),(1,32,64,64),(2,64,64,64),(2,7,3,5,9)]:
    xc=torch.randn(shape);wc=torch.randn(shape[1]);bc=torch.randn(shape[1]);gc=torch.randn(shape)
    x,w,b,g=[t.to(a.backend) for t in (xc,wc,bc,gc)]
    rm=torch.zeros_like(w); rv=torch.ones_like(w)
    for training in [True,False]:
        def forward():
            return torch.ops.aten.native_batch_norm(x,w,b,rm,rv,training,.1,1e-5)
        rmc,rvc=rm.cpu().clone(),rv.cpu().clone()
        reference=torch.ops.aten.native_batch_norm(xc,wc,bc,rmc,rvc,training,.1,1e-5)
        y,mean,inv=forward()
        def backward():
            return torch.ops.aten.native_batch_norm_backward(g,x,w,rm,rv,mean,inv,training,1e-5,[True]*3)
        reference_grad=torch.ops.aten.native_batch_norm_backward(gc,xc,wc,rmc,rvc,reference[1],reference[2],training,1e-5,[True]*3)
        errors=[]
        for index,(actual,expected) in enumerate(zip((y,mean,inv,*backward()),(*reference,*reference_grad))):
            # CUDA returns channel-sized saved tensors in eval; CPU returns empty.
            if not training and index in (1,2):
                errors.append(None);continue
            actual=actual.cpu()
            torch.testing.assert_close(actual,expected,atol=5e-4,rtol=5e-4)
            errors.append((actual-expected).abs().max().item() if actual.numel() else 0.)
        for phase,fn in [('forward',forward),('backward',backward)]:
            for _ in range(5): fn(); sync()
            timings_before=torch_vulkan.gpu_timing_stats() if a.backend=='vulkan' else {}
            times=[]
            for _ in range(15):
                sync(); t=time.perf_counter(); fn(); sync()
                times.append((time.perf_counter()-t)*1000)
            timings_after=torch_vulkan.gpu_timing_stats() if a.backend=='vulkan' else {}
            gpu_times={k:{metric:(v-timings_before.get(k,{}).get(metric,0))/15 for metric,v in data.items()} for k,data in timings_after.items()}
            activities=[torch.profiler.ProfilerActivity.CPU]
            if a.backend=='cuda': activities.append(torch.profiler.ProfilerActivity.CUDA)
            with torch.profiler.profile(activities=activities) as prof:
                for _ in range(3): fn(); sync()
                sync()
            events=[dict(name=e.key,calls=e.count/3,self_cpu_ms=e.self_cpu_time_total/3000,
                         device_ms=e.device_time_total/3000) for e in prof.key_averages()]
            rows.append(dict(shape=shape,training=training,phase=phase,median_ms=statistics.median(times),
                             range_ms=[min(times),max(times)],operators=events,gpu_timing_per_call=gpu_times,
                             max_abs_errors_forward_saved_gradients=errors))
report=dict(backend=a.backend,gpu=gpu,torch=torch.__version__,dtype='float32',warmup=5,repeats=15,
            env={k:v for k,v in os.environ.items() if k.startswith('TORCH_VULKAN_')},results=rows)
a.out.parent.mkdir(parents=True,exist_ok=True)
a.out.write_text(json.dumps(report,indent=2)+'\n')
for r in rows: print(r['shape'],r['training'],r['phase'],round(r['median_ms'],4))
