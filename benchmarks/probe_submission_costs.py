"""Microbenchmark for Vulkan eager vs deferred command submissions.

Synthetic GPU-only add/ReLU dependency chain. No CPU fallback inside the
measured loop; final readback verifies all GPU results and synchronizes.
This is intentionally not representative of a complete network.
"""
import argparse, json, statistics, time
from pathlib import Path
import torch
import torch_vulkan

p = argparse.ArgumentParser()
p.add_argument("--steps",type=int,default=70)
p.add_argument("--elements",type=int,default=4096)
p.add_argument("--repeats",type=int,default=4)
p.add_argument("--out",type=Path)
args=p.parse_args()
assert args.steps>0 and args.elements>0
print('DEVICE',torch_vulkan.device_name(), 'steps',args.steps,'elements',args.elements)
v=torch.full((args.elements,),0.03,device="vulkan")
results=[]
for rep in range(args.repeats):
    x=torch.zeros(args.elements,device="vulkan")
    before = torch_vulkan.queue_stats()
    t0=time.perf_counter()
    for i in range(args.steps):
        x=torch.relu(x+v)
    gpu_recorded_ms=(time.perf_counter()-t0)*1000
    actual=x.cpu()
    total_ms=(time.perf_counter()-t0)*1000
    torch.testing.assert_close(actual,torch.full_like(actual,args.steps*0.03),atol=3e-4,rtol=3e-4)
    after=torch_vulkan.queue_stats()
    results.append({"record_ms":round(gpu_recorded_ms,4),
                    "final_host_sync_ms":round(total_ms-gpu_recorded_ms,4),
                    "total_ms":round(total_ms,4),
                    "submissions":after["submissions"]-before["submissions"],
                    "recorded_ops":after["recorded_ops"]-before["recorded_ops"]})
med=statistics.median(r["total_ms"] for r in results)
out={"device":torch_vulkan.device_name(),"elements":args.elements,
    "rounds":args.repeats,"steps":args.steps,
    "deferred": __import__('os').environ.get("TORCH_VULKAN_DEFER_SUBMISSIONS","0"),
    "host_visible_max_bytes":__import__('os').environ.get("TORCH_VULKAN_HOST_VISIBLE_MAX_BYTES","0"),
    "median_ms":round(med,3),"samples":results,"cache":torch_vulkan.cache_stats()}
print(json.dumps(out,indent=2))
if args.out:args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(out,indent=2)+"\n")
