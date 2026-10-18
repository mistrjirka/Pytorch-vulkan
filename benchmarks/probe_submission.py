"""Split warmed native-pool recording and synchronization wall time for Nsight."""
import json
import os
import statistics
import time
import torch
import torch_vulkan

if any(os.environ.get(k) != '1' for k in ['TORCH_VULKAN_GPU_RESIDENT','TORCH_VULKAN_DEFER_SUBMISSIONS']):
 raise RuntimeError('This split requires GPU_RESIDENT=1 and DEFER_SUBMISSIONS=1')
torch.set_num_threads(8)
x=torch.randn(1,32,64,64).to('vulkan')
record=[];wait=[];total=[]
for i in range(60):
 start=time.perf_counter()
 result=torch.ops.aten.max_pool2d_with_indices(x,[2,2],[2,2],[0,0],[1,1],False)
 submitted=time.perf_counter()
 torch_vulkan.synchronize()
 end=time.perf_counter()
 if i>=10:
  record.append((submitted-start)*1000);wait.append((end-submitted)*1000);total.append((end-start)*1000)
 del result
print(json.dumps(dict(gpu=torch_vulkan.device_name(),record_ms=statistics.median(record),
                     synchronize_ms=statistics.median(wait),total_ms=statistics.median(total),range_ms=[min(total),max(total)],
                     samples=50,warmup=10,torch=torch.__version__,
                     fence_pool=os.environ.get('TORCH_VULKAN_DISABLE_FENCE_POOL')!='1',cache=torch_vulkan.cache_stats()),indent=2))
