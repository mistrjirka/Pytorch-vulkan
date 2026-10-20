"""Inspect cuDNN-selected CUDA kernel families for exact PyTorch convolutions.

Profiling is performed *after* cuDNN algorithm benchmarking/warmup.
CUDA TF32 is disabled; exactly the same device and FP32 inputs as the
Vulkan comparison are used.
"""
import argparse, collections, json, os
import torch
import torch.nn.functional as F
from torch.profiler import profile, ProfilerActivity
from probe_winograd_forward import CASES

ap=argparse.ArgumentParser()
ap.add_argument("--cases", default="groups4_wide,grouped_large,groups8_mid,medium64,large64")
ap.add_argument("--benchmark", type=int, default=1)
ap.add_argument("--out", default=None)
args=ap.parse_args()
torch.backends.cudnn.benchmark=bool(args.benchmark)
torch.backends.cudnn.allow_tf32=False
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.deterministic=False
print("Torch",torch.__version__,"CUDA",torch.version.cuda,"cuDNN",torch.backends.cudnn.version(), "GPU",torch.cuda.get_device_name())
data=[]
for name,shape,cout,groups in CASES:
 if name not in set(args.cases.split(",")):continue
 g=torch.Generator().manual_seed(7102)
 x=torch.randn(shape,generator=g,device="cpu").to("cuda")
 w=(torch.randn((cout,shape[1]//groups,3,3),generator=g)/((shape[1]//groups*9)**0.5)).to("cuda")
 b=torch.randn(cout,generator=g).to("cuda")
 def run():return F.conv2d(x,w,b,padding=1,groups=groups)
 for _ in range(8):run()
 torch.cuda.synchronize()
 with profile(activities=[ProfilerActivity.CPU,ProfilerActivity.CUDA],record_shapes=False,with_stack=False) as prof:
  for _ in range(4):run()
  torch.cuda.synchronize()
 kernels=collections.defaultdict(lambda: {"count":0,"sum_us":0})
 for e in prof.events():
  t=str(e.device_type)
  if "CUDA" in t and "CPU" not in t:
   x=kernels[e.name]
   x["count"]+=1
   x["sum_us"]+=e.device_time_total
 rows=sorted((dict(kernel=k,count=v["count"],sum_us=round(v["sum_us"],2)) for k,v in kernels.items()),key=lambda x:-x["sum_us"])
 print("\\n",name,"algorithm_benchmark",bool(args.benchmark))
 for row in rows[:16]:print(row["count"],round(row["sum_us"],2),"us",row["kernel"][:240])
 data.append(dict(case=name,benchmark=bool(args.benchmark),kernels=rows))
if args.out:
 open(args.out,"w").write(json.dumps(data,indent=2)+"\\n")
