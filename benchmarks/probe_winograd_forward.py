"""General FP32 3x3 forward benchmark for native Vulkan, tuned cuDNN CUDA and ROCm MIOpen.

Use identical seeded shapes, a CPU FP64 oracle and synchronized timings.
On Vulkan, TORCH_VULKAN_CONV_ALGO=winograd forces the optional algorithm.
On CUDA, --cuda-autotune enables actual cuDNN algorithm benchmarking.
"""
import argparse
import json
import statistics
import time
from pathlib import Path
import torch
import torch.nn.functional as F

CASES = (
  ("large32", (2,32,64,64),64,1),
  ("small64", (2,64,16,19),64,1),
  ("medium64", (2,64,64,64),64,1),
  ("large64", (1,64,128,128),64,1),
  ("wide128", (1,128,64,64),128,1),
  ("large32b1", (1,32,128,128),64,1),
  ("grouped", (2,64,32,35),64,4),
  ("odd", (1,32,37,41),48,1),
  ("small32", (1,32,32,32),32,1),
  ("grouped_small", (1,32,16,17),32,4),
  ("grouped_medium", (2,64,64,64),64,4),
  ("grouped_large", (1,64,128,128),64,4),
  ("grouped_many", (1,32,32,32),32,8),
  ("grouped_two", (1,32,32,32),32,2),
  ("odd_medium", (1,32,63,65),48,1),
  ("odd_small", (1,16,17,23),24,1),
  ("tiny_dense", (1,16,8,11),24,1),
  ("short64", (1,64,8,8),64,1),
  ("groups8_mid", (2,128,64,64),128,8),
  ("groups8_large", (1,128,128,128),128,8),
  ("groups4_wide", (2,128,64,64),128,4),
  ("groups4_96", (1,96,64,64),96,4),
)

def main():
  p=argparse.ArgumentParser()
  p.add_argument("--backend",choices=["vulkan","cuda","rocm"],default="vulkan")
  p.add_argument("--cuda-autotune",action="store_true",help="allow cuDNN to benchmark convolution algorithms during warmup")
  p.add_argument("--cases",default=",".join(c[0] for c in CASES))
  p.add_argument("--warmup",type=int,default=4)
  p.add_argument("--repeats",type=int,default=14)
  p.add_argument("--out",type=Path,required=True)
  args=p.parse_args()
  if args.backend=="vulkan":
    import torch_vulkan
    device="vulkan"
    sync=torch_vulkan.synchronize
  else:
    assert torch.cuda.is_available(), "CUDA/HIP backend is unavailable in this environment"
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=args.cuda_autotune
    device="cuda"  # ROCm PyTorch intentionally exposes the CUDA device API
    sync=torch.cuda.synchronize
  if args.backend=="rocm":
    assert torch.version.hip, "--backend rocm requires ROCm PyTorch"
  if args.backend=="cuda":
    assert torch.version.cuda and not torch.version.hip, "--backend cuda requires NVIDIA CUDA PyTorch"
  selected=set(args.cases.split(","))
  results=[]
  for name,shape,cout,groups in CASES:
    if name not in selected:continue
    gen=torch.Generator().manual_seed(7102)
    x=torch.randn(shape,generator=gen)
    w=torch.randn((cout,shape[1]//groups,3,3),generator=gen)/((shape[1]//groups*9)**0.5)
    bias=torch.randn(cout,generator=gen)
    expect=F.conv2d(x.double(),w.double(),bias.double(),padding=1,groups=groups).float()
    xv,wv,bv=[t.to(device) for t in (x,w,bias)]
    def fn():return F.conv2d(xv,wv,bv,padding=1,groups=groups)
    got=fn()
    sync()
    torch.testing.assert_close(got.cpu(),expect,atol=3e-4,rtol=1e-3)
    before=torch_vulkan.execution_stats().get("cpu_fallback_calls",0) if args.backend=="vulkan" else 0
    for _ in range(args.warmup):fn()
    sync()
    samples=[]
    for _ in range(args.repeats):
      t0=time.perf_counter()
      v=fn()
      sync()
      samples.append((time.perf_counter()-t0)*1000)
    assert args.backend!="vulkan" or before==torch_vulkan.execution_stats().get("cpu_fallback_calls",0)
    d=dict(name=name,shape=shape,cout=cout,groups=groups,backend=args.backend,median_ms=statistics.median(samples),samples=samples)
    print(f"{name:16s} {d['median_ms']:.4f} ms",flush=True)
    results.append(d)
  args.out.parent.mkdir(parents=True,exist_ok=True)
  args.out.write_text(json.dumps(results,indent=2)+"\n")

if __name__=="__main__":
  main()
