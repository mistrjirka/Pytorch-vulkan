"""Forward and backward FP32 log-softmax timings on Vulkan or CUDA."""
import argparse
import json
import statistics
import time
import os
from pathlib import Path

import torch


def sync(backend):
    if backend=="cuda":
        torch.cuda.synchronize()
    else:
        import torch_vulkan
        torch_vulkan.synchronize()


def benchmark(backend,rows,cols,repeats=9):
    torch.manual_seed(rows+cols)
    inp_cpu=torch.randn(rows,cols)
    grad_cpu=torch.randn(rows,cols)
    x=inp_cpu.to(backend).detach().requires_grad_(True)
    grad=grad_cpu.to(backend)
    from torch.nn import functional as F
    forward_times=[]
    backward_times=[]
    out=None
    for j in range(4+repeats):
        sync(backend)
        t0=time.perf_counter()
        out=F.log_softmax(x,dim=-1)
        sync(backend)
        t1=time.perf_counter()
        (gx,) = torch.autograd.grad(out,x,grad_outputs=grad)
        sync(backend)
        t2=time.perf_counter()
        if j>=4:
            forward_times.append((t1-t0)*1000)
            backward_times.append((t2-t1)*1000)
    target=F.log_softmax(inp_cpu,dim=-1)
    err=(out.detach().cpu()-target).abs().max().item()
    return {
        "rows":rows, "cols":cols,
        "forward_median_ms":round(statistics.median(forward_times),4),
        "backward_median_ms":round(statistics.median(backward_times),4),
        "forward_samples_ms":[round(n,4) for n in forward_times],
        "backward_samples_ms":[round(n,4) for n in backward_times],
        "max_forward_error":err,
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--backend",required=True,choices=("cuda","vulkan"))
    ap.add_argument("--out",type=Path)
    args=ap.parse_args()
    if args.backend=="cuda":
        name=torch.cuda.get_device_name(0)
    else:
        import torch_vulkan
        name=torch_vulkan.device_name()
    out=dict(backend=args.backend,device=name,
             native_logsoftmax=os.environ.get("TORCH_VULKAN_NATIVE_LOGSOFTMAX","auto"),
             shape_results=[])
    for rows,cols in [(128,10),(128,256),(128,1024),(32,4096),(256,4096),(512,4096)]:
        info=benchmark(args.backend,rows,cols)
        out["shape_results"].append(info)
        print(f"{rows}x{cols}: {info['forward_median_ms']} ms forward, "
              f"{info['backward_median_ms']} ms backward, "
              f"error {info['max_forward_error']}",flush=True)
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(out,indent=2)+"\n")


if __name__=="__main__":
    main()
