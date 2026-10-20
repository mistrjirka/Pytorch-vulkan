"""Within-process A/B of the old per-channel and new parallel BatchNorm.

Use one model, one set of device allocations, identical TF32 and residency
configuration, and alternate mode order. These measurements include complete
operator/model calls and explicit Vulkan synchronization.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time

import torch
import torch_vulkan


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--case",choices=["isolated","torchbench"],default="isolated")
    parser.add_argument("--torchbench-root",type=Path,default=Path(os.environ.get(
        "TORCHBENCH_ROOT",Path(__file__).resolve().parents[2]/"pytorch-vulkan-official-torchbench")))
    parser.add_argument("--shape",default="1,64,640,959")
    parser.add_argument("--repeat",type=int,default=7)
    parser.add_argument("--warmup",type=int,default=3)
    parser.add_argument("--rounds",type=int,default=3)
    parser.add_argument("--out",type=Path)
    args=parser.parse_args()
    os.environ.pop("TORCH_VULKAN_BN_EVAL_LEGACY",None)
    if args.case=="isolated":
        g=torch.Generator().manual_seed(1717)
        shape=tuple(int(v) for v in args.shape.split(","))
        assert len(shape)>=2
        channels=shape[1]
        values=(torch.randn(shape,generator=g),torch.randn(channels,generator=g)*0.2+1,
                torch.randn(channels,generator=g)*0.2,torch.randn(channels,generator=g),
                torch.rand(channels,generator=g)+0.5)
        ref=torch.ops.aten.native_batch_norm(*values,False,0.1,1e-5)[0]
        xv,wv,bv,mv,vv=[a.to("vulkan") for a in values]
        def fn():
            return torch.ops.aten.native_batch_norm(xv,wv,bv,mv,vv,False,0.1,1e-5)[0]
        torch.testing.assert_close(fn().cpu(),ref,rtol=4e-4,atol=5e-5)
    else:
        sys.path.insert(0,str(args.torchbench_root))
        from torchbenchmark.models.pytorch_unet import Model
        bench=Model(test="eval",device="vulkan",batch_size=1)
        fn=bench.eval
    batches=[]
    base=torch_vulkan.execution_stats()["cpu_fallback_calls"]
    for rnd in range(args.rounds):
        order=("legacy","parallel","parallel","legacy") if rnd%2==0 else (
            "parallel","legacy","legacy","parallel")
        for mode in order:
            if mode=="legacy":os.environ["TORCH_VULKAN_BN_EVAL_LEGACY"]="1"
            else:os.environ.pop("TORCH_VULKAN_BN_EVAL_LEGACY",None)
            for _ in range(args.warmup):fn()
            torch_vulkan.synchronize()
            timings=[]
            for _ in range(args.repeat):
                begin=time.perf_counter()
                fn()
                torch_vulkan.synchronize()
                timings.append((time.perf_counter()-begin)*1000)
            batches.append(dict(round=rnd,mode=mode,median_ms=statistics.median(timings),
                                samples_ms=timings))
    assert torch_vulkan.execution_stats()["cpu_fallback_calls"]==base
    modes={m:[x["median_ms"] for x in batches if x["mode"]==m] for m in ("legacy","parallel")}
    stats={m:statistics.median(values) for m,values in modes.items()}
    result=dict(case=args.case,shape=args.shape if args.case=="isolated" else "TorchBench default",device=torch_vulkan.device_name(),rounds=args.rounds,
                samples_per_block=args.repeat,warmups=args.warmup,medians_ms=stats,
                ratio_parallel_to_legacy=stats["parallel"]/stats["legacy"],
                fallback_delta=0,blocks=batches)
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k!="blocks"}))


if __name__=="__main__":
    main()
