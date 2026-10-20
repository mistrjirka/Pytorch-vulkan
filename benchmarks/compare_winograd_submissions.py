"""Controlled within-process same-tensor comparison of three Winograd submits vs one chain."""
import argparse
import json
import os
import statistics
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torch_vulkan

CASES=(
    ("grouped_medium",(2,64,64,64),64,4),
    ("grouped_large",(1,64,128,128),64,4),
    ("grouped_small",(2,64,32,35),64,4),
    ("dense_small",(2,64,16,19),64,1),
    ("odd",(1,32,37,41),48,1),
)

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--cases",default=",".join(x[0] for x in CASES))
    parser.add_argument("--rounds",type=int,default=3)
    parser.add_argument("--repeats",type=int,default=9)
    parser.add_argument("--warmup",type=int,default=4)
    parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args()
    os.environ["TORCH_VULKAN_CONV_ALGO"]="winograd"
    wanted=set(args.cases.split(","))
    results=[]
    for name,shape,cout,groups in CASES:
        if name not in wanted:continue
        gen=torch.Generator().manual_seed(777)
        x=torch.randn(shape,generator=gen)
        weight=torch.randn((cout,shape[1]//groups,3,3),generator=gen)*0.08
        bias=torch.randn(cout,generator=gen)*0.15
        reference=F.conv2d(x.double(),weight.double(),bias.double(),padding=1,groups=groups).float()
        xv,wv,bv=(x.to("vulkan"),weight.to("vulkan"),bias.to("vulkan"))
        def run():return F.conv2d(xv,wv,bv,padding=1,groups=groups)
        blocks=[]
        baseline=torch_vulkan.execution_stats().get("cpu_fallback_calls",0)
        for iteration in range(args.rounds):
            for algo in (("legacy","chain","chain","legacy") if iteration%2==0
                         else ("chain","legacy","legacy","chain")):
                if algo=="legacy":os.environ["TORCH_VULKAN_CONV_WINOGRAD_LEGACY_SUBMISSIONS"]="1"
                else:os.environ.pop("TORCH_VULKAN_CONV_WINOGRAD_LEGACY_SUBMISSIONS",None)
                y=run()
                torch_vulkan.synchronize()
                torch.testing.assert_close(y.cpu(),reference,atol=3e-4,rtol=1e-3)
                for _ in range(args.warmup):run()
                torch_vulkan.synchronize()
                times=[]
                for _ in range(args.repeats):
                    start=time.perf_counter()
                    _=run()
                    torch_vulkan.synchronize()
                    times.append((time.perf_counter()-start)*1000)
                blocks.append(dict(algorithm=algo,median_ms=statistics.median(times),samples_ms=times))
        assert torch_vulkan.execution_stats().get("cpu_fallback_calls",0)==baseline
        medians={a:statistics.median(x["median_ms"] for x in blocks if x["algorithm"]==a)
                 for a in ("chain","legacy")}
        print(name,medians,"speedup",round(medians["legacy"]/medians["chain"],3),flush=True)
        results.append(dict(case=name,shape=shape,groups=groups,medians_ms=medians,
                            ratio_chain_to_legacy=medians["chain"]/medians["legacy"],
                            blocks=blocks,cpu_double_parity="pass",cpu_fallback_delta=0))
    args.out.parent.mkdir(exist_ok=True,parents=True)
    args.out.write_text(json.dumps(results,indent=2)+"\n")

if __name__=="__main__":
    main()
