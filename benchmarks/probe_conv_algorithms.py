"""Compare general Conv2D shaders on real hardware with CUDA and CPU.

This benchmark keeps operator semantics identical. Env TORCH_VULKAN_CONV_ALGO
is changed per run; native selector validates hardware capabilities.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import time

import torch

CASES = [
    ("tiny", (1, 3, 16, 16), 8, (3, 3), (1, 1), (1, 1), (1, 1), 1),
    ("pointwise", (2, 32, 32, 32), 64, (1, 1), (1, 1), (0, 0), (1, 1), 1),
    ("standard3", (1, 16, 32, 32), 32, (3, 3), (1, 1), (1, 1), (1, 1), 1),
    ("standard5", (1, 16, 32, 32), 32, (5, 5), (1, 1), (2, 2), (1, 1), 1),
    ("standard7", (1, 8, 32, 32), 16, (7, 7), (1, 1), (3, 3), (1, 1), 1),
    ("depthwise3", (1, 32, 48, 48), 64, (3, 3), (1, 1), (1, 1), (1, 1), 32),
    ("depthwise5", (1, 32, 48, 48), 64, (5, 5), (1, 1), (2, 2), (1, 1), 32),
    ("grouped35", (1, 16, 31, 29), 32, (3, 5), (2, 1), (1, 2), (1, 2), 4),
    ("large3", (2, 64, 64, 64), 128, (3, 3), (1, 1), (1, 1), (1, 1), 1),
    ("large5", (1, 32, 64, 64), 64, (5, 5), (1, 1), (2, 2), (1, 1), 1),
    ("large1", (4, 64, 64, 64), 128, (1, 1), (1, 1), (0, 0), (1, 1), 1),
]


def test_case(args, case):
    if args.backend=="vulkan":
        import torch_vulkan
    name, shape, channels_out, kernel, stride, pad, dilation, groups=case
    torch.manual_seed(2026+len(name))
    cpu_x=torch.randn(*shape)
    cpu_w=torch.randn(channels_out,shape[1]//groups,*kernel)
    cpu_b=torch.randn(channels_out)
    ref=torch.nn.functional.conv2d(cpu_x,cpu_w,cpu_b,stride,pad,dilation,groups)
    results=[]
    x=cpu_x.to(args.backend)
    w=cpu_w.to(args.backend)
    b=cpu_b.to(args.backend)
    candidates=(["direct","tiled","auto"] +
                (["depthwise"] if groups==shape[1] else [])) if args.backend=="vulkan" else [args.backend]
    for algo in candidates:
        os.environ["TORCH_VULKAN_CONV_ALGO"]=algo
        def run():
            return torch.nn.functional.conv2d(x,w,b,stride,pad,dilation,groups)
        if args.backend=="cuda": torch.cuda.synchronize()
        out=run()
        if args.backend=="cuda":torch.cuda.synchronize()
        torch.testing.assert_close(out.cpu(),ref,atol=5e-4,rtol=4e-4)
        for _ in range(args.warmup):run()
        if args.backend=="cuda":torch.cuda.synchronize()
        samples=[]
        if args.backend=="vulkan":
            before=torch_vulkan.gpu_timing_stats()
        with torch.inference_mode():
            for _ in range(args.repeats):
                if args.backend=="cuda":torch.cuda.synchronize()
                st=time.perf_counter()
                y=run()
                if args.backend=="cuda":torch.cuda.synchronize()
                samples.append((time.perf_counter()-st)*1000)
        row={"algo":algo,"median_ms":round(statistics.median(samples),4),
             "min_ms":round(min(samples),4),
             "max_ms":round(max(samples),4)}
        if args.backend=="vulkan":
            after=torch_vulkan.gpu_timing_stats()
            changed=[(k,v,before.get(k,{})) for k,v in after.items()
                if v["samples"]-before.get(k,{}).get("samples",0)==args.repeats]
            if len(changed)==1:
                shader_name,a,previous=changed[0]
                row["shader"]=shader_name
                for field in ("upload_ms","shader_ms","download_ms","host_eval_ms"):
                    row[field]=round((a[field]-previous.get(field,0))/args.repeats,6)
        results.append(row)
    return {"name":name,"input_shape":list(shape),"weight_shape":list(cpu_w.shape),
            "output_shape":list(ref.shape),"stride":stride,"padding":pad,
            "dilation":dilation,"groups":groups,"results":results}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--backend",choices=["vulkan","cuda"],required=True)
    p.add_argument("--out",type=Path)
    p.add_argument("--warmup",type=int,default=3)
    p.add_argument("--repeats",type=int,default=7)
    args=p.parse_args()
    torch.set_num_threads(6)
    if args.backend=="vulkan":
        import torch_vulkan
        assert torch_vulkan.is_hardware_accelerated(), "Software Vulkan is not a GPU comparison"
        gpu=torch_vulkan.device_name()
    else:
        assert torch.cuda.is_available()
        gpu=torch.cuda.get_device_name(0)
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False
    doc={"backend":args.backend,"gpu":gpu,"torch":torch.__version__,
         "warmup":args.warmup,"repeats":args.repeats,"dtype":"float32",
         "cases":[]}
    print("GPU",gpu,flush=True)
    for case in CASES:
        result=test_case(args,case)
        doc["cases"].append(result)
        print(case[0],result["results"],flush=True)
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(doc,indent=2)+"\n")


if __name__=="__main__":main()
