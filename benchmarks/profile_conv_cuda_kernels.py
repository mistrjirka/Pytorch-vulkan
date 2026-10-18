"""Capture actual CUDA kernels for the same cases as probe_conv_backward.py."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import torch
from probe_conv_algorithms import CASES

p = argparse.ArgumentParser()
p.add_argument("--cases", default="standard5,large3,large5,large1")
p.add_argument("--out", type=Path, required=True)
args = p.parse_args()
torch.set_num_threads(6)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False
report = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__,
              dtype="float32", iterations=5, cases=[])
for name, shape, co, kernel, stride, pad, dilation, groups in CASES:
    if name not in args.cases.split(","):
        continue
    torch.manual_seed(2026 + len(name))
    x = torch.randn(*shape).cuda()
    w = torch.randn(co, shape[1] // groups, *kernel).cuda()
    b = torch.randn(co).cuda()
    y = torch.nn.functional.conv2d(x, w, b, stride, pad, dilation, groups)
    go = torch.randn(y.shape).cuda()
    ops = [lambda: torch.nn.functional.conv2d(x, w, b, stride, pad, dilation, groups)]
    for i in range(3):
        mask = [j == i for j in range(3)]
        ops.append(lambda mask=mask: torch.ops.aten.convolution_backward.default(
            go, x, w, [co], stride, pad, dilation, False, [0, 0], groups, mask))
    for phase, fn in zip(("forward", "dinput", "dweight", "dbias"), ops):
        for _ in range(5):
            out = fn()
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[
                torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as prof:
            for _ in range(5):
                out = fn()
            torch.cuda.synchronize()
        kernels = defaultdict(lambda: dict(calls=0, total_ms=0.0))
        for event in prof.events():
            if event.device_type == torch.autograd.DeviceType.CUDA:
                kernels[event.name]["calls"] += 1
                kernels[event.name]["total_ms"] += event.device_time_total / 1000
        rows = [dict(name=k, calls_per_iteration=v["calls"]/5,
                     ms_per_iteration=v["total_ms"]/5) for k, v in kernels.items()]
        rows.sort(key=lambda r: r["ms_per_iteration"], reverse=True)
        result = dict(case=name, phase=phase, kernels=rows,
                      summed_kernel_ms=sum(r["ms_per_iteration"] for r in rows))
        report["cases"].append(result)
        print(json.dumps(result), flush=True)
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(json.dumps(report, indent=2) + "\n")
