"""Matched Conv2D forward/dInput/dWeight/dBias; FP32, same seeds and shapes.

Wall medians include allocation/dispatch/synchronization. Vulkan timestamps
separate shader/upload/download; CUDA events measure device intervals (which
can include launch gaps), not summed per-kernel Nsight duration.
"""
import argparse
import json
import os
from pathlib import Path
import statistics
import time

import torch
from probe_conv_algorithms import CASES


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["cuda", "vulkan"], required=True)
    p.add_argument("--algorithms", default="auto")
    p.add_argument("--cases", default="")
    p.add_argument("--warmup", type=int, default=4)
    p.add_argument("--repeats", type=int, default=9)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    torch.set_num_threads(6)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    if args.backend == "vulkan":
        import torch_vulkan
        assert torch_vulkan.is_hardware_accelerated()
        name = torch_vulkan.device_name()
        sync = torch_vulkan.synchronize
    else:
        name = torch.cuda.get_device_name()
        sync = torch.cuda.synchronize
    report = dict(gpu=name, backend=args.backend, torch=torch.__version__,
                  dtype="float32", warmup=args.warmup, repeats=args.repeats,
                  resident=os.getenv("TORCH_VULKAN_GPU_RESIDENT", "0"),
                  deferred=os.getenv("TORCH_VULKAN_DEFER_SUBMISSIONS", "0"),
                  timestamps=os.getenv("TORCH_VULKAN_PROFILE_TIMESTAMPS", "0"),
                  cases=[])
    for case in CASES:
        label, shape, co, kernel, stride, pad, dilation, groups = case
        if args.cases and label not in args.cases.split(","):
            continue
        torch.manual_seed(2026 + len(label))
        x = torch.randn(*shape)
        w = torch.randn(co, shape[1] // groups, *kernel)
        b = torch.randn(co)
        ref = torch.nn.functional.conv2d(x, w, b, stride, pad, dilation, groups)
        grad = torch.randn_like(ref)
        def backward(go, inp, wt, mask):
            return torch.ops.aten.convolution_backward.default(
                go, inp, wt, [co], stride, pad, dilation, False, [0, 0], groups, mask)
        references = [ref, *[t.float() for t in backward(grad.double(), x.double(), w.double(), [True, True, True])]]
        xv, wv, bv, gv = [t.to(args.backend) for t in (x, w, b, grad)]
        row = dict(name=label, input_shape=shape, weight_shape=list(w.shape),
                   stride=stride, padding=pad, dilation=dilation, groups=groups,
                   measurements=[])
        for algo in args.algorithms.split(","):
            os.environ["TORCH_VULKAN_CONV_BACKWARD_ALGO"] = algo
            ops = [lambda: torch.nn.functional.conv2d(
                xv, wv, bv, stride, pad, dilation, groups)]
            for i in range(3):
                mask = [j == i for j in range(3)]
                ops.append(lambda i=i, mask=mask: backward(gv, xv, wv, mask)[i])
            for phase, fn, expected in zip(("forward", "dinput", "dweight", "dbias"),
                                           ops, references):
                out = fn()
                torch.testing.assert_close(out.cpu(), expected, atol=5e-3, rtol=5e-4)
                error = (out.cpu() - expected).abs().max().item()
                for _ in range(args.warmup):
                    out = fn()
                sync()
                before = torch_vulkan.gpu_timing_stats() if args.backend == "vulkan" else {}
                wall, events = [], []
                for _ in range(args.repeats):
                    sync()
                    if args.backend == "cuda":
                        start = torch.cuda.Event(enable_timing=True)
                        end = torch.cuda.Event(enable_timing=True)
                    t = time.perf_counter()
                    if args.backend == "cuda":
                        start.record()
                    out = fn()
                    if args.backend == "cuda":
                        end.record()
                    sync()
                    wall.append((time.perf_counter() - t) * 1000)
                    if args.backend == "cuda":
                        events.append(start.elapsed_time(end))
                result = dict(algorithm=algo, phase=phase, wall_median_ms=statistics.median(wall),
                              wall_min_ms=min(wall), wall_max_ms=max(wall), max_abs_error=error)
                if events:
                    result["cuda_event_median_ms"] = statistics.median(events)
                else:
                    after = torch_vulkan.gpu_timing_stats()
                    result["gpu"] = {}
                    for key, values in after.items():
                        prev = before.get(key, {})
                        samples = values["samples"] - prev.get("samples", 0)
                        if not samples:
                            continue
                        result["gpu"][key] = {
                            field: (values[field] - prev.get(field, 0)) / args.repeats
                            for field in ("shader_ms", "upload_ms", "download_ms", "host_eval_ms")}
                row["measurements"].append(result)
                print(label, algo, phase, json.dumps(result), flush=True)
        report["cases"].append(row)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
