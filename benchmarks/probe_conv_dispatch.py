"""Matched FP32 convolution dispatch probe, independent of any model.

Run each Vulkan tile mode in a fresh process, with the same external residency
and synchronization flags. For example::

    TORCH_VULKAN_CONV_TILE=base .venv/bin/python benchmarks/probe_conv_dispatch.py \
        --backend vulkan --out /tmp/conv-base.json
    TORCH_VULKAN_CONV_TILE=auto .venv/bin/python benchmarks/probe_conv_dispatch.py \
        --backend vulkan --out /tmp/conv-auto.json
    TORCH_VULKAN_CONV_AUTOTUNE=1 TORCH_VULKAN_CONV_TILE=auto .venv/bin/python benchmarks/probe_conv_dispatch.py \
        --backend vulkan --out /tmp/conv-autotune.json
    .venv-cuda/bin/python benchmarks/probe_conv_dispatch.py \
        --backend cuda --out /tmp/conv-cuda.json

Wall samples include allocation, dispatch and final synchronization, but exclude
CPU reference checks and explicit .cpu() downloads. Backend-internal eager
uploads/downloads remain included. Vulkan timestamp deltas are reported when
available; their absence is explicit (resident/deferred paths may not instrument
them). Optional CUDA event samples run separately from wall samples and measure
device intervals including launch gaps, not the sum of kernel execution times.
Use --profile-cuda-kernels for a separate five-iteration trace of actual kernels.
"""
import argparse
from collections import defaultdict
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import time

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class Case:
    name: str
    shape: tuple
    cout: int
    kernel: tuple
    stride: tuple = (1, 1)
    padding: tuple = (0, 0)
    dilation: tuple = (1, 1)
    groups: int = 1


CASES = (
    Case("pointwise", (2, 32, 32, 35), 64, (1, 1)),
    Case("rect3x5", (2, 16, 31, 35), 24, (3, 5),
         stride=(2, 1), padding=(1, 4), dilation=(1, 2)),
    Case("filter10x10", (1, 16, 40, 43), 32, (10, 10), padding=(5, 5)),
    Case("filter11x11", (1, 16, 40, 43), 32, (11, 11), padding=(5, 5)),
    Case("grouped", (2, 24, 29, 31), 40, (3, 5),
         padding=(1, 2), groups=4),
    Case("depthwise", (2, 24, 29, 31), 48, (3, 3),
         padding=(1, 1), groups=24),
    Case("conv3x3_large", (2, 32, 64, 64), 64, (3, 3), padding=(1, 1)),
    Case("conv3x3_small", (2, 64, 16, 19), 64, (3, 3), padding=(1, 1)),
)


def backward(case, grad, x, weight, mask):
    return torch.ops.aten.convolution_backward.default(
        grad, x, weight, [case.cout], case.stride, case.padding, case.dilation,
        False, [0, 0], case.groups, mask)


def inputs_and_references(case, seed):
    """Generate CPU inputs identically regardless of backend or selected subset."""
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(case.shape, generator=generator)
    weight_shape = (case.cout, case.shape[1] // case.groups, *case.kernel)
    weight = torch.randn(weight_shape, generator=generator)
    # Fan-in scaling keeps this a representative finite FP32 workload, while
    # double CPU accumulation makes the reference independent of CPU FP32 paths.
    weight /= math.sqrt(math.prod(weight_shape[1:]))
    bias = torch.randn(case.cout, generator=generator)
    forward_ref = F.conv2d(x.double(), weight.double(), bias.double(),
                          case.stride, case.padding, case.dilation, case.groups)
    grad = torch.randn(forward_ref.shape, generator=generator)
    gradients = backward(case, grad.double(), x.double(), weight.double(),
                         [True, True, True])
    references = (forward_ref.float(), *(value.float() for value in gradients))
    digest = hashlib.sha256()
    for tensor in (x, weight, bias, grad):
        digest.update(tensor.numpy().tobytes())
    return (x, weight, bias, grad), references, digest.hexdigest()


def summary(samples):
    return dict(median_ms=statistics.median(samples), min_ms=min(samples),
                max_ms=max(samples), samples_ms=samples)


def timestamp_delta(before, after, repeats):
    result = {}
    for key, values in after.items():
        previous = before.get(key, {})
        count = values.get("samples", 0) - previous.get("samples", 0)
        if not count:
            continue
        result[key] = dict(samples=count, **{
            field + "_per_invocation": (values[field] - previous.get(field, 0)) / repeats
            for field in ("shader_ms", "upload_ms", "download_ms", "host_eval_ms")
            if field in values
        })
    return result


def counter_delta(before, after):
    return {key: value - before.get(key, 0) for key, value in after.items()
            if value != before.get(key, 0)}


def measure(fn, expected, args, sync, timing_stats, dispatch_stats):
    sync()
    dispatch_before = dispatch_stats()
    start = time.perf_counter()
    output = fn()
    sync()
    first_call_ms = (time.perf_counter() - start) * 1000
    dispatch_after_first = dispatch_stats()
    actual = output.cpu()
    torch.testing.assert_close(actual, expected, atol=args.atol, rtol=args.rtol)
    error = (actual.double() - expected.double()).abs()
    parity = dict(max_abs_error=error.max().item(),
                  rms_error=error.square().mean().sqrt().item(),
                  reference_max_abs=expected.abs().max().item())
    del actual, error, output
    for _ in range(args.warmup):
        output = fn()
        sync()
        del output
    before = timing_stats()
    dispatch_before_wall = dispatch_stats()
    wall = []
    for _ in range(args.repeats):
        sync()
        start = time.perf_counter()
        output = fn()
        sync()
        wall.append((time.perf_counter() - start) * 1000)
        del output
    after = timing_stats()
    dispatch_after_wall = dispatch_stats()
    result = dict(first_call_ms=first_call_ms, wall=summary(wall), parity=parity)
    if args.backend == "vulkan":
        gpu = timestamp_delta(before, after, args.repeats)
        result["vulkan_gpu"] = gpu
        result["dispatch"] = dict(
            before=dispatch_before, after_first_call=dispatch_after_first,
            before_steady_state=dispatch_before_wall, after_steady_state=dispatch_after_wall,
            first_call_delta=counter_delta(dispatch_before, dispatch_after_first),
            steady_state_delta=counter_delta(dispatch_before_wall, dispatch_after_wall))
        result["vulkan_timestamp_status"] = "available" if gpu else "not_recorded"
    elif args.cuda_events:
        # Do not mix event creation/recording overhead into wall A/B samples.
        event_times = []
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        for _ in range(args.repeats):
            sync()
            start_event.record()
            output = fn()
            end_event.record()
            sync()
            event_times.append(start_event.elapsed_time(end_event))
            del output
        result["cuda_event_interval"] = summary(event_times)
    return result


def summarize_cuda_events(events, iterations):
    grouped = defaultdict(lambda: dict(count=0, total_us=0.0))
    for event in events:
        if event.device_type != torch.autograd.DeviceType.CUDA:
            continue
        grouped[event.name]["count"] += 1
        grouped[event.name]["total_us"] += event.device_time_total
    kernels, memory_operations = [], []
    for name, timing in grouped.items():
        row = dict(name=name, count=timing["count"],
                   calls_per_iteration=timing["count"] / iterations,
                   total_us=timing["total_us"],
                   us_per_iteration=timing["total_us"] / iterations)
        # Kineto's CUDA device events also contain explicit copies and clears.
        target = memory_operations if any(token in name.lower() for token in (
            "memcpy", "memset")) else kernels
        target.append(row)
    for rows in (kernels, memory_operations):
        rows.sort(key=lambda row: row["us_per_iteration"], reverse=True)
    return dict(iterations=iterations, kernels=kernels,
                memory_operations=memory_operations,
                summed_kernel_us_per_iteration=sum(r["us_per_iteration"] for r in kernels),
                summed_memory_us_per_iteration=sum(r["us_per_iteration"] for r in memory_operations))


def profile_cuda_kernels(fn, sync):
    iterations = 5
    for _ in range(iterations):
        output = fn()
        del output
    sync()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA]) as profiler:
        for _ in range(iterations):
            output = fn()
            del output
        sync()
    result = summarize_cuda_events(profiler.events(), iterations)
    result["note"] = "Summed measured GPU kernel durations; excludes CPU/launch gaps; not end-to-end latency."
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", choices=("cuda", "vulkan"), required=True)
    parser.add_argument("--cases", default="", help="Comma-separated case names; default all.")
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=15)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--atol", type=float, default=5e-3)
    parser.add_argument("--rtol", type=float, default=5e-4)
    parser.add_argument("--cuda-events", action="store_true")
    parser.add_argument("--profile-cuda-kernels", action="store_true",
                        help="After normal timings, trace five warmed iterations with CUDA kernel names/times.")
    parser.add_argument("--cuda-benchmark", action="store_true",
                        help="Enable cuDNN candidate benchmarking; otherwise use heuristics.")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.profile_cuda_kernels and args.backend != "cuda":
        parser.error("--profile-cuda-kernels requires --backend cuda")
    if args.warmup < 0 or args.repeats < 1 or args.threads < 1:
        parser.error("warmup must be nonnegative; repeats and threads must be positive")
    selected = set(args.cases.split(",")) if args.cases else {c.name for c in CASES}
    unknown = selected - {c.name for c in CASES}
    if unknown:
        parser.error("unknown cases: " + ", ".join(sorted(unknown)))
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = args.cuda_benchmark
    torch.backends.cudnn.deterministic = False
    if args.backend == "vulkan":
        import torch_vulkan
        sync = torch_vulkan.synchronize
        timing_stats = torch_vulkan.gpu_timing_stats
        dispatch_stats = torch_vulkan.conv_dispatch_stats
        device_name = torch_vulkan.device_name()
    else:
        sync = torch.cuda.synchronize
        timing_stats = lambda: {}
        dispatch_stats = lambda: {}
        device_name = torch.cuda.get_device_name()
    revision = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=False).stdout.strip()
    report = dict(
        backend=args.backend, device=device_name, torch=torch.__version__,
        revision=revision,
        worktree_dirty=bool(subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()),
        convolution_source_sha256=hashlib.sha256(b"".join(
            path.name.encode()+path.read_bytes() for path in sorted([
                Path("csrc/ops/convolution.cpp"), *Path("csrc/shaders").glob("conv2d*.glsl"),
                *Path("csrc/shaders").glob("conv2d*.comp"), *Path("csrc/shaders").glob("conv2d*.spv")]))).hexdigest(),
        dtype="float32", reference="CPU float64 accumulation -> float32",
        seed=args.seed, warmup=args.warmup, repeats=args.repeats, threads=args.threads,
        atol=args.atol, rtol=args.rtol, cuda_tf32=False,
        cudnn_benchmark=args.cuda_benchmark, cudnn_deterministic=False,
        cuda_kernel_profile_iterations=5 if args.profile_cuda_kernels else 0,
        vulkan_tile=os.getenv("TORCH_VULKAN_CONV_TILE", "auto"),
        env={key: value for key, value in os.environ.items()
             if key.startswith("TORCH_VULKAN_") or key in (
                 "CUDA_VISIBLE_DEVICES", "VK_ICD_FILENAMES")},
        timing_notes=[
            "Wall: allocation + dispatch + synchronization, including backend-internal transfers; excludes reference and explicit .cpu() download.",
            "First call may include compilation/tuning; exclude it from steady-state speedups.",
            "CUDA events are separate device intervals, including possible launch gaps.",
            "Vulkan timestamp availability depends on execution path.",
            "Each backward output mask requests only one gradient; not a full training step.",
        ],
        cases=[])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for index, case in enumerate(CASES):
        if case.name not in selected:
            continue
        inputs, references, checksum = inputs_and_references(case, args.seed + index)
        x, weight, bias, grad = (value.to(args.backend) for value in inputs)
        ops = [lambda: F.conv2d(x, weight, bias, case.stride, case.padding,
                               case.dilation, case.groups)]
        for phase_index in range(3):
            mask = [j == phase_index for j in range(3)]
            ops.append(lambda i=phase_index, mask=mask: backward(
                case, grad, x, weight, mask)[i])
        row = dict(**asdict(case), weight_shape=list(weight.shape),
                   output_shape=list(grad.shape), input_sha256=checksum,
                   measurements={})
        for phase, fn, expected in zip(("forward", "dinput", "dweight", "dbias"),
                                        ops, references):
            measurement = measure(fn, expected, args, sync, timing_stats, dispatch_stats)
            if args.profile_cuda_kernels:
                measurement["cuda_kernel_profile"] = profile_cuda_kernels(fn, sync)
            row["measurements"][phase] = measurement
            print(f"{case.name:18} {phase:8} "
                  f"{measurement['wall']['median_ms']:.4f} ms "
                  f"max_error={measurement['parity']['max_abs_error']:.3g}", flush=True)
        report["cases"].append(row)
        if args.backend == "vulkan":
            report["dispatch_final"] = dispatch_stats()
        args.out.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
