"""Nsight Systems comparison: identical XDNA2 MLP step, CUDA vs Vulkan.

Capture NVTX stages and GPU kernel/workload activity. GPU execution is synchronized
at every marked stage boundary to make time attribution unambiguous.
This instrumented time is not intended as a speed benchmark; compare medians
from the uninstrumented benchmark script separately.
"""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import nvtx
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["cuda", "vulkan"], required=True)
    p.add_argument("--zoo", type=Path, required=True)
    p.add_argument("--samples", type=int, default=5)
    p.add_argument("--warmup", type=int, default=4)
    p.add_argument("--optimizer", choices=["standard", "vulkan_fused"], default="vulkan_fused")
    p.add_argument("--result", type=Path)
    args = p.parse_args()

    if args.backend == "cuda":
        assert torch.cuda.is_available()
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        name = torch.cuda.get_device_name(0)
    else:
        import torch_vulkan
        assert torch_vulkan.is_hardware_accelerated()
        name = torch_vulkan.device_name()
    torch.set_num_threads(8)
    torch.manual_seed(42)
    sys.path.insert(0, str((args.zoo / "benchmarks").resolve()))
    import model_zoo
    model, _, loss = model_zoo.MODELS["mlp"]()
    model.to(args.backend).train()
    gen = torch.Generator().manual_seed(123)
    x = torch.randn(128, 784, generator=gen).to(args.backend)
    y = torch.randint(10, (128,), generator=gen).to(args.backend)

    if args.backend == "vulkan" and args.optimizer == "vulkan_fused":
        from torch_vulkan.optim import AdamW
        optim = AdamW(model.parameters(), lr=1e-4)
    else:
        optim = torch.optim.AdamW(model.parameters(), lr=1e-4, foreach=False)

    def sync():
        if args.backend == "cuda":
            torch.cuda.synchronize(0)
        else:
            import torch_vulkan
            torch_vulkan.synchronize()

    def step(i, capture):
        times = {}
        with nvtx.annotate("measured_step" if capture else "warmup", domain="pv-profiling"):
            for stage in ("zero_grad", "forward", "loss", "backward", "optimizer"):
                sync()
                start = time.perf_counter()
                with nvtx.annotate(stage, domain="pv-profiling"):
                    if stage == "zero_grad":
                        optim.zero_grad(set_to_none=True)
                    elif stage == "forward":
                        pred = model(x)
                    elif stage == "loss":
                        value = loss(pred, y)
                    elif stage == "backward":
                        value.backward()
                    else:
                        optim.step()
                    sync()
                times[stage] = (time.perf_counter() - start) * 1000
            sync()
        return times

    for i in range(args.warmup):
        step(i, False)
    times = [step(i, True) for i in range(args.samples)]
    stats = {"torch_version": torch.__version__, "backend": args.backend,
             "device": name, "optimizer": args.optimizer,
             "samples": times, "median_stage_ms": {
                 stage: round(statistics.median(row[stage] for row in times), 4)
                 for stage in times[0]},
             "total_median_ms": round(statistics.median(sum(x.values()) for x in times), 4)}
    print(json.dumps(stats, indent=2))
    if args.result:
        args.result.parent.mkdir(parents=True, exist_ok=True)
        args.result.write_text(json.dumps(stats, indent=2) + "\n")


if __name__ == "__main__":
    main()
