#!/usr/bin/env python3
"""Run selected original TorchBench models on matched physical Vulkan/CUDA GPUs.

Example:
  python benchmarks/run_official_torchbench_matrix.py \
    --cases dcgan:eval:2,pytorch_unet:eval:1 --warmup 2 --repeats 5
Runs each model/backend in a clean process and stores unsupported results.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cases", default="dcgan:eval:2,pytorch_unet:eval:1",
                   help="model:train_or_eval:batch_size comma-separated")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--rounds", type=int, default=1)
    p.add_argument("--cuda-uuid", default="GPU-30acbf2b-a41a-8b88-e298-fe66c4138b29",
                   help="Explicit physical CUDA device; default RTX 2080 Ti on test host")
    p.add_argument("--vulkan-index", default="0")
    p.add_argument("--torchbench", type=Path, default=Path("/workspace/pytorch-vulkan-official-torchbench"))
    p.add_argument("--out", type=Path, default=Path("benchmarks/results/official-torchbench"))
    p.add_argument("--timeout", type=int, default=300)
    args = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    cases = []
    for name in args.cases.split(","):
        parts = name.split(":")
        if len(parts) != 3 or parts[1] not in ("eval", "train"):
            p.error("Invalid case: " + name)
        model, mode, size = parts
        cases.append((model, mode, int(size)))
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=args.cuda_uuid,
               TORCH_VULKAN_PHYSICAL_DEVICE_INDEX=args.vulkan_index,
               TORCH_VULKAN_FORBID_CPU_FALLBACK="1",
               OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
    # Normal production defaults: do not enable profiling or override GPU residency.
    for key in ["TORCH_VULKAN_PROFILE_TIMESTAMPS",
                "TORCH_VULKAN_PROFILE_HOST_HOTPATH", "TORCH_VULKAN_CONV_ALGO",
                "TORCH_VULKAN_DEFER_SUBMISSIONS", "TORCH_VULKAN_GPU_RESIDENT"]:
        env.pop(key, None)
    records = []
    for round_number in range(args.rounds):
        order = ("cuda", "vulkan") if round_number % 2 == 0 else ("vulkan", "cuda")
        for model, mode, size in cases:
            for backend in order:
                dst = out / f"{model}-{mode}-b{size}-r{round_number}-{backend}.json"
                exe = root / (".venv-cuda" if backend == "cuda" else ".venv") / "bin/python"
                cmd = [str(exe), str(root / "benchmarks/run_official_torchbench.py"),
                       "--backend", backend, "--model", model, "--test", mode,
                       "--batch-size", str(size), "--warmup", str(args.warmup),
                       "--repeats", str(args.repeats), "--torchbench", str(args.torchbench),
                       "--out", str(dst)]
                try:
                    with (out / f"{dst.stem}.log").open("w") as log:
                        result = subprocess.run(cmd, cwd=root, env=env, stdout=log,
                                                stderr=subprocess.STDOUT,
                                                timeout=args.timeout)
                    code = result.returncode
                except subprocess.TimeoutExpired:
                    code = "timeout"
                data = json.loads(dst.read_text()) if dst.is_file() else {}
                record = dict(model=model, test=mode, batch_size=size,
                              backend=backend, round=round_number,
                              status=data.get("status", "timeout_or_crash"),
                              exit_code=code, device=data.get("device"),
                              latency_ms=data.get("latency_ms", {}).get("median"),
                              numerical_parity=data.get("eval_reference", {}).get("passed"),
                              cpu_fallback_calls=data.get("execution", {}).get("cpu_fallback_calls"),
                              error=data.get("error"))
                if backend == "cuda" and record["device"] and "2080 Ti" not in record["device"] and args.cuda_uuid.startswith("GPU-30ac"):
                    record["status"] = "wrong_gpu"
                records.append(record)
                print(json.dumps(record), flush=True)
                (out / "index.json").write_text(json.dumps(records, indent=2) + "\n")
    comparisons = []
    for model, mode, size in cases:
        pair = {}
        for backend in ("cuda", "vulkan"):
            medians = [r["latency_ms"] for r in records
                       if r["model"] == model and r["test"] == mode
                       and r["batch_size"] == size and r["backend"] == backend
                       and r["status"] == "ok" and r["latency_ms"] is not None]
            pair[backend] = statistics.median(medians) if len(medians) == args.rounds else None
        comparisons.append(dict(model=model, test=mode, batch_size=size,
                                cuda_ms=pair["cuda"], vulkan_ms=pair["vulkan"],
                                slowdown=(pair["vulkan"] / pair["cuda"]
                                          if pair["cuda"] and pair["vulkan"] else None)))
    (out / "summary.json").write_text(json.dumps(comparisons, indent=2) + "\n")
    print("SUMMARY", json.dumps(comparisons), flush=True)


if __name__ == "__main__":
    main()
