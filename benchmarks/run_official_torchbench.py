#!/usr/bin/env python3
"""Run UNMODIFIED upstream TorchBench model implementations on CUDA or Vulkan.

Requires /workspace/pytorch-vulkan-official-torchbench (override with --torchbench).
Run each backend in its respective .venv/.venv-cuda Python and process.
Synchronized wall timing includes actual upstream model eval()/train().
Never substitutes CPU operators or edits an upstream model for compatibility.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

import torch


def as_tensor_list(value):
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (tuple, list)):
        out = []
        for part in value:
            out.extend(as_tensor_list(part))
        return out
    if isinstance(value, dict):
        out = []
        for key in sorted(value):
            out.extend(as_tensor_list(value[key]))
        return out
    return []


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--torchbench", type=Path, default=Path("/workspace/pytorch-vulkan-official-torchbench"))
    p.add_argument("--backend", required=True, choices=["vulkan", "cuda"])
    p.add_argument("--model", required=True)
    p.add_argument("--test", required=True, choices=["eval", "train"])
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repeats", type=int, default=10)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    if args.warmup < 0 or args.repeats < 1:
        p.error("warmup must be >= 0 and repeats >= 1")
    if args.backend == "vulkan" and os.environ.get("TORCH_VULKAN_FORBID_CPU_FALLBACK") != "1":
        p.error("Vulkan benchmark requires TORCH_VULKAN_FORBID_CPU_FALLBACK=1")
    row = dict(model=args.model, test=args.test, batch_size=args.batch_size,
               backend=args.backend, warmup=args.warmup, repeats=args.repeats,
               fp32=True, tf32=False, torch_version=torch.__version__,
               torch_git=torch.version.git_version,
               benchmark_commit=subprocess.check_output(
                   ["git", "-C", str(args.torchbench), "rev-parse", "HEAD"],
                   text=True).strip(),
               backend_commit=subprocess.check_output(
                   ["git", "rev-parse", "HEAD"], text=True).strip(),
               env={k: v for k, v in os.environ.items()
                    if k.startswith("TORCH_VULKAN_") or k == "CUDA_VISIBLE_DEVICES"})
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def persist():
        args.out.write_text(json.dumps(row, indent=2, default=str) + "\n")

    try:
        if not (args.torchbench / "torchbenchmark").is_dir():
            raise FileNotFoundError(args.torchbench)
        sys.path.insert(0, str(args.torchbench))
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.manual_seed(12345)
        if args.backend == "vulkan":
            import torch_vulkan as vk
            if not vk.is_hardware_accelerated():
                raise RuntimeError("Vulkan backend is not GPU accelerated")
            sync = vk.synchronize
            row["device"] = vk.device_name()
        else:
            sync = torch.cuda.synchronize
            row["device"] = torch.cuda.get_device_name()
        mod = importlib.import_module("torchbenchmark.models." + args.model)
        start = time.perf_counter()
        m = mod.Model(test=args.test, device=args.backend, batch_size=args.batch_size)
        sync()
        row["initialization_seconds"] = time.perf_counter() - start
        row["batch_size_effective"] = m.batch_size
        if args.test == "eval":
            # Verify the actual upstream model's outputs against identical
            # weights and inputs on CPU, without modifying the timed model.
            net, inputs = m.get_module()
            reference = copy.deepcopy(net).cpu()  # preserve upstream training/eval state
            reference_inputs = as_tensor_list(inputs)
            if not isinstance(inputs, (tuple, list)):
                inputs = (inputs,)
            reference_inputs = tuple(
                a.detach().cpu() if isinstance(a, torch.Tensor) else a for a in inputs
            )
            with torch.no_grad():
                cpu_output = reference(*reference_inputs)
                gpu_output = net(*inputs)
            sync()
            lhs = as_tensor_list(cpu_output)
            rhs = as_tensor_list(gpu_output)
            if len(lhs) != len(rhs):
                raise AssertionError("model parity outputs do not match structure")
            failures = []
            max_abs = 0
            for i, (cpu, gpu) in enumerate(zip(lhs, rhs)):
                actual = gpu.detach().cpu()
                if tuple(actual.shape) != tuple(cpu.shape):
                    failures.append("shape mismatch at output " + str(i))
                    continue
                err = (actual.float() - cpu.float()).abs()
                max_abs = max(max_abs, float(err.max()) if err.numel() else 0)
                try:
                    torch.testing.assert_close(actual, cpu, atol=8e-5, rtol=2e-3)
                except AssertionError as exc:
                    failures.append(f"output {i}: {str(exc)[:250]}")
            row["eval_reference"] = dict(outputs=len(lhs), max_abs=max_abs,
                                          passed=not failures, failures=failures)
            if failures:
                raise AssertionError("CPU reference numerical parity failed")
        else:
            row["train_reference"] = "not_checked: model includes optimizer state and potentially random steps"

        run = m.eval if args.test == "eval" else m.train
        if args.backend == "vulkan":
            vk.reset_execution_stats()
        samples = []
        for i in range(args.warmup + args.repeats):
            sync()
            start = time.perf_counter()
            run()
            sync()
            elapsed_ms = 1000 * (time.perf_counter() - start)
            if i >= args.warmup:
                samples.append(elapsed_ms)
        row["latency_ms"] = dict(median=statistics.median(samples),
                                 min=min(samples), max=max(samples), samples=samples)
        if args.backend == "vulkan":
            row["execution"] = vk.execution_stats()
            if row["execution"].get("cpu_fallback_calls") != 0:
                raise AssertionError("CPU fallback detected: " + str(row["execution"]))
        row["status"] = "ok"
    except Exception as exc:
        row["status"] = "unsupported_or_error"
        row["error_type"] = type(exc).__name__
        row["error"] = str(exc)
        row["traceback"] = traceback.format_exc(limit=14)
    persist()
    print(json.dumps({k: row.get(k) for k in
                      ["model", "test", "backend", "device", "status", "latency_ms",
                       "eval_reference", "error_type", "error"]}), flush=True)
    if row["status"] != "ok":
        sys.exit(1)


if __name__ == "__main__":
    main()
