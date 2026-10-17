from __future__ import annotations

"""Validate FP32 Vulkan training against the *actual* pytorch-XDNA2 model zoo.

Requires a checkout of https://github.com/mistrjirka/pytorch-XDNA2. Its model
definitions are imported, not copied into this project. The zoo's production
BF16/batch sizes are deliberately reduced until Vulkan supports them.
"""

import argparse
import copy
import json
import subprocess
from pathlib import Path
import sys
import time

import torch
import torch.nn.functional as F
import torch_vulkan


def make_problem(name: str, zoo):
    # CPU reference and Vulkan receive bit-identical initial tensors.
    model, _, loss_fn = zoo.MODELS[name]()
    g = torch.Generator().manual_seed(1)
    if name == "mlp":
        return model, torch.randn(32, 784, generator=g), torch.randint(10, (32,), generator=g), loss_fn
    if name == "unet":
        return model, torch.randn(1, 3, 16, 16, generator=g), torch.rand(1, 1, 16, 16, generator=g), loss_fn
    if name == "embedding":
        return model, torch.randint(50000, (16, 64), generator=g), torch.randint(20, (16,), generator=g), loss_fn
    if name == "style_net":
        # Original StyleNet architecture (all 5 residual blocks); smaller batch/spatial dimensions.
        return model, torch.randn(1, 3, 8, 8, generator=g), torch.rand(1, 3, 8, 8, generator=g), loss_fn
    if name == "resnet18":
        return model, torch.randn(2, 3, 32, 32, generator=g), torch.randint(100, (2,), generator=g), loss_fn
    raise ValueError(f"No size-safe FP32 adapter defined for {name}")


def run_case(name: str, zoo, repeats: int):
    torch.manual_seed(17)
    cpu, x_cpu, y_cpu, loss_fn = make_problem(name, zoo)
    cpu.train()
    vulkan = copy.deepcopy(cpu).to("vulkan").train()
    x_vulkan, y_vulkan = x_cpu.to("vulkan"), y_cpu.to("vulkan")

    def step(model, x, target):
        model.zero_grad(set_to_none=True)
        prediction = model(x)
        loss = loss_fn(prediction, target)
        loss.backward()
        return prediction.detach().cpu().clone(), float(loss.detach().cpu().item())

    started = time.perf_counter()
    reference, ref_loss = step(cpu, x_cpu, y_cpu)
    cpu_ms = (time.perf_counter() - started) * 1000
    started = time.perf_counter()
    actual, vulkan_loss = step(vulkan, x_vulkan, y_vulkan)
    vulkan_first_ms = (time.perf_counter() - started) * 1000

    output_err = (reference - actual).abs().max().item()
    grads = {}
    for (n, a), (other, b) in zip(cpu.named_parameters(), vulkan.named_parameters()):
        assert n == other
        if a.grad is None and b.grad is None:
            continue
        assert a.grad is not None and b.grad is not None, f"Missing gradient: {n}"
        assert b.grad.device.type == "vulkan", f"Gradient fell back to CPU: {n}"
        grads[n] = (a.grad - b.grad.detach().cpu()).abs().max().item()
    maximum_grad_error = max(grads.values(), default=0.0)

    # Match the original XDNA2 zoo's AdamW optimizer, not merely autograd.
    cpu_optim = torch.optim.AdamW(cpu.parameters(), lr=1e-4)
    gpu_optim = torch.optim.AdamW(vulkan.parameters(), lr=1e-4)
    cpu_optim.step()
    gpu_optim.step()
    updated_err = max(
        (a.detach() - b.detach().cpu()).abs().max().item()
        for a, b in zip(cpu.parameters(), vulkan.parameters())
    )

    # Benchmark repeat steps independently after initialization / pipeline warmup.
    times = []
    for _ in range(repeats):
        started = time.perf_counter()
        step(vulkan, x_vulkan, y_vulkan)
        times.append((time.perf_counter() - started) * 1000)

    return {
        "parameters": sum(p.numel() for p in cpu.parameters()),
        "cpu_first_ms": round(cpu_ms, 3),
        "vulkan_first_ms": round(vulkan_first_ms, 3),
        "vulkan_steady_ms": round(sorted(times)[len(times) // 2], 3) if times else None,
        "loss_cpu": ref_loss,
        "loss_vulkan": vulkan_loss,
        "max_output_abs_error": output_err,
        "max_grad_abs_error": maximum_grad_error,
        "max_adamw_weight_abs_error": updated_err,
        "input_shape": list(x_cpu.shape),
        "dtype": str(x_vulkan.dtype),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zoo", type=Path, required=True, help="pytorch-XDNA2 checkout root")
    parser.add_argument("--models", default="mlp,unet,embedding,style_net")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if args.repeats < 0:
        parser.error("--repeats must be nonnegative")
    if not (args.zoo / "benchmarks" / "model_zoo.py").is_file():
        parser.error(f"Missing {args.zoo}/benchmarks/model_zoo.py")
    sys.path.insert(0, str((args.zoo / "benchmarks").resolve()))
    import model_zoo

    torch.set_num_threads(args.threads)
    results = {
        "torch": torch.__version__,
        "vulkan_device": torch_vulkan.device_name(),
        "physical_gpu": torch_vulkan.is_hardware_accelerated(),
        "zoo_repo": "https://github.com/mistrjirka/pytorch-XDNA2",
        "zoo_commit": subprocess.check_output(
            ["git", "-C", str(args.zoo), "rev-parse", "HEAD"], text=True
        ).strip(),
        "note": "FP32 reduced-size smoke probes, NOT the original BF16 zoo benchmark",
        "cases": {},
    }
    print(f"GPU: {results['vulkan_device']} (physical={results['physical_gpu']})", flush=True)
    for name in args.models.split(","):
        name = name.strip()
        try:
            result = run_case(name, model_zoo, args.repeats)
            results["cases"][name] = result
            print(f"{name}: Vulkan {result['vulkan_steady_ms']} ms, CPU {result['cpu_first_ms']} ms; "
                  f"dOutput={result['max_output_abs_error']:.4g}, "
                  f"dGrad={result['max_grad_abs_error']:.4g}, "
                  f"dWeight={result['max_adamw_weight_abs_error']:.4g}", flush=True)
        except Exception as exc:
            results["cases"][name] = {"error": f"{type(exc).__name__}: {str(exc).splitlines()[0]}"}
            print(f"{name}: {results['cases'][name]['error']}", flush=True)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(results, indent=2) + "\n")
    if any("error" in value for value in results["cases"].values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
