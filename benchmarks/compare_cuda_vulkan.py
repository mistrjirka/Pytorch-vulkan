"""Compare CPU, CUDA, and Vulkan on the SAME actual XDNA2 model architectures.

One backend and model per process. Launch with corresponding Python environments.
Uses matching FP32 parameters and cached synthetic inputs; CUDA is synchronized
before/after timed intervals. Reported latency is end-to-end Python wall time,
including transfers and CPU fallbacks *inside model execution*, excluding data
creation and initial model placement.

This script intentionally does not substitute CUDA kernels for Vulkan or change
the model architectures. 'tiny'/'small' are reduced input sizes, not redefined
network layers, needed to run the currently unoptimized Vulkan backend.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

import torch


CONFIG = {
    # shape under each scale / task. Real architecture unchanged.
    "mlp": {"tiny": (32, 784), "small": (128, 784), "medium": (256, 784), "reference": (256, 784)},
    "unet": {"tiny": (1, 3, 16, 16), "small": (1, 3, 32, 32), "medium": (1, 3, 64, 64), "reference": (8, 3, 96, 96)},
    "style_net": {"tiny": (1, 3, 8, 8), "small": (1, 3, 16, 16), "medium": (1, 3, 36, 36), "reference": (40, 3, 36, 36)},
    "resnet18": {"tiny": (2, 3, 32, 32), "small": (2, 3, 48, 48), "medium": (4, 3, 64, 64), "reference": (32, 3, 64, 64)},
    "mobilenet_v3": {"tiny": (2, 3, 32, 32), "small": (2, 3, 64, 64), "medium": (2, 3, 96, 96), "reference": (32, 3, 96, 96)},
    "embedding": {"tiny": (16, 64), "small": (32, 64), "medium": (128, 64), "reference": (128, 64)},
    "transformer": {"tiny": (1, 128), "small": (2, 128), "medium": (4, 128), "reference": (16, 128)},
    "vgg11_128": {"tiny": (1, 3, 32, 32), "small": (1, 3, 64, 64), "medium": (2, 3, 128, 128), "reference": (8, 3, 128, 128)},
}


def data_for_case(name: str, shape, generator: torch.Generator):
    if name == "mlp":
        x = torch.randn(*shape, generator=generator)
        label = torch.randint(0, 10, (shape[0],), generator=generator)
    elif name == "unet":
        x = torch.randn(*shape, generator=generator)
        label = torch.rand(shape[0], 1, shape[2], shape[3], generator=generator).round()
    elif name == "style_net":
        x = torch.randn(*shape, generator=generator)
        label = torch.rand(*shape, generator=generator)
    elif name == "resnet18":
        x = torch.randn(*shape, generator=generator)
        label = torch.randint(0, 100, (shape[0],), generator=generator)
    elif name == "mobilenet_v3":
        x = torch.randn(*shape, generator=generator)
        label = torch.randint(0, 100, (shape[0],), generator=generator)
    elif name == "vgg11_128":
        x = torch.randn(*shape, generator=generator)
        label = torch.randint(0, 100, (shape[0],), generator=generator)
    elif name == "embedding":
        x = torch.randint(0, 50000, shape, generator=generator)
        label = torch.randint(0, 20, (shape[0],), generator=generator)
    elif name == "transformer":
        x = torch.randint(0, 4096, shape, generator=generator)
        label = torch.randint(0, 4096, (shape[0] * shape[1],), generator=generator)
    else:
        raise ValueError(f"Unsupported case {name}")
    return x, label


def synchronize(device: str):
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "vulkan":
        import torch_vulkan
        torch_vulkan.synchronize()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", choices=CONFIG, required=True)
    ap.add_argument("--backend", choices=("cpu", "cuda", "vulkan"), required=True)
    ap.add_argument("--scale", choices=("tiny", "small", "medium", "reference"), default="small")
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--mode", choices=("inference", "training", "both"), default="both")
    ap.add_argument("--optimizer", choices=("standard", "vulkan_fused"), default="standard")
    ap.add_argument("--zoo", type=Path, required=True)
    ap.add_argument("--shader-dir", type=Path,
                    help="Vulkan-only override for the directory of compiled SPIR-V shaders.")
    ap.add_argument("--output", type=Path)
    args = ap.parse_args()
    shader_dir = None
    if args.shader_dir:
        if args.backend != "vulkan":
            ap.error("--shader-dir requires --backend vulkan")
        args.shader_dir = args.shader_dir.resolve()
        if not args.shader_dir.is_dir() or not any(args.shader_dir.glob("*.spv")):
            ap.error("--shader-dir must contain compiled .spv files")
    assert args.warmup >= 0 and args.repeats >= 1
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    if args.backend == "vulkan":
        import torch_vulkan
        shader_dir = (args.shader_dir or
                      Path(torch_vulkan.__file__).resolve().parents[1] / "csrc/shaders").resolve()
        if args.shader_dir:
            torch_vulkan._C._set_shader_dir(str(shader_dir))
            os.environ["TORCH_VULKAN_SHADER_DIR"] = str(shader_dir)
        device_name = torch_vulkan.device_name()
        assert torch_vulkan.is_hardware_accelerated(), "Software Vulkan is not GPU benchmark"
    elif args.backend == "cuda":
        assert torch.cuda.is_available()
        device_name = torch.cuda.get_device_name()
        # Same float32 algorithm family as Vulkan as far as practical: disable
        # TF32 and reduced-precision GEMM, while allowing cuDNN algorithm selection.
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    else:
        device_name = "CPU"
    torch.manual_seed(42)
    if not (args.zoo / "benchmarks" / "model_zoo.py").exists():
        raise FileNotFoundError(args.zoo)
    sys.path.insert(0, str((args.zoo / "benchmarks").resolve()))
    import model_zoo

    model_name = args.model
    shape = CONFIG[model_name][args.scale]
    model, _, loss_function = model_zoo.MODELS[model_name]()
    model.to(args.backend, dtype=torch.float32)
    generator = torch.Generator().manual_seed(123)
    input_cpu, label_cpu = data_for_case(model_name, shape, generator)
    input_dev, label_dev = input_cpu.to(args.backend), label_cpu.to(args.backend)
    info = {
        "backend": args.backend,
        "device_name": device_name,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "model": model_name,
        "scale": args.scale,
        "input_shape": list(shape),
        "parameters": sum(p.numel() for p in model.parameters()),
        "dtype": "float32",
        "shader_directory": str(shader_dir) if shader_dir else None,
        "shader_directory_override": bool(args.shader_dir),
        "shader_spv_sha256": ({path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in sorted(shader_dir.glob("*.spv"))}
                               if shader_dir else {}),
        "optimizer": args.optimizer,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "threads": args.threads,
        "vulkan_controls": {key: os.environ.get(key, "default") for key in (
            "TORCH_VULKAN_GPU_RESIDENT", "TORCH_VULKAN_DEFER_SUBMISSIONS",
            "TORCH_VULKAN_CONV_ALGO", "TORCH_VULKAN_CONV_BACKWARD_ALGO",
            "TORCH_VULKAN_CONV_TILE", "TORCH_VULKAN_CONV_AUTOTUNE", "TORCH_VULKAN_CONV_WORKSPACE_MB",
            "TORCH_VULKAN_DISABLE_BN_RESIZE", "TORCH_VULKAN_DISABLE_NATIVE_POOL",
            "TORCH_VULKAN_DISABLE_NATIVE_CAT", "TORCH_VULKAN_GPU_PACK_VIEWS",
            "TORCH_VULKAN_DISABLE_SCALAR_BINARY", "TORCH_VULKAN_DISABLE_FENCE_POOL",
            "TORCH_VULKAN_ZERO_REUSED_BUFFERS", "TORCH_VULKAN_ADAMW_FORCE_SYNC",
            "TORCH_VULKAN_DISABLE_NATIVE_SCALAR_ADD")},
        "modes": {},
        "zoo_commit": subprocess.check_output(
            ["git", "-C", str(args.zoo), "rev-parse", "HEAD"], text=True
        ).strip(),
    }

    modes = ("inference", "training") if args.mode == "both" else (args.mode,)
    for mode in modes:
        dispatch_before = torch_vulkan.conv_dispatch_stats() if args.backend == "vulkan" else {}
        try:
            # Reset model from an identical random initialization for both
            # inference and training. This also resets BN running statistics.
            torch.manual_seed(42)
            model, _, loss_function = model_zoo.MODELS[model_name]()
            model.to(args.backend, dtype=torch.float32)
            model.train(mode == "training")
            if mode == "training":
                if args.optimizer == "vulkan_fused":
                    if args.backend != "vulkan":
                        raise RuntimeError("vulkan_fused optimizer only supports Vulkan")
                    from torch_vulkan.optim import AdamW as VulkanAdamW
                    opt = VulkanAdamW(model.parameters(), lr=1e-4)
                else:
                    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, foreach=False)
            samples = []
            first_loss = None
            first_output = None
            first_grad = None
            for i in range(args.warmup + args.repeats):
                synchronize(args.backend)
                start = time.perf_counter()
                if mode == "inference":
                    with torch.inference_mode():
                        pred = model(input_dev)
                else:
                    opt.zero_grad(set_to_none=True)
                    pred = model(input_dev)
                    loss = loss_function(pred, label_dev)
                    loss.backward()
                    opt.step()
                synchronize(args.backend)
                duration = (time.perf_counter() - start) * 1000
                if i == 0:
                    # This transfer is OUTSIDE the timed step.
                    data = pred.detach().cpu().float()
                    first_output = {"shape": list(pred.shape),
                                    "mean": data.mean().item(),
                                    "rms": data.square().mean().sqrt().item()}
                    if mode == "training":
                        first_loss = float(loss.detach().cpu().item())
                if i >= args.warmup:
                    samples.append(duration)
            info["modes"][mode] = {
                "median_ms": round(statistics.median(samples), 3),
                "min_ms": round(min(samples), 3),
                "max_ms": round(max(samples), 3),
                "samples_ms": [round(x, 3) for x in samples],
                "first_loss": first_loss,
                "first_output": first_output,
            }
            if args.backend == "vulkan":
                info["modes"][mode]["conv_dispatch"] = {
                    key: value - dispatch_before.get(key, 0)
                    for key, value in torch_vulkan.conv_dispatch_stats().items()
                    if value != dispatch_before.get(key, 0)
                }
            print(f"{model_name:13s} {mode:9s} {args.backend:6s} "
                  f"median={info['modes'][mode]['median_ms']:9.3f} ms  "
                  f"range=[{min(samples):.3f},{max(samples):.3f}]",
                  flush=True)
        except Exception as e:
            info["modes"][mode] = {"error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-2500:]}
            print(f"{model_name:13s} {mode:9s} {args.backend:6s} FAILED: {str(e).splitlines()[0]}", flush=True)
            break

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(info, indent=2) + "\n")
    if any("error" in v for v in info["modes"].values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
