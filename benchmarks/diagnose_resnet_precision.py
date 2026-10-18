"""Diagnose FP32 ResNet sensitivity using matching CPU FP32/FP64 references.

Hooks intentionally synchronize every leaf output; this is not a speed test.
The actual torchvision architecture is imported from the provided model zoo.
Repeated leaf calls retain separate entries: "layer.relu", "layer.relu#2", etc.
"""
import argparse
from collections import defaultdict
import copy
import hashlib
import importlib
import json
from pathlib import Path
import subprocess
import sys

import torch
import torch.nn.functional as F


MODEL_SEED = 17
INPUT_SEED = 1


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zoo", type=Path, required=True)
    parser.add_argument("--backend", choices=["cuda", "vulkan"], required=True)
    parser.add_argument("--size", type=int, default=32)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--shader-dir", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.size < 1 or args.batch < 1:
        parser.error("--size and --batch must be positive")
    return args


def git_revision(directory):
    result = subprocess.run(
        ["git", "-C", str(directory), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def input_state_digest(x, target, model):
    digest = hashlib.sha256()
    for tensor in (x, target, *model.state_dict().values()):
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def batch_norm_same_input_error(module, input_cpu, output_cpu):
    """Separate local BN rounding from error accumulated by earlier layers."""
    reference = F.batch_norm(
        input_cpu.double(), None, None,
        module.weight.detach().cpu().double(),
        module.bias.detach().cpu().double(),
        True, 0, module.eps).float()
    variance = input_cpu.double().var(dim=(0, 2, 3), unbiased=False)
    return {
        "err_vs_double_same_input": (output_cpu - reference).abs().max().item(),
        "min_var": variance.min().item(),
        "max_inv": torch.rsqrt(variance + module.eps).max().item(),
        "shape": list(input_cpu.shape),
    }


def run_model(base, x, target, device, dtype, diagnose_batch_norm):
    model = copy.deepcopy(base).to(device=device, dtype=dtype)
    activations = {}
    local_bn_errors = {}
    invocation_counts = defaultdict(int)
    handles = []

    def make_hook(name):
        def capture(module, inputs, output):
            invocation_counts[name] += 1
            invocation = invocation_counts[name]
            key = name if invocation == 1 else f"{name}#{invocation}"
            # Clone before a later in-place operation can mutate this activation.
            input_cpu = inputs[0].detach().cpu().clone()
            output_cpu = output.detach().cpu().clone()
            activations[key] = output_cpu
            if diagnose_batch_norm and isinstance(module, torch.nn.BatchNorm2d):
                local_bn_errors[key] = batch_norm_same_input_error(
                    module, input_cpu, output_cpu)
        return capture

    for name, module in model.named_modules():
        if name and not list(module.children()):
            handles.append(module.register_forward_hook(make_hook(name)))
    try:
        output = model(x.to(device=device, dtype=dtype))
        loss = F.cross_entropy(output, target.to(device))
        loss.backward()
        gradients = {
            name: parameter.grad.cpu().double()
            for name, parameter in model.named_parameters()
        }
        result = {"loss": loss.item(), "bn": local_bn_errors}
    finally:
        for handle in handles:
            handle.remove()
    return result, activations, gradients


def gradient_summary(actual, reference):
    """L2 over every parameter element; not an average of per-layer ratios."""
    if actual.keys() != reference.keys():
        raise ValueError("Parameter names differ between runs")
    error_squared = 0.0
    reference_squared = 0.0
    max_abs = 0.0
    elements = 0
    for name, value in actual.items():
        expected = reference[name]
        difference = value.double() - expected.double()
        error_squared += difference.square().sum().item()
        reference_squared += expected.double().square().sum().item()
        max_abs = max(max_abs, difference.abs().max().item())
        elements += value.numel()
    return {
        "relative_l2": (error_squared / max(reference_squared, 1e-30)) ** 0.5,
        "error_l2": error_squared ** 0.5,
        "reference_l2": reference_squared ** 0.5,
        "max_abs": max_abs,
        "parameters": len(actual),
        "elements": elements,
    }


def compare_to_double(activations, gradients, double_activations, double_gradients):
    if activations.keys() != double_activations.keys():
        raise ValueError("Leaf invocation names differ between runs")
    layers = {}
    for name, value in activations.items():
        difference = value.double() - double_activations[name]
        layers[name] = {
            "abs": difference.abs().max().item(),
            "rms": difference.square().mean().sqrt().item(),
        }
    parameter_errors = {}
    for name, value in gradients.items():
        reference = double_gradients[name]
        difference = value - reference
        relative_l2 = (
            difference.square().sum() /
            reference.square().sum().clamp_min(1e-30)
        ).sqrt().item()
        parameter_errors[name] = {
            "abs": difference.abs().max().item(),
            # Keep the existing field for readers of earlier reports.
            "rms_rel": relative_l2,
            "relative_l2": relative_l2,
            "ref_max": reference.abs().max().item(),
        }
    return {
        "layers_vs_double": layers,
        "grads_vs_double": parameter_errors,
        "gradient_summary_vs_double": gradient_summary(gradients, double_gradients),
        "layer_summary_vs_double": {
            "max_abs": max(row["abs"] for row in layers.values()),
            "worst_layer": max(layers, key=lambda name: layers[name]["abs"]),
            "leaf_invocations": len(layers),
        },
    }


def print_summary(rows, backend, output_path):
    print(output_path)
    print("loss", [(label, rows[label]["loss"]) for label in ("cpu", "double", backend)])
    for label in ("cpu", backend):
        gradients = rows[label]["gradient_summary_vs_double"]
        layers = rows[label]["layer_summary_vs_double"]
        print(
            f"{label} vs FP64: gradient relative L2={gradients['relative_l2']:.6g}, "
            f"gradient max abs={gradients['max_abs']:.6g}, "
            f"worst activation={layers['worst_layer']} "
            f"({layers['max_abs']:.6g})"
        )
    for name, error in rows["pair"]["layers"].items():
        print(name, error, rows[backend]["bn"].get(name, ""))
    worst_gradients = sorted(
        rows["pair"]["grads"].items(), key=lambda item: item[1], reverse=True)[:5]
    print("worst gradients", worst_gradients)
    print("whole-network gradient vs CPU", rows["pair"]["gradient_summary"])
    if "conv1.weight" in rows[backend]["grads_vs_double"]:
        print("double ref gradient",
              rows[backend]["grads_vs_double"]["conv1.weight"],
              rows["cpu"]["grads_vs_double"]["conv1.weight"])


def main():
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    zoo = args.zoo.resolve()
    sys.path.insert(0, str(zoo / "benchmarks"))
    model_zoo = importlib.import_module("model_zoo")

    if args.backend == "vulkan":
        import torch_vulkan
        if args.shader_dir:
            torch_vulkan._C._set_shader_dir(str(args.shader_dir.resolve()))
        device_name = torch_vulkan.device_name()
    else:
        device_name = torch.cuda.get_device_name()

    torch.set_num_threads(6)
    torch.manual_seed(MODEL_SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    base = model_zoo.MODELS["resnet18"]()[0].train()
    generator = torch.Generator().manual_seed(INPUT_SEED)
    x = torch.randn(args.batch, 3, args.size, args.size, generator=generator)
    target = torch.randint(100, (args.batch,), generator=generator)

    shader_dir = args.shader_dir or root / "csrc/shaders"
    metadata = {
        "backend": args.backend,
        "shape": list(x.shape),
        "input_state_sha256": input_state_digest(x, target, base),
        "torch": torch.__version__,
        "device": device_name,
        "commit": git_revision(root),
        "zoo_commit": git_revision(zoo),
        "model_seed": MODEL_SEED,
        "input_seed": INPUT_SEED,
        "tf32": False,
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
        "bn_shader_sha256": {
            name: hashlib.sha256((shader_dir / (name + ".spv")).read_bytes()).hexdigest()
            for name in ("batch_norm_forward", "batch_norm_backward")
        },
        "note": "Leaf hooks synchronize device reads; numerical diagnosis only, not timing.",
        "leaf_invocation_keys": "First call uses module name; later calls append #2, #3, etc.",
    }
    rows = {"metadata": metadata}
    outputs = {}
    gradients = {}
    configurations = (
        ("cpu", torch.float32, "cpu"),
        ("cpu", torch.float64, "double"),
        (args.backend, torch.float32, args.backend),
    )
    for device, dtype, label in configurations:
        result, activations, parameter_gradients = run_model(
            base, x, target, device, dtype, diagnose_batch_norm=device == args.backend)
        rows[label] = result
        outputs[label] = activations
        gradients[label] = parameter_gradients

    for label in ("cpu", args.backend):
        rows[label].update(compare_to_double(
            outputs[label], gradients[label], outputs["double"], gradients["double"]))
    rows["pair"] = {
        "layers": {
            name: (value - outputs["cpu"][name]).abs().max().item()
            for name, value in outputs[args.backend].items()
        },
        "grads": {
            name: (value - gradients["cpu"][name]).abs().max().item()
            for name, value in gradients[args.backend].items()
        },
        "gradient_summary": gradient_summary(gradients[args.backend], gradients["cpu"]),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows, indent=2) + "\n")
    print_summary(rows, args.backend, args.out)


if __name__ == "__main__":
    main()
