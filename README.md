# PyTorch Vulkan (experimental)

A first-class PyTorch `PrivateUse1` Vulkan device for GPU inference and, eventually,
training. **Currently an experimental bring-up, not a production backend.**

This work begins from [Peter Clemente III's MIT-licensed torch-vulkan](https://github.com/Peterc3-dev/torch-vulkan)
inference backend; see [third-party notices](THIRD_PARTY_NOTICES.md).
It is being modernized for **PyTorch 2.14.1** and expanded toward training.

## Setup (Linux)

Requires Python 3.12+, a working Vulkan ICD, Vulkan SDK (`libvulkan-dev`),
CMake 3.18+, a C++20 compiler, and `glslangValidator` (package `glslang-tools`).
The build currently uses [Kompute](https://github.com/KomputeProject/kompute),
which CMake fetches if not installed.

```sh
python3 -m venv .venv
. .venv/bin/activate
pip install 'torch==2.14.1' --index-url https://download.pytorch.org/whl/cpu
pip install -U setuptools wheel pytest
bash csrc/shaders/compile.sh
pip install -e . --no-build-isolation
pytest -q tests/
python examples/linear_train.py
```

```python
import torch
import torch_vulkan
assert torch_vulkan.is_available()
a = torch.randn(32, 16).to('vulkan')
b = torch.randn(16, 8).to('vulkan')
print((a @ b).cpu())
```

## Verified functionality (2026-10-17)

PyTorch **2.14.1+cpu**, on Linux / Lavapipe (software Vulkan):

- 23 tests passing: transfers, FP32 add/multiply, GEMM, ReLU backward,
  autograd gradients through multiply/GEMM, and miscellaneous inference ops.
- Simple SGD training loop runs: MSE decreased from **4.6667 to 0.7864** in
  five steps on the test problem. Unsupported loss/reduction/optimizer ops
  currently execute through CPU fallback.
- Backend reports physical driver name and `is_hardware_accelerated()`.
  The development sandbox currently selects llvmpipe; no physical-GPU
  correctness or performance claims can be made yet.
- PyTorch includes a separate legacy `tensor.is_vulkan` property that does **not**
  reflect the custom device. Use `tensor.device.type == "vulkan"` instead.

## Important limitations

- Float32 inference-focused kernels, incomplete operator, dtype, and layout coverage.
- Not yet a generally supported `torch.nn.Module` training device: backward kernels
  and shape/stride semantics are being implemented and verified.
- Current executor performs frequent host/device synchronization. Benchmarks against
  CUDA/ROCm would be misleading until asynchronous scheduling and memory residency
  are fixed.
- A `vulkan` tensor is not necessarily GPU-resident: check the selected driver
  (Lavapipe is a software Vulkan device).

## Implementation

`torch_vulkan` is a C++ PyTorch extension with `PrivateUse1` dispatch,
Kompute-managed Vulkan buffers and GLSL/SPIR-V compute shaders in `csrc/shaders/`.
No PyTorch fork is necessary. The tested subset is validated on PyTorch 2.14.1;
broader operator coverage and hardware-driver correctness remain unverified.
