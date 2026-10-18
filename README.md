# PyTorch Vulkan (experimental)

A real Vulkan compute backend for **PyTorch 2.14.1** using `PrivateUse1`
dispatch. It supports a growing subset of **FP32 training and inference**
without requiring a PyTorch fork, CUDA, or ROCm.

The project originated from [Peter Clemente III's MIT-licensed
torch-vulkan](https://github.com/Peterc3-dev/torch-vulkan). See
[third-party notices](THIRD_PARTY_NOTICES.md). This is still an experimental
backend, **not a drop-in replacement for CUDA or ROCm**.

## Installation (Linux)

Requires Python 3.12+, a C++20 compiler, CMake 3.18+, Vulkan SDK headers,
an actual Vulkan ICD, and `glslangValidator`.

```sh
# Ubuntu 24.04; adapt package names on Arch.
sudo apt-get install cmake g++ libvulkan-dev libvulkan1 glslang-tools \
    vulkan-tools libegl1 libgl1 libxext6

python3 -m venv .venv
. .venv/bin/activate
python -m pip install --index-url https://download.pytorch.org/whl/cpu 'torch==2.14.1'
python -m pip install setuptools wheel pytest numpy
bash csrc/shaders/compile.sh
python -m pip install -e . --no-build-isolation --no-deps
python -m pytest -q tests
```

```python
import torch
import torch_vulkan

print(torch_vulkan.device_name())
assert torch_vulkan.is_hardware_accelerated()  # Excludes software Lavapipe.
x = torch.randn(32, 16).to("vulkan")
w = torch.randn(16, 8).to("vulkan").requires_grad_()
y = x @ w
y.sum().backward()
print(y.cpu(), w.grad.cpu())
```

The Python wheel can be **CPU-only PyTorch**: the custom backend dispatches
SPIR-V through the Vulkan driver. `torch.cuda.is_available()` does not
indicate Vulkan support.

### Selecting a physical GPU

The backend currently exposes **one logical** `vulkan:0` device per process.
Choose which physical device backs it **before importing the module**:

```sh
vulkaninfo --summary
VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json \
TORCH_VULKAN_PHYSICAL_DEVICE_INDEX=1 python examples/linear_train.py
```

In the development Sandbox, Vulkan device 0 is the **NVIDIA RTX 2080 Ti**;
device 1 is the **Tesla V100**. The names and indices depend on your driver.
`torch_vulkan.device_name()` reports the actual device. Minimal NVIDIA
containers can require `libegl1`, `libgl1`, and `libxext6` for the ICD
to load.

## Verified functionality (October 17, 2026)

On **PyTorch 2.14.1+cpu**:

| Driver | Test results |
|---|---:|
| RTX 2080 Ti (NVIDIA 580.178.04) | 37/37 passed |
| Tesla V100-SXM2 (NVIDIA 580.178.04) | 37/37 passed |
| Lavapipe software Vulkan | 36 passed, hardware-only test skipped |

Supported, tested operations include FP32 GEMM (including transposed inputs),
elementwise add/multiply, ReLU and GELU forward/backward, LayerNorm with
correct saved statistics, storage-sharing `view`/`as_strided`,
and **native Vulkan Conv2D forward, dInput, dWeight, dBias** with
stride/padding/dilation/group variants. Other operations can fall back to CPU.

The simple SGD example at `examples/linear_train.py` reduces MSE from
4.6667 to 0.7864 in five steps.

### Performance investigation and experimental optimizations

The [18 October 2026 investigation](docs/performance-investigation-2026-10-18.md)
profiles CPU fallbacks and Vulkan GPU activity, documents the optimizer,
allocation and submission fixes, and explains why the remaining CUDA gap
requires a new pipeline/descriptor cache and persistent GPU residency.

The opt-in `torch_vulkan.optim.AdamW` fuses optimizer state updates
on Vulkan rather than repeatedly falling back to CPU. It does **not** alter
`torch.optim.AdamW`. Deferred command submission and mapped host memory are
additional *experimental* environment-configured modes, disabled by default.

### Real GPU performance versus CUDA

See **[Vulkan vs CUDA vs CPU benchmarks](benchmarks/GPU_COMPARISON.md)**
for eight XDNA2 model-zoo architectures and per-stage optimizer profiling.
Native CUDA is currently far faster on the RTX 2080 Ti, particularly for
complete AdamW training (CPU fallbacks and staging dominate). The larger-image
StyleNet 36×36 inference took 97.2 ms on Vulkan versus 1.53 ms on CUDA.
CUDA 13 / PyTorch 2.14.1 cannot execute on V100 (SM70); both physical GPUs
are still tested for Vulkan correctness.

### Real GPU testing against the latest XDNA2 model zoo

We import model definitions from
[mistrjirka/pytorch-XDNA2](https://github.com/mistrjirka/pytorch-XDNA2)
(commit `c7c063101ec5209caf8022f5b4ea6948d1a8d6ed`), using
**smaller FP32 synthetic batches** because the upstream benchmark runs BF16.
These are functional probes, **not the original full-size benchmark**.
All four models complete forward, backward, and an AdamW parameter update.

| Model | Test input | RTX 2080 Ti forward + backward | Max output error vs CPU |
|---|---|---:|---:|
| MLP | 32×784 | 34 ms | 2.2e-6 |
| U-Net | 1×3×16×16 | 109 ms | 3.9e-6 |
| Embedding | 16×64 token IDs | 280 ms | 3.0e-8 |
| StyleNet (all five residual blocks) | 1×3×8×8 | 284 ms | 1.1e-5 |

Times are illustrative warm-run medians (RTX, PyTorch FP32). CPU and GPU
are not benchmarked under tightly controlled conditions; GPU performance is
**currently slower than CPU** for these small cases. See
[measurements](benchmarks/results/rtx2080ti_model_zoo_fp32.json) and
[reproduction instructions](benchmarks/README.md).

The previously dominant `aten::t` host-side transpose caused a 1.27 s
forward+backward time for a 32-sample MLP; metadata-only transpose plus
stride-aware GEMM reduced it to around **31–38 ms**. Native GELU backward and
a closer-to-exact GELU further improved correctness.

## Limitations

- **FP32 only for compute kernels.** BF16 and FP16 training are not supported.
- Direct convolution uses a **correctness-first** scalar implementation,
  not an optimized tensor-core or implicit-GEMM kernel. It will be slow for
  large channel counts.
- Many activations, losses, reductions, batchnorm backward, AdamW parts, and
  other ops use CPU fallback, with staging and synchronization overhead.
  The backend is **not fully GPU-resident**.
- LSTM fails because the fused LSTM cell implementation is missing.
  `ConvTranspose2d` is also not supported. A reduced-size ResNet18
  training comparison exhibited numerical divergence near small spatial
  BatchNorm layers; it is **not validated**.
- Only the two NVIDIA GPUs and software Lavapipe have been tested;
  AMD/Intel portability still needs verification.
- `torch.Tensor.is_vulkan` refers to legacy PyTorch Vulkan support and
  is **not** this `PrivateUse1` backend; use
  `tensor.device.type == "vulkan"`.
- CPU fallback for MSE loss emits an output-resize deprecation warning.
- Avoid nonzero storage offsets or arbitrary strides in custom kernels unless
  the kernel explicitly handles them; the implemented copies and GEMM support
  the validated layouts.

## Structure

`torch_vulkan`: device registration/runtime hooks.
`csrc/ops/`: PyTorch operator registration and execution.
`csrc/shaders/`: Vulkan GLSL and compiled SPIR-V.
`benchmarks/`: tests against external model definitions.
`tests/`: PyTorch numerical and gradient regression tests.
The Vulkan buffer runtime currently uses
[Kompute](https://github.com/KomputeProject/kompute) (Apache 2.0).
Set `TORCH_VULKAN_TRACE_FALLBACK=1` to print CPU-fallback calls and latency.
