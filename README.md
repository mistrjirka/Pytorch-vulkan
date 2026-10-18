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

## Verified functionality (October 18, 2026)

On **PyTorch 2.14.1+cpu**:

| Driver | Test results |
|---|---:|
| RTX 2080 Ti (NVIDIA 580.178.04) | 361/361 passed |
| Tesla V100-SXM2 (NVIDIA 580.178.04) | 361/361 passed |
| Lavapipe software Vulkan | 360 passed, one hardware-only test skipped |

Supported, tested operations include FP32 GEMM (including transposed inputs),
elementwise add/multiply, ReLU and GELU forward/backward, LayerNorm with
correct saved statistics, storage-sharing `view`/`as_strided`,
native single-axis FP32 sum, larger-logit log-softmax forward/backward,
fused GEMM+bias for Linear layers, and **native Vulkan Conv2D forward,
dInput, dWeight, dBias** with
stride/padding/dilation/group variants. Conv2D also accepts FP32 channels-last,
transposed, sliced, offset and broadcast views via GPU packing. Native FP32
BatchNorm training/evaluation and nearest-neighbor 2D resize include backward
and strided-view support. See the [BatchNorm and resize report](docs/batch-norm-and-nearest-resize.md)
for semantics, matched A/B timings and remaining CUDA gaps. The
[BatchNorm precision investigation](docs/batch-norm-precision.md) covers stable
centering, two-value gradients and the ResNet FP32/FP64 comparison. Native FP32 max-pool
forward/backward, concatenation and arithmetic view packing now cover general
layouts. Completed-fence reuse removes repeated driver allocation overhead.
The [pooling, layout and fence report](docs/pooling-layouts-and-fences.md) records
U-Net inference improving from 39.20 to 3.32 ms and full training from 86.38 to
17.20 ms, compared with CUDA at 1.08/9.48 ms on the same RTX 2080 Ti. Those
measurements use opt-in residency, deferred submission and fused AdamW. The
[next runtime iteration](docs/runtime-allocation-and-scalar-updates.md) removes
reused-buffer clearing and CPU scalar-counter updates, and defers AdamW with
the rest of the graph. Fresh interleaved runs reduce inference 3.28→2.51 ms
and training 19.16→14.96 ms (CUDA 1.09/9.56 ms). Other operations
can fall back to CPU. See the [convolution backward and layout report](docs/convolution-backward-and-layouts.md)
for matched CUDA/Vulkan profiles, general dispatch and current limitations.

The simple SGD example at `examples/linear_train.py` reduces MSE from
4.6667 to 0.7864 in five steps.

### General convolution selection

Convolution now selects from explicit capability/workspace-checked candidates
for forward and each gradient. Three shared tile configurations preserve exact
runtime filter dimensions, including 10x10 and 11x11. Optional
`TORCH_VULKAN_CONV_AUTOTUNE=1` measures and caches choices by operator geometry
and device; it is disabled by default because small workloads can regress.
See [candidate dispatch, validation and matched CUDA measurements](docs/convolution-candidate-dispatch.md).

### Performance investigation and experimental optimizations

The [18 October 2026 investigation](docs/performance-investigation-2026-10-18.md)
profiles CPU fallbacks and Vulkan GPU activity, documents the optimizer,
allocation and submission fixes, and explains why the remaining CUDA gap
motivated the now-implemented shared pipeline cache and experimental GPU residency.

The opt-in `torch_vulkan.optim.AdamW` fuses optimizer state updates
on Vulkan rather than repeatedly falling back to CPU. It does **not** alter
`torch.optim.AdamW`. Deferred command submission and mapped host memory are
additional *experimental* environment-configured modes, disabled by default.

### Earlier optimized MLP measurement (experimental)

For the 1.87M-parameter XDNA2 MLP, batch 128 on the RTX 2080 Ti,
optimized Vulkan now takes 4.614 ms inference and 17.166 ms for a
complete forward + loss + backward + AdamW training step, against native CUDA
0.172 ms and 2.218 ms respectively. This requires GPU residency and the
opt-in fused AdamW implementation, and is still substantially slower than CUDA.

Set `TORCH_VULKAN_GPU_RESIDENT=1` before importing the backend, then use
`from torch_vulkan.optim import AdamW` rather than
`torch.optim.AdamW` to use the fused optimizer. The experimental residency
mode requires more validation for arbitrary PyTorch workloads and is not
enabled by default.

Full methodology, GPU kernel timestamps, fallback profiling, results
on both NVIDIA GPUs, and outstanding bottlenecks are documented in the
**[latest CUDA/Vulkan bottleneck analysis](docs/cuda-vulkan-bottleneck-analysis.md)**.

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
- Conv2D selects general tiled/depthwise/reduction paths with scalar fallbacks.
  It still lacks cuDNN's broader algorithm selection (such as Winograd), and
  layout packing adds work. Conv2D supports 2D, nonempty FP32 configurations;
  this does not imply layout/dtype coverage for every PyTorch operator.
- Several loss operators, attention, some reductions, pooling families other than max-pool2d,
  and other unsupported ops still use CPU fallback. Device residency is
  experimental and opt-in; not all workflows stay fully on GPU.
- LSTM fails because the fused LSTM cell implementation is missing.
  `ConvTranspose2d` is also not supported. A reduced-size ResNet18
  training comparison remains sensitive to FP32 rounding in its final
  two-value BatchNorm reductions, including on CUDA. Local BatchNorm precision
  defects are fixed and compared with FP64; this is **not a validation of
  ResNet18 training convergence**. See the precision investigation above.
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
