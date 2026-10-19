# Portable FP32 convolution on AMD Vulkan — 19 October 2026

## Compatibility and limitations

Conv2D forward, dInput, dWeight and dBias use generic FP32 GLSL/SPIR-V.
The portable tiled kernels use workgroup-shared memory and uniform barriers,
not NVIDIA CUDA, FP16 downcasts, CUDA Tensor Cores, or a fixed subgroup width.
The dispatcher in csrc/ops/convolution.cpp checks Vulkan workgroup, shared
memory, grid and buffer limits, and a generic direct/tiled fallback remains
available for unsupported optimizations.

**Portable by design does not establish correctness or speed on AMD silicon.**
The physical GPUs currently accessible in Development Sandbox are RTX 2080 Ti
and Tesla V100; Lavapipe supplies a third software Vulkan implementation.
Attempts to access the connected AMD host were unsuccessful. In particular,
we have not measured the RX 7800 XT/RADV or Radeon 880M.

AMD's performance guide recommends coalescing, LDS tiling and portable
workgroup sizes; RDNA uses wave32 and older GCN wave64. The new shader
does not assume either wave size.

References:
- https://gpuopen.com/learn/rdna-performance-guide/
- https://docs.vulkan.org/guide/latest/compute_shaders.html
- https://archlinux.org/packages/extra/x86_64/vulkan-radeon/files/

## Implemented portable optimization

The compiled register-tile FP32 3×3 input-gradient shader now
hoists spatial batch/row/column quotient/remainder calculations
outside its K reduction loop, since each invocation's N coordinate
is invariant across the reduction. A compile-time assertion verifies
THREADS % TN == 0 for this path.

The change is enabled via CONV_HOIST_SPATIAL only in the
conv2d_backward_input_tiled_register3.comp wrapper. Every other compiled
convolution shader is byte-identical to the baseline; all shapes, strides,
dilations and groups retain runtime semantics.
No vendor-specific extension, new algorithm family, proprietary source,
or external Winograd code is used.

The first broader experiment altered every tile and regressed grouped
convolutions. A separate 3×3 forward variant showed no stable benefit,
so both experiments were discarded. Only the input-gradient specialization
is retained. Compiled SPIR-V passes spirv-val.

### Cross-platform numerical testing

- Strict 139-case convolution test suite: **139 passed per device**
  on RTX 2080 Ti, Tesla V100 and Lavapipe, in both eager and deferred mode.
- Full repository regression suite: **3,685 passed** each on RTX/V100;
  **3,684 passed and one existing skip** on Lavapipe.
- Strict CPU numerical fallback rejection was enabled for focused tests,
  and unchanged CPU FP64 forward/gradient oracle comparisons passed.
- **No physical AMD GPU has been tested.** SIMD/wave subgroup widths are not
  hardcoded; vendor-specific compiler performance is unknown.

### Performance results

Isolated A/B benchmarks used two independent process medians per variant,
interleaving baseline and candidate, with CPU float64 oracle parity.
The dense large 3×3 (2×32×64×64, Cout=64) input-gradient results were:

| GPU | Phase | Previous (ms) | Candidate (ms) |
|---|---|---:|---:|
| RTX 2080 Ti | Input gradient | 0.240 | 0.223 |
| Tesla V100 | Input gradient | 0.208 | 0.188 |

A separate complete large U-Net A/B test with 18 timed steps, 6
warmups and two independent processes per variant found:

- Baseline Vulkan training **51.66 ms**, inference **9.02 ms**.
- DInput-only specialization training **51.40 ms**, inference **9.07 ms**.

This is a modest (~0.5%) training improvement. The inference change is
within normal run-to-run variation; no inference gain is claimed.
All four processes passed unchanged CPU output/loss/all-gradient parity,
with zero recorded CPU fallback. The goal of at most 1.5× CUDA remains
**unmet**. Raw CUDA/Vulkan comparisons are documented separately in
convolution-kernel-and-dispatch-profiling-2026-10-19.md. No AMD
performance measurement is available.

The [A/B evidence](../benchmarks/results/amd-portable-conv-2026-10-19/summary.json)
and [complete process records](../benchmarks/results/amd-portable-conv-2026-10-19/raw.jsonl.gz)
include the original and new SPIR-V hashes for reproducibility.

## Run the physical AMD test on Arch Linux

Current Arch Linux vulkan-radeon provides
/usr/share/vulkan/icd.d/radeon_icd.json, not the obsolete
radeon_icd.x86_64.json filename.

~~~bash
sudo pacman -S --needed vulkan-radeon vulkan-icd-loader vulkan-tools
VK_DRIVER_FILES=/usr/share/vulkan/icd.d/radeon_icd.json vulkaninfo --summary
~~~

After building the backend in this repository using README.md:

~~~bash
VK_DRIVER_FILES=/usr/share/vulkan/icd.d/radeon_icd.json \
  bash scripts/verify_amd_convolution.sh
~~~

The script refuses a non-AMD or software device, enforces CPU-fallback
rejection, then runs the 139-case convolution numerical suite.
Optionally collect an AMD-only performance baseline:

~~~bash
AMD_VULKAN_BENCH_OUTPUT=/tmp/amd-fp32-conv.json \
VK_DRIVER_FILES=/usr/share/vulkan/icd.d/radeon_icd.json \
  bash scripts/verify_amd_convolution.sh
~~~

AMD performance should ideally be compared with the same Radeon GPU running
ROCm on identical FP32 geometry and training phases, rather than compared
across unrelated physical GPU models.

## Next algorithms to implement and gate

- General FP32 Winograd 3×3 forward and both gradients, selected only when
  the complete transform/matrix/inverse pipeline beats tiled implicit GEMM.
- Reduce GPU-side patch-gather/indexing work in split weight gradients.
- Reduce Vulkan descriptor and command recording overhead, maintaining
  correctness and lifetime of in-flight resources.
- Preserve a universal FP32 fallback, device-limit checks, full numerical
  tests and measured dispatch choices on each AMD/NVIDIA architecture.

Neither Winograd nor Vulkan descriptor-buffer optimizations are implemented
in this incremental change.
