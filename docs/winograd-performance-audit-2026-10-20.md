# Vulkan Winograd audit: code, scheduling, numerical safety, and references (20 October 2026)

## Result and disposition

The existing three-stage FP32 F(2x2,3x3) forward path is mathematically sound on the tested devices/shapes, but initially made **three independently synchronized device-local GPU submissions**, a measurable execution antipattern. The new generic VulkanContext::run_compute_chain() records all stages and the required shader-write/read barriers into one submission. The default FP32 Winograd path no longer needs performance flags, and only takes over a conservative reusable class of grouped 3x3 convolutions. All other shapes retain existing implicit GEMM/depthwise/direct candidates.

The algorithm remains forward-only. This is an explicit architectural limitation: dInput and dWeight use their existing GPU algorithms. It cannot on its own meet the full-training or end-to-end <=1.5x CUDA target.

## Findings from implementation review

1. **Three CPU/GPU fences in the eager device-local path — fixed.** The previous code called run_compute() for filter transform, input transform, and multiply/inverse transform, each flushing and waiting. In nonresident mode it also transferred two intermediate buffers to/from the CPU, despite no CPU use. The new run_compute_chain() records initial staging uploads once, then three compute dispatches with shader RAW barriers, downloads only final output in explicitly nonresident mode, and correctly tracks intermediate/result GPU residency when resident. eHost/deferred mode already records three ordered GPU dispatches with memory barriers; that path remains unchanged.
2. **Weight-transform amortization.** Each forward transforms weights once. Immutable inference graphs could cache the result across forwards. **Do not cache by raw pointer**: PyTorch optimizer updates, in-place mutations, aliases and storage reuse would make the cache stale; maintainable inference-only caching needs reliable tensor-version/storage lifetime tracking. ncnn performs offline/model-load weight preprocessing, an inference-specific optimization.
3. **GEMM tile underutilization.** The current fused output shader computes an 8x8x8 Winograd-domain matrix tile with 16 scalar accumulators per lane, 8 KiB shared buffers and 16 coefficient planes. This is deliberately portable FP32, but smaller than tuned blocked GEMM; no subgroup-shuffle/cooperative-matrix variant, double buffering, prefetching, channel pack4 or explicit global-memory layout search. ncnn offers pack4 specialized shaders, local-memory variants, cooperative-matrix kernels and separate GEMM/output transform. A GPU-portable path must stay available.
4. **Memory overhead.** F(2,3) explicitly materializes 16 FP32 coefficients per input 2x2 output tile and per filter. This is ~4 transformed input floats per output pixel/channel (ignoring borders), plus transformed filter and normal output. A 64 MiB configurable workspace limit safely rejects inapplicable shapes, but the transform buffers still stress allocation/traffic. F(4,3) uses 36 coefficients per 4x4 output tile, ~2.25 floats per output pixel/channel, at the cost of heavier transforms and possible FP32 accuracy loss. Explore both under numerical gates, not a universal switch.
5. **Portability guard — fixed.** The Winograd candidate now verifies at least four per-stage/set storage buffer descriptors, 64 invocations, 8 KiB shared memory, 64-byte push constants, workgroup size/grids, transformed tensor indexing bounds and available workspace. No GPU vendor name controls correctness.
6. **Numerical risk.** Winograd transforms can have cancellation and roundoff relative to implicit GEMM. Current FP32 F(2,3) passed the independent CPU float64 oracle for random, grouped, odd-shaped and padded cases, including no-bias, in-place weight updates, and the stress test at input scales 1,100,10000. For scale 10000, max error vs CPU double was 0.00171 Winograd versus 0.00293 old tiled on the sampled high-offset, zero-sum-weight case; mean errors were 0.000164 and 0.000139 respectively. This is NOT a blanket stability guarantee, particularly for a future F(4,3) version.
7. **Heuristic/model overfitting.** Dispatch is based on filter, stride, dilation, channels per group, group count, spatial work and device/workspace capability, not network identity. On tested RTX 2080 Ti and Tesla V100, a useful guarded class is grouped 3x3 with >=4 groups, >=16 input/output channels per group and >=2048 output positions, provided the workspace fits. Ungrouped, tiny, depthwise and insufficient-workspace cases keep their existing paths. AMD physical-device performance remains **unverified**; do not assume the same speedups on Radeon.
8. **Autotune cost.** Explicit per-shape autotuning includes Winograd's scratch allocation, which is correct for timing but startup-heavy. It stays OFF by default. We must avoid model-dependent overrides or enabling slow first-use autotuning unconditionally.

## Controlled performance: same-process legacy vs chained submission

RTX 2080 Ti, FP32, identical resident tensor allocations, alternating submission path within a process (ABBA/BAAB), explicit synchronize after each op, independent float64 CPU oracle, zero CPU fallback:

| Geometry | Three submits | One chain | Speedup |
|---|---:|---:|---:|
| Grouped medium | 0.19239 ms | 0.18518 ms | 1.039x |
| Grouped large | 0.29070 ms | 0.28340 ms | 1.026x |
| Grouped small | 0.11481 ms | 0.10719 ms | 1.071x |
| Dense small forced Winograd | 0.12566 ms | 0.11992 ms | 1.048x |
| Odd-size forced Winograd | 0.11458 ms | 0.10646 ms | 1.076x |

Tesla V100:

| Geometry | Three submits | One chain | Speedup |
|---|---:|---:|---:|
| Grouped medium | 0.19770 ms | 0.18941 ms | 1.044x |
| Grouped large | 0.29291 ms | 0.28538 ms | 1.026x |
| Grouped small | 0.10958 ms | 0.10241 ms | 1.070x |

Independent shape checks comparing generic tiled vs Winograd:

| Geometry | RTX tiled | RTX Winograd | V100 tiled | V100 Winograd |
|---|---:|---:|---:|---:|
| 8 groups, Cin/Cout=128, 64x64 batch2 | 0.462 | 0.281 | 0.414 | 0.286 |
| 8 groups, Cin/Cout=128, 128x128 batch1 | 0.843 | 0.469 | 0.742 | 0.490 |
| 4 groups, Cin/Cout=128, 64x64 batch2 | 0.829 | 0.432 | 0.735 | 0.424 |

These are matched isolated operations, **not** whole-model CUDA parity. Earlier dense large 3x3 measurements showed Winograd regressions, so dense forward continues to use the faster tiled path.

Raw results: benchmarks/results/winograd-three-stage-20261020/ (untracked experiment artifacts). Reproduction scripts: benchmarks/probe_winograd_forward.py, benchmarks/compare_winograd_submissions.py.

## Validation

- Full ordinary RTX 2080 Ti repository suite: 3,932 tests passed, 14 warnings.
- Winograd focused tests: 9 passed on RTX 2080 Ti, 9 passed on V100, 9 passed on Lavapipe software Vulkan.
- Default device-local, eager GPU-resident and explicitly nonresident execution: 7 original Winograd tests passed in each mode.
- Forward output, native gradients and negative tests for unsupported filters/workspace were checked. No-bias/in-place weight-update tests ensure transformed weights are not silently reused.
- Stage scratch shaders validated by spirv-val; numerical fallback forbidden in focused tests.

## Better algorithms/implementations worth evaluating

**ncnn, Tencent, BSD-3-Clause**:
https://github.com/Tencent/ncnn/blob/master/src/layer/vulkan/convolution_vulkan.cpp
https://github.com/Tencent/ncnn/blob/master/src/layer/vulkan/shader/convolution_3x3s1d1_winograd_gemm.comp

The Vulkan backend supports both F(2,3) and F(4,3), split input transform / GEMM / output transform, reusable pretransformed model weights, optional pack4 and cooperative matrices, and device/geometry-aware dispatch. Its implementation is designed for inference and should not be copied wholesale into general-purpose PyTorch training. License is BSD-3-Clause (source files contain SPDX notice); copying code would require preservation of its terms and attribution.

**MNN, Alibaba, Apache-2.0**:
https://github.com/alibaba/MNN

Provides other Winograd design precedents and hardware-aware inference kernels, but check the exact backend/file implementation and copyright notices before porting.

**Da Yan, Wei Wang, Xiaowen Chu, _Optimizing Batched Winograd Convolution on GPUs_, PPoPP 2020**:
https://cse.hkust.edu.hk/~weiwa/papers/yan-ppopp20.pdf

Highlights coalesced data layout, shared-memory bank conflicts, larger blocking, pipelining and register pressure. The study reports up to 2.65x vs an older cuDNN release on specific NVIDIA devices; do not interpret as a guaranteed improvement on today's RTX/V100/AMD or as permission to copy proprietary SASS fragments.

**Vulkan cooperative matrices, Khronos**:
https://docs.vulkan.org/refpages/latest/refpages/source/VK_KHR_cooperative_matrix.html

KHR-cooperative-matrix must be enabled and its supported matrix sizes/subgroup constraints queried for each device. Use as an optional accelerated algorithm alongside portable FP32, not a mandatory runtime dependency.

## Next optimization order

1. Keep the validated single-submission chain and bounded grouped dispatch.
2. Profile GPU shader time per transform/GEMM/inverse stage, memory coalescing and occupancy independently, rather than assuming arithmetic is now dominant.
3. Prototype larger GEMM microtiles, register blocking and a channel-packed Winograd layout; compare first on both GPUs and Radeon when available.
4. Explore F(4,3) with explicit FP32 numerical error and 64-MiB memory budget checks, benchmark against both F(2,3) and implicit GEMM.
5. Investigate inference-safe transformed-weight caching only with explicit tensor version/invalidation.
6. Evaluate cooperative matrix support by capability, with a portable default/fallback; avoid opaque GPU flags or network-specific exceptions.

The unforced official TorchBench U-Net eval model (RTX 2080 Ti, batch 1, 2 warmups, 4 timed calls) still passes independent CPU output parity with zero numerical CPU fallback: 322.299 ms median (320.830–323.336 ms). U-Net's ungrouped convolutions do **not** trigger the new grouped Winograd default; the stable result is a no-regression check, not an attributed model-level Winograd speedup.
