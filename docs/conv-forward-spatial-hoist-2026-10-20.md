# General FP32 forward-convolution coordinate hoisting (20 October 2026)

## Goal

Improve the native, universal Vulkan convolution backend toward the overall **at most 1.5× CUDA** performance target without model-specific dispatch, numerical fallbacks or hidden performance flags.

## Problem and reusable change

The FP32 NCHW implicit-GEMM shader previously recalculated output-image quotient/remainder and input coordinates on every cooperative input fetch across the convolution reduction dimension. The tiled workgroup distributes columns such that **THREADS is divisible by TN**, and each invocation repeatedly loads the same spatial column. The new `register3_hoist` shader calculates per-lane spatial batch, row and column once outside the reduction loop and reuses them for 3×3 loads. All filter strides, padding and dilation remain runtime parameters. Padding guards and accumulation order are unchanged. There are no network-name conditions.

A separate shader/candidate permits controlled tests against the existing `register3` path. Automatic dispatch selects the new candidate **only for dense, sufficiently large FP32 3×3 forward convolutions** already selected for the previous register3 shader, subject to Vulkan capability checks. Backward input/weight keep their prior automatic implementations. Diagnostic `TORCH_VULKAN_CONV_TILE` overrides remain optional. Compilation uses `scripts/compile_conv_tiles.py` and SPIR-V validation.

## Matched TorchBench U-Net inference

Upstream TorchBench `torchbenchmark.models.pytorch_unet.Model(test="eval", device=..., batch_size=1)`, unmodified, FP32 output 640×959, RTX 2080 Ti, native Vulkan GPU residency+deferred execution, 2 warmups and 4 synchronized samples per process, ABBA alternation.

| Configuration | First process median | Second process median |
|---|---:|---:|
| Previous register3 | 349.976 ms | 356.772 ms |
| New register3_hoist | 322.563 ms | 322.920 ms |

The paired process medians give roughly **8–9.5% faster full model inference**. All four runs verified the original model's CPU reference (max absolute discrepancy 6.71e-8), with no recorded CPU fallback. This is substantially short of the 1.5× CUDA target. The benchmark includes TorchBench's normal postprocessing; it is not a standalone convolution measurement.

## Independent reusable operator checks

`benchmarks/probe_conv_dispatch.py`, original seeded FP32 double-precision CPU oracle, 4 warmup and 10 synchronized repeats, two independent processes per candidate:

| Convolution | Previous register3 | Hoisted version | Phase |
|---|---:|---:|---|
| conv3x3_large | 0.2349 / 0.2367 ms | 0.2216 / 0.1902 ms | Forward |
| conv3x3_small | 0.1790 / 0.1730 ms | 0.1314 / 0.1275 ms | Forward |

Changes in dInput/dWeight timings for the forced candidate include run-to-run noise; **this patch does not claim a new automatic backward algorithm**. The new forced-shader numerical tests exercise dense, grouped, strided and dilated 3×3 forward and all gradients, with strict CPU-fallback prohibition: 3 passed. The same code remains eligible only for 3×3 filters, with existing generic algorithms covering everything else.

Raw JSON results: `benchmarks/results/conv-forward-hoist-20261020/`.

## Remaining work

1. Test the selected default path on RTX 2080 Ti and V100, and validate unchanged full regression coverage.
2. Rerun default, unforced TorchBench and matched CUDA after rebuild.
3. Larger gains require genuinely different algorithm families, especially vendor-neutral Winograd/F(2,3) for dense 3×3 and proper optimized gradients. A few percent in address arithmetic will not by itself reach 1.5× CUDA. Evaluate device memory residency, occupancy, host scheduling and dispatch policies as separate measurable sources.

## Post-integration acceptance

- **RTX 2080 Ti**: full `pytest -q tests` suite **3,918 passed** (14 warnings). This includes 3 new shader/gradient parity tests; previous baseline had 3,915 passes.
- **Tesla V100**: strict `TORCH_VULKAN_FORBID_CPU_FALLBACK=1`, `pytest -q tests/test_conv_dispatch.py tests/test_conv_universality.py tests/test_convolution.py`: **146 passed**.
- **Production defaults verified**: `TORCH_VULKAN_CONV_TILE`, `TORCH_VULKAN_CONV_ALGO` and `TORCH_VULKAN_CONV_AUTOTUNE` unset, fresh process selects `selected_forward_tiled_register3_hoist: 1`, with both GPU residency and deferred submissions active.
- **Fresh matched official TorchBench U-Net inference**, same RTX 2080 Ti, FP32, 3 warmups and 6 timed calls per backend: **327.713 ms Vulkan, 38.814 ms CUDA (8.44x CUDA latency)**. CPU reference output parity passed for each. Vulkan records zero CPU fallback. This is **not** the 1.5x target.
- Exact JSON: `benchmarks/results/conv-forward-hoist-20261020/unet-auto.json` and `unet-cuda.json` (raw files are not necessarily tracked).

Only the eligible general forward 3×3 path changes automatically. The related SPIR-V shader variants are checked in and validated by `spirv-val`; no performance environment flag is required.
