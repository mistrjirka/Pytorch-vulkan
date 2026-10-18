# General convolution backward and layout support — 18 October 2026

## What changed

- Shared immutable Kompute pipelines and pooled, separately owned descriptor sets; bounded pipeline and bound-algorithm caches. This includes the previously uncommitted runtime work.
- Independent forward and backward selection. Input gradients use tiled implicit GEMM when suitable; weight gradients use parallel reductions or tiled split reductions; bias gradients use workgroup reductions. Scalar reference paths remain selectable.
- Filters, channels, groups, stride, padding and dilation stay runtime parameters. Ragged channel/spatial/reduction tiles are masked.
- FP32 channels-last, transposed, sliced, offset and broadcast views are packed into logical NCHW order by a Vulkan shader. Input, weight, bias and output-gradient layouts are covered. Packing adds work; outputs are contiguous, and this does not imply arbitrary-layout support for every other operator.
- Split weight-gradient intermediates remain on GPU between kernels. A dedicated dependent-pair executor inserts a compute write-to-read barrier and uses one submission/fence. Deferred residency also preserves the dependency. Workspace is capped at 64 MiB and the physical storage-buffer limit; selection falls back when unsuitable.
- Invalid effective kernels larger than the padded input now fail before signed integer division can accidentally accept an invalid shape.

## Matched GPU computation

RTX 2080 Ti, FP32, TF32 disabled. Same convolution geometries and seeded tensors. Vulkan entries are average timestamp intervals over 9 warmed executions; CUDA entries are summed actual GPU-kernel durations per execution over 5 profiler iterations. CUDA event intervals, host medians and ranges are separately saved. The Vulkan split interval includes its internal barrier; these figures are not whole training-step latency.

| Case | Phase | Direct Vulkan ms | Selected Vulkan ms | Speedup | CUDA kernels ms |
|---|---|---:|---:|---:|---:|
| standard5 | dinput | 0.3802 | 0.1225 | 3.10× | 0.0220 |
| standard5 | dweight | 0.1064 | 0.0478 | 2.23× | 0.0316 |
| standard5 | dbias | 0.0646 | 0.0092 | 7.02× | 0.0048 |
| standard7 | dinput | 0.3534 | 0.1130 | 3.13× | not profiled |
| standard7 | dweight | 0.0752 | 0.0370 | 2.03× | not profiled |
| standard7 | dbias | 0.0527 | 0.0096 | 5.48× | not profiled |
| large3 | dinput | 7.5227 | 1.5901 | 4.73× | 0.0866 |
| large3 | dweight | 4.3168 | 0.9129 | 4.73× | 0.1643 |
| large3 | dbias | 0.7107 | 0.0251 | 28.35× | 0.0087 |
| large5 | dinput | 1.8503 | 0.6085 | 3.04× | 0.5144 |
| large5 | dweight | 0.8835 | 0.3063 | 2.88× | 0.1265 |
| large5 | dbias | 0.2016 | 0.0097 | 20.74× | 0.0102 |
| large1 | dinput | 1.8110 | 0.3760 | 4.82× | 0.0350 |
| large1 | dweight | 2.3473 | 0.2851 | 8.23× | 0.1345 |
| large1 | dbias | 1.4280 | 0.0557 | 25.66× | 0.0189 |

`large3`: input [2,64,64,64], weight [128,64,3,3]; `large5`: [1,32,64,64] × [64,32,5,5]; `large1`: [4,64,64,64] × [128,64,1,1]. Complete geometry for every case is in the JSON.

**What CUDA does differently:** the captured large 3×3 forward and input-gradient paths use cuDNN Winograd kernels. Weight gradients use Winograd transforms plus `volta_sgemm_64x64_nt`. The larger 5×5 paths select other convolution engines. Vulkan currently has generic tiled FP32 kernels, no Winograd or cooperative-matrix path. The remaining shader gap therefore calls for additional general algorithms and capability/shape selection, not a kernel hardcoded to one network.

**An experiment rejected:** one workgroup per weight helped depthwise/pointwise cases but slowed some large filters. Tiled split reduction fixed compute reuse, yet initially lost wall time by downloading and reuploading its scratch tensor and waiting twice. The final dependent-pair executor removes that scratch roundtrip. Experimental files (`backward_candidates_*`, `backward_tiled_*`) preserve this evidence; use `backward_final_*` for the final implementation.

## End-to-end evidence and remaining bottleneck

Full model training includes forward, loss, backward and AdamW. Vulkan uses its opt-in fused AdamW, residency and deferred submissions; CUDA uses native AdamW. Both Vulkan variants use the same forward selector; only backward selection changes. Four warmups, nine measured iterations, identical model definitions and synthetic input shapes. These are single-run observations with substantial training variance, not stable speedup claims.

| Model / input | Direct backward, ms | Selected backward, ms (range) | CUDA, ms |
|---|---:|---:|---:|
| unet [1, 3, 64, 64] | 280.7 | 271.4 (193.0–345.3) | 8.88 |
| style_net [1, 3, 36, 36] | 521.7 | 414.9 (340.9–448.5) | 10.83 |
| mobilenet_v3 [2, 3, 96, 96] | 489.5 | 479.3 (469.3–499.4) | 29.83 |

U-Net and MobileNet changes are smaller than run variability. StyleNet improved in this run, but needs repeated controlled runs before attributing an exact model-level speedup. These tables do not establish end-to-end model gradient parity; numerical convolution/gradient parity is tested separately.

A matched U-Net training operator profile after eight warmup steps identifies **260 `aten::_to_copy` calls and about 151 ms of self CPU time per profiled Vulkan step**. This includes transfer/synchronization/dispatch work inside the copy operator; it is not 151 ms of PCIe DMA. CPU fallback paths include BatchNorm backward, nearest-neighbor upsampling backward, and max-pooling forward/backward. In three measured steps there were **zero new pipeline compilations**, despite 83 bound-algorithm misses and 63 evictions. Thus pipeline compilation is not the measured dominant remaining problem.

**Next priority:** general native BatchNorm, pooling and upsampling forward/backward to remove these copy/fallback boundaries, while preserving their complete PyTorch semantics. Then investigate Winograd and more efficient tiled/cooperative-matrix convolution variants using the same matched profiling method. FP16/BF16, transposed convolution and fused LSTM are still unsupported; AMD/Intel remain untested.

## Correctness

- Full suite: RTX 2080 Ti **117 passed**; V100 **117 passed**; Lavapipe **116 passed, one hardware-only skip**. Three pre-existing output-resize warnings remain.
- Convolution tests: **35 passed** with residency, and **35 passed** with residency plus deferral. Includes forced optimized/direct paths, rectangular/1×1/5×5/7×7 kernels, groups, channel multipliers, asymmetric stride/dilation, all backward masks, and strided layouts.
- Benchmark gradients also compare with FP64 CPU references converted to FP32. Direct reduction and optimized tree/split reduction need not be bitwise equal. The benchmark uses atol=0.005, rtol=0.0005 for the large unnormalized sums; smaller regression tests use tighter tolerances.
- Both physical GPUs executed all 11 benchmark geometries. CUDA comparison uses the RTX GPU UUID explicitly because CUDA and Vulkan enumeration differ and this installed CUDA 13 wheel does not support V100.

## Reproduction

```bash
TORCH_VULKAN_PROFILE_TIMESTAMPS=1 .venv/bin/python benchmarks/probe_conv_backward.py \
  --backend vulkan --algorithms direct,auto --out /tmp/conv-vulkan.json
TORCH_VULKAN_GPU_RESIDENT=1 .venv/bin/python benchmarks/probe_conv_backward.py \
  --backend vulkan --algorithms direct,auto --out /tmp/conv-resident.json
CUDA_VISIBLE_DEVICES=<RTX_GPU_UUID> .venv-cuda/bin/python benchmarks/probe_conv_backward.py \
  --backend cuda --out /tmp/conv-cuda.json
CUDA_VISIBLE_DEVICES=<RTX_GPU_UUID> .venv-cuda/bin/python benchmarks/profile_conv_cuda_kernels.py \
  --out /tmp/cuda-kernels.json
TORCH_VULKAN_GPU_RESIDENT=1 TORCH_VULKAN_DEFER_SUBMISSIONS=1 \
  .venv/bin/python benchmarks/profile_model_operators.py --backend vulkan --model unet \
  --zoo ../pytorch-XDNA2-vulkan-zoo-reference --out /tmp/unet-profile.json
```

`TORCH_VULKAN_CONV_BACKWARD_ALGO=auto|direct|optimized` is an A/B diagnostic control. `optimized` forces the tiled input-gradient path and allows reductions on small tensors, subject to device checks. It does not promise the fastest algorithm for every shape. Residency/deferral remain experimental and opt-in.

Raw measurements: [benchmarks/results/convolution_algorithms](../benchmarks/results/convolution_algorithms/).

## Review note

An additional read-only OpenCode/DeepSeek review was attempted on an isolated source copy. It read the files but reached its 180-second timeout before producing findings, so it is not counted as an independent completed review. Validation claims above come from executed numerical tests and profiles.
