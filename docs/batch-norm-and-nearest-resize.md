# Native BatchNorm and nearest resize — 18 October 2026

U-Net's previous Vulkan training profile spent most of its CPU time in transfers around unsupported operators. This change implements FP32 BatchNorm and legacy nearest-neighbor 2D resize on Vulkan, including backward. Their normal execution no longer calls the CPU fallback.

## Semantics and portability

- BatchNorm accepts nonempty rank 2–8 tensors, interpreting dimensions as N, C, and flattened spatial dimensions. BatchNorm1d, BatchNorm2d and BatchNorm3d are tested. Training and evaluation, optional affine parameters, optional running statistics in training, momentum, biased normalization variance, unbiased running variance and all backward output masks are supported. Evaluation requires running mean and variance. Evaluation returns empty saved-stat tensors, matching the CPU implementation; CUDA's saved-tensor shape differs in evaluation.
- One 128-thread workgroup reduces each channel and writes its output. Mean calculation subtracts a reference value before summing; variance uses centered differences, avoiding `E[x*x] - E[x]^2` cancellation. Running-stat writes participate in residency bookkeeping and deferred ordering.
- Nearest resize accepts rank-4 FP32 input, explicit output sizes or scale factors, enlargement, reduction, rectangular sizes and empty batches. Its backward gathers gradients deterministically without floating-point atomics. Binary search matches the forward FP32 coordinate mapping for fractional scales; identity and 2x mapping have simple shortcuts. This implements `nearest`, not `nearest-exact`, bilinear or 1D/3D resize.
- Strided, channels-last, transposed, sliced, offset and broadcast inputs are packed on the GPU when necessary. A shared packing shader handles up to eight dimensions. BatchNorm channel vectors can be strided and offset. Results use contiguous storage.
- Workgroup size/count, shared memory and storage descriptor/range limits are checked. No shape or GPU name selects a special model-specific path. FP16/BF16 remain unsupported. AMD and Intel hardware remain unverified.

`TORCH_VULKAN_DISABLE_BN_RESIZE=1`, set before importing the extension, omits these four new registrations and restores the existing generic CPU fallback. This diagnostic switch provides an A/B comparison without changing model code, convolution kernels, optimizer, or the runtime.

## Matched U-Net measurements

All rows use the same RTX 2080 Ti, FP32 model (472,833 parameters), input `[1,3,64,64]`, seeded initialization and data, eight warmup iterations and fifteen measured iterations. CUDA TF32 is disabled. Vulkan uses `TORCH_VULKAN_GPU_RESIDENT=1`, `TORCH_VULKAN_DEFER_SUBMISSIONS=1` and fused AdamW; CUDA uses ordinary AdamW with `foreach=False`. Both optimizers use the same hyperparameters. Training includes forward, loss, backward, optimizer and final synchronization. Inference runs first in each process. Setup and initial placement are excluded.

| Path | Inference median (range), ms | Training median (range), ms |
|---|---:|---:|
| Original fallback, diagnostic switch | 78.736 (77.033–79.365) | 346.112 (299.574–351.282) |
| Native BatchNorm + resize, run 1 | 39.130 (37.559–39.370) | 123.423 (81.606–156.188) |
| Native BatchNorm + resize, run 2 | 40.061 (39.070–41.154) | 83.466 (80.849–108.553) |
| CUDA | 1.063 (1.058–1.074) | 9.521 (8.725–9.696) |

The measured inference improvement is about 2x. The two training medians give 2.8–4.1x against the matched fallback run; the large Vulkan variation means this is an observed range, not a stable speed guarantee. CUDA remains substantially faster. The first training loss was identical to the displayed FP32 precision (0.70502573), and first-output statistics agreed closely; this is a consistency check, not full convergence validation.

The separate operator-profile workload uses seed 2026 and eight warmups followed by three profiled steps. Its copy count fell from **260 to 80** with BatchNorm alone, then to **72** with nearest resize. Copy self CPU time fell from approximately **151 to 54 ms per profiled step**. That time includes dispatch/waits inside copies and must not be reported as pure DMA time. Max-pool forward/backward and remaining add/loss fallbacks are the next substantial targets. Profiled and unprofiled times are not interchangeable.

## Where CUDA is still faster

`benchmarks/probe_batch_norm.py` generates identical input, affine and output-gradient values on the CPU before placing them on either backend. It checks forward and gradients against CPU, then measures five warmups and fifteen synchronized calls. CPU profiler events, actual CUDA kernel names and optional Vulkan timestamps are saved.

Training BatchNorm, `[1,32,64,64]`:

| Measurement | Previous Vulkan fallback | Native Vulkan | CUDA |
|---|---:|---:|---:|
| Forward synchronized wall time, ms | 2.8584 | 0.5430 | 0.0600 |
| Backward synchronized wall time, ms | 5.4496 | 0.5421 | 0.0451 |
| Forward GPU computation, ms | — | 0.02119 | 0.01007 |
| Backward GPU computation, ms | — | 0.01775 | 0.00526 |

CUDA forward launches statistics, running-stat update and transform kernels; its backward uses one native BatchNorm kernel. Vulkan currently uses one channel workgroup for each forward/backward operation. Its GPU computation remains about 2x/3.4x slower in this case, but the much larger synchronized wall-time gap is outside the shader. A separate resident timing probe measured roughly 0.04 ms recording and 0.47–0.51 ms synchronization per warmed call, with bound-algorithm cache hits.

Vulkan GPU timestamps above come from the eager transfer path, which separately records upload, shader and download; they are not timestamps of the deferred model path. For this shape the eager forward measured approximately 0.0455 ms upload and 0.1663 ms download. Resident wall times avoid those transfers. Do not add numbers from these distinct execution modes to derive model latency.

On `[2,64,64,64]`, synchronized forward/backward improved from **11.517/21.884 ms** to **0.642/0.719 ms**. Small isolated calls can regress: `[32,128]` forward rose from **0.163 to 0.541 ms** because CPU fallback is cheap at that size and the Vulkan submission/wait dominates. Native execution is most useful when it keeps an existing GPU workload resident and batchable; this change does not claim every isolated size is faster.

## Validation

The full suite passes **214 tests on RTX 2080 Ti**, **214 on Tesla V100**, and **213 on Lavapipe with one hardware-only skip**. Three existing MSE fallback output-resize warnings remain. The 97 new focused tests also pass with resident deferred execution and host-visible deferred execution.

Coverage includes forward values, input/affine gradients, all eight gradient masks, training/evaluation, absent affine/running statistics, running-stat mutations, dimensions 2–6, layout/offset/broadcast views, large-offset nearly constant data, zero variance/epsilon, the one-sample running-variance NaN, resize fractional scales/downsampling, empty batches and chains of deferred autograd operations. Profiler tests assert that native calls do not invoke `_to_copy`.

Two reference details are explicit in the tests. This installed CPU optimized BatchNorm backward misreads some strided channel vectors, so its reference uses identical values in contiguous vectors. For a mathematically zero scale gradient after a 250-element reduction, Lavapipe differed from CPU by 5.54e-5; the chain test allows 1e-4 absolute FP32 reduction error. Other numerical checks retain their tighter tolerances.

Raw results are in `benchmarks/results/batch_norm/`. The `final_*` files contain matched BatchNorm data and parity checks; the `unet_*` files contain model timings and staged operator profiles. The original profile is in `benchmarks/results/convolution_algorithms/unet_vulkan_operator_profile.json`.

## Reproduction

```bash
# Native and original-fallback U-Net, separate processes:
TORCH_VULKAN_GPU_RESIDENT=1 TORCH_VULKAN_DEFER_SUBMISSIONS=1 \
  .venv/bin/python benchmarks/compare_cuda_vulkan.py --backend vulkan \
  --model unet --scale medium --mode both --optimizer vulkan_fused \
  --warmup 8 --repeats 15 --zoo ../pytorch-XDNA2-vulkan-zoo-reference
# Repeat with TORCH_VULKAN_DISABLE_BN_RESIZE=1 for the fallback baseline.

TORCH_VULKAN_GPU_RESIDENT=1 TORCH_VULKAN_DEFER_SUBMISSIONS=1 \
  .venv/bin/python benchmarks/probe_batch_norm.py --backend vulkan --out /tmp/bn-resident.json
TORCH_VULKAN_PROFILE_TIMESTAMPS=1 \
  .venv/bin/python benchmarks/probe_batch_norm.py --backend vulkan --out /tmp/bn-timestamps.json
CUDA_VISIBLE_DEVICES=GPU-30acbf2b-a41a-8b88-e298-fe66c4138b29 \
  .venv-cuda/bin/python benchmarks/probe_batch_norm.py --backend cuda --out /tmp/bn-cuda.json
```
