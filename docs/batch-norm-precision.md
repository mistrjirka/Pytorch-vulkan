# BatchNorm precision and ResNet diagnosis

This follow-up to `bc4922d` fixes FP32 BatchNorm arithmetic for the supported nonempty rank-2–8 inputs, including packed strided layouts. Selection depends only on the reduction size and training/evaluation semantics. No model names, layer IDs or tensor addresses participate in dispatch.

## Changes

- Forward retains the shifted mean as two terms through variance calculation and normalization. Rounding `origin + mean_shift` first can lose the entire difference between nearly equal FP32 samples. The public saved mean remains an ordinary FP32 channel vector.
- General training backward recovers the saved mean's residual from the input and corrects both centered input and covariance. Evaluation continues to use the supplied running statistics unchanged.
- For exactly two values per channel, backward uses the equivalent derivative `dx0 = (g0-g1)/2 * gamma * eps * invstd³`, `dx1 = -dx0`. Scale gradient uses the two differences directly. This avoids subtracting nearly equal projection terms and avoids the shared-memory reduction. The same path applies to batch-two, spatial-two, and higher-rank configurations.
- Overflow guards avoid introducing `0 * infinity` during a zero mean correction, and halve opposite large gradients before subtracting. Existing single-value, zero-epsilon, masks, affine and running-variance semantics remain covered.
- The shared native-operator helper now caches immutable physical-device limits, as convolution already did. This removes repeated driver queries for BatchNorm, packing, pooling, concatenation and resize. No isolated speedup is attributed to that cache in these measurements.

The backward shader requires 1,536 bytes of shared memory, checked against the device's actual limit. Buffer, descriptor and deferred execution lifetimes are unchanged.

## Accuracy against FP64

Same rounded FP32 input/weight/gradient values are promoted to FP64 for the reference. The table is maximum absolute error with 512 channels, input offset 10,000 and noise scale 0.1. CPU FP32 and CUDA FP32 can share cancellation errors, so neither is used as the correctness oracle here.

| Values per channel | Result | Vulkan before | Vulkan after | CUDA FP32 |
|---:|---|---:|---:|---:|
| 2 | output | 0.201977 | 1.90662e-07 | 0.198144 |
| 2 | dinput | 28.6072 | 2.08534e-05 | 28.9456 |
| 2 | dweight | 0.313142 | 4.0204e-07 | 0.318718 |
| 3 | output | 0.0574438 | 3.22452e-07 | 0.211665 |
| 3 | dinput | 3.68467 | 1.92915e-05 | 24.928 |
| 3 | dweight | 0.0916934 | 6.16932e-07 | 0.350908 |
| 128 | output | 0.00731326 | 6.06346e-07 | 0.0206991 |
| 128 | dinput | 0.043177 | 7.17365e-06 | 0.0711837 |
| 128 | dweight | 0.149105 | 6.1077e-06 | 0.321383 |

These are particular finite numerical probes, not an all-input error bound. General FP32 reductions still have finite-range and rounding limits.

## Matched performance

RTX 2080 Ti, FP32, TF32 disabled, eight warmup calls and 30 synchronized samples per phase. The wall table uses `[8,64,32,32]`; phases exclude explicit application `.cpu()` reads. Eager Vulkan's internal transfers are included. Resident mode uses the existing opt-in `TORCH_VULKAN_GPU_RESIDENT=1` and `TORCH_VULKAN_DEFER_SUBMISSIONS=1`; it is not a new default or a speedup introduced by this patch.

| Phase | Vulkan before, eager ms | Vulkan after, eager ms | Vulkan after, resident ms | CUDA ms |
|---|---:|---:|---:|---:|
| forward | 0.9535 | 0.9204 | 0.0955 | 0.0563 |
| backward | 1.1776 | 1.1261 | 0.1085 | 0.0515 |

Separate GPU measurements compare Vulkan timestamp intervals with the sum of actual CUDA kernel durations from five profiler iterations. These exclude host dispatch and explicit transfers; they are not wall-time measurements.

| Shape | Phase | Old Vulkan shader µs | New Vulkan shader µs | CUDA kernels µs |
|---|---|---:|---:|---:|
| [2,512] | forward | 10.27 | 10.33 | 6.34 |
| [2,512] | backward | 8.42 | 6.56 | 4.27 |
| [8,64,32,32] | forward | 32.50 | 34.72 | 15.06 |
| [8,64,32,32] | backward | 42.09 | 42.95 | 19.00 |

The old-shader profile uses the original two SPIR-V binaries with the current host code, isolating shader changes. The original wall baseline uses the original host build. Run-to-run wall variation remains, so small wall differences are not claimed as robust speedups.

For the spatial case, the new Vulkan forward interval spends about 0.82 ms uploading/downloading versus 0.034 ms in the shader; backward spends about 0.98 ms transferring versus 0.043 ms in the shader. CUDA uses native statistics/transform and backward reduction/elementwise kernels, captured by name in `cuda.json`. Eager transfer and submission costs dominate this BatchNorm comparison. With residency enabled, a smaller kernel/dispatch gap remains. The separate convolution profiles still identify missing general Winograd/FFT algorithm families as a major compute gap; this patch adds neither family.

## Why the small ResNet diverges

The actual model-zoo torchvision ResNet18 is unchanged. Every backend receives the same hashed model state and inputs. CPU FP64 is run from the same rounded initial FP32 parameters. Leaf hooks identify where differences grow, and each Vulkan/CUDA BatchNorm also gets a local FP64 comparison using its own actual input.

At `[2,3,32,32]`, the final stage has only two values per channel and some inverse standard deviations near 316. Earlier FP32 differences are amplified repeatedly. CUDA and CPU FP32 also diverge from FP64. The Vulkan fix improves gradient accuracy against FP64, but agreement with CPU FP32 is not a sufficient correctness criterion for this case.

| Input | Backend | Whole-network gradient relative L2 vs FP64 | Max output error vs FP64 |
|---|---|---:|---:|
| 2×3×32×32 | CPU FP32 | 10.246% | 0.00970368 |
| 2×3×32×32 | Vulkan before | 5.2831% | 0.00344624 |
| 2×3×32×32 | Vulkan FP32 | 3.9864% | 0.0110703 |
| 2×3×32×32 | CUDA FP32 | 13.888% | 0.0133043 |
| 2×3×64×64 | CPU FP32 | 0.0011708% | 6.94177e-06 |
| 2×3×64×64 | Vulkan FP32 | 0.0010933% | 5.54768e-06 |
| 2×3×64×64 | CUDA FP32 | 0.00095263% | 3.88184e-06 |

The 32×32 output error versus FP64 increased from 0.00345 to 0.0111 even though the gradient error decreased. This sensitivity prevents claiming an across-the-board end-to-end accuracy improvement from the local fix.

Relative L2 is calculated over all parameter elements, not averaged over layer percentages. The 64×64 probe preserves the architecture and batch size while increasing the final spatial extent. It diagnoses conditioning; it does not replace the original 32×32 result or validate training convergence. These runs cover one forward/backward pass, not a training trajectory. Hook-induced synchronization also means they are not performance benchmarks.

## Verification and reproduction

33 additional tests cover FP64 parity for reductions of size 1/2/3/8/129, large offsets, nearly equal samples, cancelling and extreme gradients, positive/zero epsilon, training/evaluation, optional affine, output masks, running-stat mutation and non-contiguous layouts.

- RTX 2080 Ti: 455 passed, 3 warnings in 12.72s
- Tesla V100: 455 passed, 3 warnings in 11.40s
- Lavapipe: 454 passed, 1 skipped, 3 warnings in 10.91s
- RTX resident/deferred: 455 passed, 3 warnings in 12.94s

Both changed SPIR-V binaries pass `spirv-val`. The three existing MSE output-resize warnings remain. Lavapipe skips the hardware-acceleration assertion. AMD/Intel execution remains unverified. Independent read-only review checked the algebra, overflow guards, workgroup control flow, capability check and cached-limits lifetime.

```sh
.venv/bin/python benchmarks/probe_batch_norm_precision.py \
  --backend vulkan --out /tmp/bn-wall.json
TORCH_VULKAN_PROFILE_TIMESTAMPS=1 .venv/bin/python \
  benchmarks/probe_batch_norm_precision.py --backend vulkan --out /tmp/bn-gpu.json
CUDA_VISIBLE_DEVICES=GPU-30acbf2b-a41a-8b88-e298-fe66c4138b29 .venv-cuda/bin/python \
  benchmarks/probe_batch_norm_precision.py --backend cuda --out /tmp/bn-cuda.json
.venv/bin/python benchmarks/diagnose_resnet_precision.py \
  --zoo ../pytorch-XDNA2-vulkan-zoo-reference --backend vulkan \
  --size 32 --batch 2 --out /tmp/resnet32.json
```

Both probes accept `--shader-dir` for controlled old/new shader comparisons in separate processes. Raw input hashes, shader hashes, samples, CUDA kernel names and numerical results are under [benchmarks/results/batch-norm-precision](../benchmarks/results/batch-norm-precision/). `before.json` recorded an unused `TORCH_VULKAN_GPU_TIMING` environment variable; actual timestamp instrumentation is controlled by `TORCH_VULKAN_PROFILE_TIMESTAMPS` and was off in that wall baseline.
