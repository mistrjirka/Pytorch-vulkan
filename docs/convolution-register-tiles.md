# Convolution register tile measurements

A reusable four-output-per-thread candidate reduces isolated large 3×3 convolution latency by 32–36% on RTX 2080 Ti. It regresses several other geometries, so the default heuristic is unchanged. The candidate is available through explicit tile selection or opt-in per-phase autotuning. Whole-model effects are smaller and variable.

## Implementation and generality

The new `register` candidate computes a 32×32 output tile with a 16×16 workgroup (256 threads), a reduction tile of 16, four FP32 accumulators per thread and 4,096 bytes of shared memory. Each thread shares loaded values across two channel rows and two positions. Forward, input-gradient and weight-gradient shaders share the same accumulation and store helpers.

Convolution geometry remains runtime data: batches, channels, groups, rectangular filters, stride, dilation, padding and incomplete tiles. There are no model names or fixed network dimensions in dispatch. A 10×10 and an 11×11 convolution can use the same compiled candidate; their exact geometry still determines eligibility, work count and the tuning-cache key.

Capability checks use actual workgroup dimensions, independently of the output tile dimensions. Existing workspace checks, split weight-gradient reduction, gradient masks, lifetime handling and fallback candidates remain in place. The weight-gradient finalizer is unchanged; no floating-point atomics, tensor cores or reduced precision were introduced. Layout packing continues through the existing path.

`TORCH_VULKAN_CONV_AUTOTUNE=1` lets each phase select from compatible candidates. `TORCH_VULKAN_CONV_TILE=register` explicitly selects this tile when a tiled algorithm is used; it does not replace direct/depthwise/reduction choices. Leave it unset for ordinary tuning. Forcing the register tile globally is not recommended.

New `candidates_timed_<phase>_<candidate>` counters count shortlist entries per tuning event, not individual timing samples. Whole-model benchmark output now records dispatch-counter deltas outside timed intervals; these include warmup.

## Matched isolated measurements

All comparisons below use the same RTX 2080 Ti, FP32, seed 2026 and eight warmup iterations. Input checksums match across CUDA and every applicable Vulkan run. Vulkan uses Torch 2.14.1+cpu; CUDA uses Torch 2.14.1+cu130 with TF32 disabled and cuDNN benchmarking disabled.

The wall comparison uses current host code in both variants. A loads the original shader binaries from commit `3410a3fc34e6a4542bb1712cbc84bae40eecb3f9` with the `tall` tile preference. B uses current binaries with `register`. The order is A1, B1, B2, A2, each in a fresh process with 15 samples. Tables pool 30 samples per variant. Resident buffers and deferred submissions are enabled. Wall time includes allocation, dispatch and synchronization but excludes explicit result download and reference checks.

These are comparisons against the previous tall candidate, not the unchanged base default or a guarantee that tuning always picks the fastest candidate.

### Large 3×3 case

Input [2,32,64,64], 64 output channels, stride 1, padding 1, groups 1. Times are milliseconds.

| Phase | Previous tall wall | Register wall | Reduction | CUDA wall |
|---|---:|---:|---:|---:|
| forward | 0.3390 | 0.2182 | 35.6% | 0.0745 |
| dinput | 0.5380 | 0.3435 | 36.1% | 0.0729 |
| dweight | 0.4294 | 0.2938 | 31.6% | 0.1392 |

### Device execution and remaining CUDA gap

Vulkan timestamps come from separate eager, nonresident runs with `TORCH_VULKAN_PROFILE_TIMESTAMPS=1`; only the shader interval is tabulated. CUDA values are the sum of actual GPU kernel durations from five profiled iterations, including the forward bias kernel. CUDA sums exclude launch gaps and memory operations; Vulkan split-weight intervals include the finalizer and intervening barrier. These are device execution comparisons, not end-to-end overhead measurements.

| Large 3×3 phase | Previous tall GPU | Refactored tall GPU | Register GPU | CUDA kernels | Register / CUDA |
|---|---:|---:|---:|---:|---:|
| forward | 0.2809 | 0.2789 | 0.1582 | 0.0400 | 3.96× |
| dinput | 0.4636 | 0.4617 | 0.2673 | 0.0287 | 9.33× |
| dweight | 0.3342 | 0.3511 | 0.2109 | 0.0928 | 2.27× |

CUDA uses Winograd kernels for this case; weight gradients also include Winograd transforms and `volta_sgemm_64x64_nt`. The input-gradient device gap narrows from about 16.2× to 9.3×. This is still a large compute gap, despite the register tile improvement. The traced algorithm difference suggests that adding broader algorithm families, with general eligibility and fallback rules, could offer larger gains than further tuning one tile; this has not yet been measured in Vulkan.

The 10×10 and 11×11 CUDA forward traces both use an FFT32 family. Their backward choices differ by phase and geometry: the 10×10 input gradient uses FFT, while the 11×11 trace uses `cudnn::detail::dgrad_engine`. Both weight-gradient traces use `wgrad_alg0_engine`. These observations describe these exact runs; they do not imply CUDA always chooses the same family for nearby filters.

### Wins and regressions across the suite

Wall medians in milliseconds; speedup is previous tall divided by register. Tile preferences affect only paths that choose a tiled algorithm. In particular, the depthwise/direct/reduction paths may be identical, so their small timing differences cannot be attributed to the new tile.

| Case | Phase | Previous tall | Register | Speedup |
|---|---|---:|---:|---:|
| pointwise | forward | 0.0755 | 0.0793 | 0.95× |
| pointwise | dinput | 0.0980 | 0.1001 | 0.98× |
| pointwise | dweight | 0.1334 | 0.1540 | 0.87× |
| rect3x5 | forward | 0.0902 | 0.1170 | 0.77× |
| rect3x5 | dinput | 0.1274 | 0.1565 | 0.81× |
| rect3x5 | dweight | 0.1170 | 0.1468 | 0.80× |
| filter10x10 | forward | 0.2767 | 0.3010 | 0.92× |
| filter10x10 | dinput | 0.5129 | 0.7253 | 0.71× |
| filter10x10 | dweight | 0.2912 | 0.2217 | 1.31× |
| filter11x11 | forward | 0.3184 | 0.3560 | 0.89× |
| filter11x11 | dinput | 0.6022 | 0.8673 | 0.69× |
| filter11x11 | dweight | 0.3226 | 0.2765 | 1.17× |
| grouped | forward | 0.0857 | 0.0930 | 0.92× |
| grouped | dinput | 0.1312 | 0.1446 | 0.91× |
| grouped | dweight | 0.1566 | 0.1627 | 0.96× |
| depthwise | forward | 0.0665 | 0.0709 | 0.94× |
| depthwise | dinput | 0.0899 | 0.0934 | 0.96× |
| depthwise | dweight | 0.0881 | 0.0925 | 0.95× |
| conv3x3_large | forward | 0.3390 | 0.2182 | 1.55× |
| conv3x3_large | dinput | 0.5380 | 0.3435 | 1.57× |
| conv3x3_large | dweight | 0.4294 | 0.2938 | 1.46× |
| conv3x3_small | forward | 0.1317 | 0.1468 | 0.90× |
| conv3x3_small | dinput | 0.1819 | 0.2089 | 0.87× |
| conv3x3_small | dweight | 0.1462 | 0.1502 | 0.97× |

The large-filter result is phase dependent: register blocking improves weight gradients while making input gradients substantially slower. This supports independent per-phase selection. The measurements do not isolate occupancy, integer address calculation or memory bandwidth as the cause of the regressions.

## Whole-model results

Unmodified medium models from zoo commit `c7c063101ec5209caf8022f5b4ea6948d1a8d6ed`, U-Net input [1,3,64,64] and StyleNet input [1,3,36,36], eight warmup and 15 timed iterations. Vulkan uses resident/deferred execution and convolution autotuning; the before run has three tiles, the after runs have four. Training uses fused Vulkan AdamW versus standard CUDA AdamW with foreach disabled. These are complete backend configurations, not convolution-only measurements.

Only one before run and two after runs were collected; this is not an ABBA whole-model comparison. Medians in milliseconds:

| Model / mode | Before | After run 1 | After run 2 | CUDA |
|---|---:|---:|---:|---:|
| U-Net inference | 2.157 | 2.014 | 2.016 | 1.093 |
| U-Net training | 15.741 | 14.456 | 15.233 | 8.459 |
| StyleNet inference | 40.298 | 39.293 | 39.651 | 1.531 |
| StyleNet training | 150.731 | 152.566 | 137.265 | 12.198 |

U-Net improved modestly in these runs. StyleNet training varies substantially, including one slower after median, so these results do not establish a reliable StyleNet training speedup. The tuner measures isolated synchronized calls; that ranking need not minimize a deferred whole training step.

The repeated U-Net training run selected register forward/input-gradient/weight-gradient 92/92/161 times; StyleNet selected 46/92/322. Counts include warmup and show actual phase-specific use, not the amount of time saved. Initial losses agree with CUDA to approximately 6e-8; this does not validate long-run training trajectories.

## Validation

The complete suite passes on the tested devices:

| Configuration | Result |
|---|---|
| rtx | 476 passed, 3 warnings in 12.88s |
| v100 | 476 passed, 3 warnings in 12.28s |
| lvp | 475 passed, 1 skipped, 3 warnings in 13.86s |
| deferred | 476 passed, 3 warnings in 12.81s |

The dispatch test file now contains 82 cases, including 21 added register-tile cases: grouped and rectangular convolutions; 9/10/11/12 filters; stride, dilation and noncontiguous layouts; forward and all gradients; channel/position tails; 255/256/257 split boundaries; masks; tuning-cache reuse; and workspace-limited eligibility. Adjacent-filter pipeline reuse accepts either a compiled-pipeline hit or reuse of an already-bound algorithm, with no new pipeline miss.

All 17 shader binaries handled by `scripts/compile_conv_tiles.py` compile and pass `spirv-val`. Three pre-existing MSE resize warnings remain; Lavapipe skips the hardware-acceleration assertion. AMD and Intel hardware are untested. V100 correctness was checked through Vulkan; CUDA 13 cannot execute on its SM70 architecture.

The final validation host-source SHA256 and shader hashes match the working files used for this report. Final benchmark shader hashes also match. No core source or shader changes were made after validation.

## Reproduction and raw data

Raw results are in [benchmarks/results/conv-register](../benchmarks/results/conv-register/). `tall_a1/a2` and `register_b1/b2` are the controlled wall runs; `*_gpu_final` and `tall_gpu_before` provide device intervals; `cuda.json` contains exact traced kernel names. Files named `initial` preserve exploratory runs and are not pooled into the controlled wall comparison. Whole-model before/initial/repeat files are identified in the tables above.

Build the extension and compile the shaders using the project instructions. To reconstruct the old shader directory:

```bash
mkdir -p /tmp/conv-register-baseline
git archive 3410a3fc34e6a4542bb1712cbc84bae40eecb3f9 csrc/shaders | tar -x -C /tmp/conv-register-baseline
```

Run the following in A/B/B/A order, using a distinct output filename per process. A uses `--shader-dir /tmp/conv-register-baseline/csrc/shaders` and tile `tall`; B omits the override and sets tile `register`.

```bash
TORCH_VULKAN_GPU_RESIDENT=1 TORCH_VULKAN_DEFER_SUBMISSIONS=1 TORCH_VULKAN_CONV_TILE=tall \
  .venv/bin/python benchmarks/probe_conv_dispatch.py --backend vulkan \
  --shader-dir /tmp/conv-register-baseline/csrc/shaders --out /tmp/tall-a1.json

TORCH_VULKAN_CONV_TILE=register TORCH_VULKAN_PROFILE_TIMESTAMPS=1 \
  .venv/bin/python benchmarks/probe_conv_dispatch.py --backend vulkan \
  --cases conv3x3_large,filter10x10,filter11x11 --out /tmp/register-gpu.json

CUDA_VISIBLE_DEVICES=GPU-30acbf2b-a41a-8b88-e298-fe66c4138b29 \
  .venv-cuda/bin/python benchmarks/probe_conv_dispatch.py --backend cuda \
  --profile-cuda-kernels --out /tmp/cuda.json
```

Unset resident/deferred environment flags for the GPU timestamp run, and clear algorithm/autotune overrides for the forced-tile wall comparison. Select the same physical GPU for CUDA and Vulkan; the UUID above identifies the measured RTX.

The new `--shader-dir` option works in both convolution and whole-model benchmark CLIs, validates its path and records the active SPIR-V hashes. Always use a fresh process when switching directories: changing the shader directory does not invalidate existing shader/pipeline caches.

For whole-model reproduction use `benchmarks/compare_cuda_vulkan.py --model unet` (or `style_net`), `--scale medium --warmup 8 --repeats 15 --zoo /path/to/zoo`. Use `--backend vulkan --optimizer vulkan_fused` with resident/deferred/autotune enabled, or `--backend cuda --optimizer standard`. The original whole-model baseline used both host code and shaders from commit 3410a3f; a shader-only override reconstructs the isolated A comparison, not that original full-model configuration.
