# General command batching and faster unit-stride convolution gradients

Two changes build on `ef6aaf3364d876358676669d568a49d9d0c1e3ed`:

1. **Batch adjacent buffer-dependency commands.** Kompute previously called `vkCmdPipelineBarrier` once per buffer in `OpAlgoDispatch` and primary-buffer `OpMemoryBarrier`. The pinned-runtime patch now passes those buffer barriers together. Every buffer, byte range, access mask, stage mask and ignored queue-family setting is preserved. Nothing moves across a dispatch or copy; staging-buffer barriers and empty-list behavior are unchanged. This affects all operators using these runtime operations.
2. **Avoid unnecessary input-gradient coordinate arithmetic.** Tiled Conv2D input-gradient shaders skip stride divisibility tests and coordinate divisions when both strides are one. The guard uses operator geometry and applies to arbitrary filters, channels, groups, padding and dilation. Other strides retain divisibility checks and division. All four existing tile variants use the same updated shader body; no new model-specific kernel or dispatch rule was added.

The runtime change follows the [Vulkan buffer-barrier array API](https://docs.vulkan.org/refpages/latest/refpages/source/vkCmdPipelineBarrier.html). `scripts/patch_barrier_batch.py` applies only to bundled Kompute v0.9.0, preserves its Apache-2.0 attribution, is idempotent and rejects unexpected source changes. External unpatched Kompute retains its existing behavior.

## Isolating the two effects

The barrier-only experiment reduced StyleNet's measured host `vkCmdPipelineBarrier` calls from **1,696 to 436 per step** (74.3% fewer API calls, with the same buffer dependencies). Inside Nsight, their total host API interval fell from **1.434 to 0.506 ms**. Those instrumented intervals include tracing overhead; they do not predict a 0.928 ms reduction in ordinary step time.

The uninstrumented barrier-only results were mixed. Autotuning selected different kernels across processes. Fixed register-tile controls also showed substantial training-time variation. A 140-operation add/ReLU chain barely changed: warmed recording medians were approximately 1.272/1.291 ms before and 1.270/1.258 ms after. This change is not presented as a large standalone network speedup.

The subsequent shader experiment holds the compiled runtime and tile choice fixed, changing only the shader directory. Each capture uses eight warmups and thirty samples. Times below are input-gradient GPU dispatch intervals in microseconds:

| Case | Tile | Before | After | Less GPU time |
|---|---|---:|---:|---:|
| Large 3×3 | register | 223.58 | 177.97 | 20.4% |
| Large 3×3 | tall | 462.87 | 331.08 | 28.5% |
| Small 3×3 | tall | 96.32 | 67.79 | 29.6% |
| 10×10 filter | tall | 435.85 | 319.36 | 26.7% |
| 11×11 filter | tall | 529.61 | 388.60 | 26.6% |
| Grouped 3×5 | tall | 60.93 | 47.56 | 21.9% |

Across the unit-stride cases and the two measured tile variants, GPU time drops approximately **14–30%**, equivalent to **1.16–1.42× speed**. The asymmetric nonunit-stride case also passes numerical checks; its measured interval changes from 55.86 to 52.92 µs with the tall tile. These are operator-family improvements, not model substitutions.

Vulkan timestamps bracket dispatch operations (including their dependency recording on the GPU timeline), with uploads/downloads reported separately. The diagnostic uses eager execution for timestamps, not the resident/deferred network schedule. CUDA's comparator records summed device-kernel durations separately from CUDA event intervals and CPU wall time. They are useful directional evidence, not identical measurement boundaries.

## Combined network measurements

RTX 2080 Ti, FP32, identical model/input seeds and architectures. Vulkan runs before/after/after/before; CUDA has two fresh captures. Each process uses twelve warmups and fifty synchronized samples. Training includes forward, loss, backward and AdamW. Vulkan uses the existing fused AdamW; CUDA uses standard AdamW with `foreach=False`. StyleNet uses the model-zoo L1 loss, not a VGG perceptual loss.

| Model / mode | Before Vulkan (ms) | After Vulkan (ms) | CUDA (ms) |
|---|---:|---:|---:|
| U-Net inference | 2.075 | 2.051 | 1.083 |
| U-Net training | 12.268 | 12.245 | 9.469 |
| StyleNet inference | 4.926 | 4.938 | 1.571 |
| StyleNet training | 17.963 | 17.069 | 12.192 |
| MLP inference | 0.740 | 0.731 | 0.224 |
| MLP training | 3.214 | 3.032 | 2.213 |

The pooled training reduction is **5.0% for StyleNet** and **5.6% for MLP**; U-Net is effectively unchanged. StyleNet training run medians are 17.943/18.015 ms before and 16.943/17.183 ms after. MLP is 3.205/3.243 ms before and 3.114/3.001 ms after. These are measurements on this GPU and workload, not a guarantee for other devices or shapes.

The pooled final StyleNet training ratio is about **1.40× CUDA**. CUDA StyleNet training run medians vary from 9.983 to 12.284 ms; retain that variation when interpreting the ratio. Inference is largely unchanged. Input shapes are U-Net `[1,3,64,64]`, StyleNet `[1,3,36,36]`, and MLP `[256,784]`.

The final stage-synchronized diagnostic still has zero warmed allocation/free calls. Barrier API calls remain 1,696 → 436; their instrumented host interval is 1.415 → 0.412 ms. Four stage fence waits total 5.690 ms before and 5.368 ms after. Those four waits are introduced by the profiler stage boundaries and are not a normal-step submission count.

## Largest remaining measured convolution gap

The final automatically selected large-3×3 input-gradient dispatch takes **169.80 µs**, versus **28.71 µs** summed CUDA kernel time on matching inputs: approximately **5.91×**. The earlier automatic Vulkan capture was 248.29 µs, but the fixed-tile experiment above is the stronger evidence for the shader change itself. The large case has input `[2,32,64,64]`, weights `[64,32,3,3]`, stride one and padding one.

In the same final diagnostic, large-3×3 forward is about 3.41× CUDA and weight gradient 1.97×. CUDA is not faster in every case: the grouped input-gradient case measures a lower Vulkan interval. `convolution-comparison.json` retains all eight geometries, all four phases, input hashes and numerical errors. CUDA uses cuDNN heuristics with TF32 disabled; this is not an exhaustive cuDNN tuning study.

Vulkan Nsight workload events remain absent inside measured network stages, so no whole-network Vulkan GPU-time breakdown is claimed. The next concrete target is repeated coordinate decoding and tiling in the input-gradient kernel, tested again across ordinary, rectangular, grouped and dilated convolutions.

## Correctness and portability

- **846 passed** on RTX 2080 Ti resident/deferred.
- **846 passed** on Tesla V100 resident/deferred.
- **845 passed, 1 skipped** on Lavapipe; the skipped test requires a physical GPU.
- **844 passed, 2 skipped** on RTX legacy eager; deferred-only tests are skipped.
- **65 passed** in focused host-visible-storage checks.
- All four regenerated SPIR-V modules pass `spirv-val`.

Six new runtime tests exercise repeated buffer bindings, fan-in/fan-out, in-place aliases, deferred gradients and host-read boundaries. Twelve additional forced-tile cases cover unit-stride grouped/depthwise dilation and both strides nonunit. Existing coverage includes rectangular and 9/10/11/12-sized filters, channel tails, arbitrary layouts/offsets, scratch reductions and partial gradients. The patch was checked against pristine v0.9.0 sources, reapplication, and source-drift rejection before any partial writes.

Outputs, loss and all **102 parameter gradients** are exactly equal to the baseline network captures. All CUDA comparisons pass `rtol=5e-4, atol=2e-5`; the largest parameter-gradient absolute difference is about **1.45e-7**. The final manifest and per-model parity files identify the tested binaries and inputs.

Three pre-existing MSE output-resize warnings remain in the test logs. AMD/Intel hardware remains untested, and no Khronos synchronization-validation-layer run is claimed.

## Evidence and reproduction

[Barrier-only and isolated shader experiments](../benchmarks/results/barrier-batching-ef6aaf3/) retain the initial mixed results, fixed-kernel controls, CUDA kernel profiles, and before/after shader captures. [Combined final results](../benchmarks/results/runtime-conv-ef6aaf3/) contain the final latency/parity captures, test logs, Nsight summaries and manifest. Each network capture records the actual loaded extension hash and SPIR-V hashes; the final manifest also records the source diff and runtime-patch hashes. The same isolated worktree was used throughout, without modifying the primary checkout's concurrent work.

The existing scripts reproduce these measurements:

```bash
# Run separately with baseline/new packages selected by PYTHONPATH.
TORCH_VULKAN_GPU_RESIDENT=1 TORCH_VULKAN_DEFER_SUBMISSIONS=1 \
TORCH_VULKAN_CONV_AUTOTUNE=1 python benchmarks/compare_cuda_vulkan.py \
  --backend vulkan --model style_net --scale medium --warmup 12 --repeats 50 \
  --optimizer vulkan_fused --zoo ../pytorch-XDNA2-vulkan-zoo-reference \
  --output style-net.json

# Isolate the shader change with the same runtime and a fixed tile.
TORCH_VULKAN_GPU_RESIDENT=0 TORCH_VULKAN_DEFER_SUBMISSIONS=0 \
TORCH_VULKAN_PROFILE_TIMESTAMPS=1 TORCH_VULKAN_CONV_TILE=register \
TORCH_VULKAN_CONV_ALGO=tiled TORCH_VULKAN_CONV_BACKWARD_ALGO=optimized \
python benchmarks/probe_conv_dispatch.py --backend vulkan \
  --shader-dir /path/to/before-or-after/shaders --warmup 8 --repeats 30 \
  --out convolution.json
```

CUDA uses the CUDA-enabled environment on the same GPU UUID, `--backend cuda --optimizer standard` for networks, and `--cuda-events --profile-cuda-kernels` for the convolution probe. `capture_network_parity.py` / `compare_network_parity.py` verify model outputs and gradients. `profile_matched_nsys.py` / `summarize_vulkan_stages.py` provide stage diagnostics. `summarize_constant_buffers.py RESULT_DIRECTORY --nsys-directory RAW_DIRECTORY` validates the final capture counts/hashes and creates `summary.json`.

Large Chrome/Nsight/SQLite traces and parity tensors remain under ignored `benchmarks/results/nsys/` in the persistent development worktree. Their stage synchronization is diagnostic and differs from an ordinary deferred training step.
