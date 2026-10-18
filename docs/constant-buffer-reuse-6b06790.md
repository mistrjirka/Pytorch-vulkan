# Reuse immutable Vulkan constants

This change removes recurring zero-bias allocations in biasless Conv2D and uses the same mechanism for missing LayerNorm scale/bias, including the default scale in LayerNorm backward. The baseline is a clean build of `6b06790d866bd76e9f7abe7eb8c6f6d1f11a6a98` in an isolated worktree. Unrelated GroupNorm/diffusion work in the primary checkout was excluded.

## General implementation

`VulkanAllocator::constant_float_buffer(count, value)` supplies internal read-only FP32 zero/one buffers. Capacity rounds up to a power of two: requests for 10 and 11 elements share a 16-element buffer, while each kernel still receives its actual shape. The cache contains at most 17 capacities per value, through 65,536 floats. Their combined device and staging payload is below 2 MiB; driver allocation granularity and object overhead are additional. Larger requests retain the existing temporary Vulkan-buffer path.

Cached allocations stay alive for the allocator's lifetime and never enter its mutable free list. They are registered in the existing residency tracking. In resident mode, the first use records an upload in the normal submission; later readers reuse the initialized device data. No extra CPU synchronization or per-model condition was added. Legacy nonresident execution keeps its existing synchronous upload behavior.

Convolution algorithms, shaders, dtype support, grouping, strides, dilation, layouts and autograd semantics are unchanged. The change does not introduce model names, model-specific channel counts or spatial-size heuristics.

## Validation

- RTX 2080 Ti, resident/deferred: **828 passed** in the full suite.
- Tesla V100: **75 passed** in the focused Conv2D/LayerNorm suite.
- Lavapipe: **74 passed, 1 skipped** in the focused suite.
- RTX legacy eager: **75 passed** in the focused suite.
- RTX resident/deferred with host-visible allocator storage: **75 passed** in the focused suite.

The 22 new checks cover repeated biasless forward/backward, explicit-bias interleaving, grouped/depthwise convolution, rectangular kernels, asymmetric stride/dilation, transposed/sliced/offset/channels-last inputs, mixed queued Conv2D/LayerNorm, neighboring capacities, and sizes just below/at/above the cache limit. Existing tests cover the other operator cases. An existing pipeline-sharing test now retains its input/output allocations to ensure distinct bindings: whole bound-algorithm reuse legitimately bypasses the pipeline lookup counter that test measures.

The three existing MSE output-resize warnings remain tracked in the full-suite log. AMD/Intel execution remains unverified. Lavapipe skips the check that requires a physical GPU.

## Measurement method

Both Vulkan builds and CUDA run on the same RTX 2080 Ti in FP32. Model/input seeds, architecture and input sizes match. Each latency capture has eight warmups and fifteen measured synchronized samples; Vulkan uses before/after/after/before order, followed by two CUDA captures. Results pool the thirty samples per variant; per-run medians and ranges are retained in `summary.json`. Training includes forward, loss, backward and AdamW. Vulkan uses the existing fused AdamW implementation; CUDA uses standard AdamW with `foreach=False`. StyleNet uses the model-zoo L1 loss, not a VGG perceptual loss.

Nsight captures use the same StyleNet workload with explicit stage synchronization and three measured steps after eight warmups. Host API intervals in those captures are diagnostic and must not be added to or subtracted from the ordinary latency numbers as predicted savings.

## Matched latency results

| Model / mode | Before Vulkan (ms) | After Vulkan (ms) | CUDA (ms) |
|---|---:|---:|---:|
| U-Net inference | 2.013 | 2.026 | 1.087 |
| U-Net training | 12.203 | 11.280 | 8.294 |
| StyleNet inference | 25.287 | 4.896 | 1.550 |
| StyleNet training | 39.923 | 17.729 | 12.139 |
| MLP inference | 0.736 | 0.738 | 0.225 |
| MLP training | 2.820 | 2.962 | 2.034 |

StyleNet training improves **2.25×** (55.6% less time); inference improves **5.16×** (80.6% less time). The new training latency is **1.46× CUDA**, and inference is **3.16× CUDA**. Both before/after repetitions show the large StyleNet gain: training run medians are 40.573/39.273 ms before and 17.799/17.618 ms after; inference is 25.384/25.235 ms before and 4.866/4.926 ms after.

U-Net and MLP serve as controls and do not exercise the missing-affine/default-bias paths in these models. Their training results vary across captures; no speedup or no-regression claim is made from their pooled numbers. MLP pooled training is 0.142 ms slower here, with overlapping before/after per-run medians. This does not establish a causal regression. All raw samples and run medians are retained.

For all three models, outputs, loss and all **102 parameter gradients** (42 U-Net, 50 StyleNet, 10 MLP) are exactly equal to the baseline captures. Every CUDA comparison passes `rtol=5e-4, atol=2e-5`; the largest parameter-gradient absolute difference is approximately 1.65e-7.

## Allocation evidence and remaining gap

| Warmed StyleNet diagnostic, per step | Before | After |
|---|---:|---:|
| `vkAllocateMemory` calls | 32 | 0 |
| `vkFreeMemory` calls | 38.67 | 0 |
| Allocation/free host API time | 24.230 ms | 0 ms |

This confirms removal of the observed steady-state memory churn. Startup and first-use allocations still occur. Freed objects can be retained by algorithm descriptors, so their release can appear in later stages than the convolution that created them.

In the fresh, separately synchronized diagnostic, the remaining host stage gaps are:

| Stage | Vulkan mean wall (ms) | CUDA mean wall (ms) | Vulkan fence wait (ms) |
|---|---:|---:|---:|
| forward | 8.773 | 3.248 | 1.882 |
| loss | 0.445 | 0.096 | 0.077 |
| backward | 11.326 | 6.618 | 3.468 |
| optimizer | 3.224 | 6.733 | 0.252 |

Forward and backward remain the main targets; their diagnostic gaps are about 5.53 and 4.71 ms respectively. The fused Vulkan optimizer is faster than the standard CUDA optimizer in this diagnostic, so whole-step ratios do not describe the convolution gap alone.

Command recording also merits measurement: Vulkan records 1,696 pipeline barriers per diagnostic step, totaling 1.388 ms inside that API. Descriptor allocation/update/free totals about 0.344 ms. Those intervals do not include all surrounding C++/PyTorch work and are not a complete CPU breakdown. The four fence waits total 5.679 ms, but this stage-synchronized profiler deliberately introduces four submissions; it does not establish four waits in an ordinary deferred training step.

The matched CUDA trace still uses cuDNN Winograd forward kernels plus specialized dInput/dWeight kernels. It records about 0.910 ms of summed forward CUDA kernel durations and 1.012 ms in backward. Nsight recorded **no Vulkan workload events inside the measured stage ranges**, despite populated host API events. Consequently this report does not claim a measured Vulkan kernel-time ratio or assign the remaining gap wholly to shaders. Next work should isolate matched convolution forward/backward workloads with usable GPU timestamps, then assess general algorithm coverage and redundant command/barrier recording.


## Reproduction and evidence

Compact results and validation logs are in [constant-buffers-6b06790](../benchmarks/results/constant-buffers-6b06790/). `manifest.json` records both compiled-extension hashes, the source diff hash, model-zoo commit, sample counts and device UUID. The baseline binary was rebuilt from clean source; its hash differs from the earlier build because it was built in the isolated directory. The runner verifies the imported extension hash in every Vulkan latency capture.

Existing scripts reproduce the measurements:

```bash
TORCH_VULKAN_GPU_RESIDENT=1 TORCH_VULKAN_DEFER_SUBMISSIONS=1 \
TORCH_VULKAN_CONV_AUTOTUNE=1 python benchmarks/compare_cuda_vulkan.py \
  --backend vulkan --model style_net --scale medium \
  --warmup 8 --repeats 15 --optimizer vulkan_fused \
  --zoo ../pytorch-XDNA2-vulkan-zoo-reference --output style-net.json
```

Run separately with each package selected via `PYTHONPATH`; CUDA uses its CUDA-enabled Python environment, the same GPU UUID, `--backend cuda --optimizer standard`. Repeat for `unet` and `mlp`. `capture_network_parity.py` and `compare_network_parity.py` compare outputs, loss and every parameter gradient. `profile_matched_nsys.py` supplies NVTX stage ranges; `summarize_vulkan_stages.py` summarizes the Vulkan API captures. `summarize_constant_buffers.py RESULT_DIRECTORY --nsys-directory RAW_DIRECTORY` checks binary hashes, aggregates timings/parity, and extracts the matched stage evidence. Large Nsight/SQLite captures and parity tensors are retained under ignored `benchmarks/results/nsys/constant-buffers-6b06790/` in the development worktree.
