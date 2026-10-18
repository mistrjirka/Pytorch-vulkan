# Whole-network profile after register convolution tiles

Measured backend: commit `795d34e661593c386218388b93c9ebb4092b5552`, RTX 2080 Ti, FP32, TF32 disabled. The most actionable remaining problem is CPU fallback for common operations. A controlled LeakyReLU diagnostic reduced StyleNet training from 136.6 to 45.9 ms (pooled medians, 2.97×). No backend optimization was added or enabled by this investigation.

## Current network latency

Two fresh processes per backend, eight warmup and 15 timed iterations per process, order Vulkan/CUDA/CUDA/Vulkan for each model. These are uninstrumented full inference or forward + loss + backward + AdamW steps. Vulkan uses resident buffers, deferred submissions, opt-in convolution autotuning and fused AdamW. CUDA uses standard AdamW with foreach disabled. The comparison is between these complete configurations; optimizer implementations differ.

The table shows each run's median rather than hiding process-to-process variation in one pooled value. Times are milliseconds.

| Model / mode | Vulkan run 1 / 2 | CUDA run 1 / 2 |
|---|---:|---:|
| unet inference | 2.033 / 2.022 | 1.071 / 1.081 |
| unet training | 15.246 / 11.022 | 9.272 / 9.433 |
| style_net inference | 39.915 / 39.281 | 1.631 / 1.544 |
| style_net training | 140.724 / 140.284 | 12.344 / 12.086 |
| mlp inference | 0.827 / 0.984 | 0.223 / 0.223 |
| mlp training | 3.888 / 3.998 | 2.201 / 1.989 |

The models are unchanged: U-Net [1,3,64,64], StyleNet [1,3,36,36], MLP [256,784], from zoo commit `c7c063101ec5209caf8022f5b4ea6948d1a8d6ed`. StyleNet uses the zoo's L1 loss, **not** the separate perceptual VGG loss or a batch-40 training workload.

U-Net training varies from 11.0 to 15.2 ms between processes. Selected convolution plans also differ. This establishes variation, not that tuning alone caused all of it. StyleNet has large variation within each run (~101–181 ms), but both normal-run medians are ~140 ms. Exact timings and all samples are retained in the JSON files.

## Highest-impact finding: activation fallback

StyleNet executes 16 LeakyReLU activations and their backward operations. This backend has no native registration for them, so they enter the generic CPU fallback. The fallback first flushes queued GPU work; inputs are copied to the CPU and outputs are copied back. The CPU trace attributes 32 copy operations to forward LeakyReLU and 48 to its backward per step.

The nested copy intervals total 22.64 ms for forward and 36.10 ms for backward in the instrumented operator capture. They include allocation and synchronization costs; they are not pure transfer-engine time. Counting both outer Vulkan and inner CPU calls would incorrectly report 32 activations instead of 16.

A separate Nsight capture shows 71 queue submissions/waits per synchronized training step, 32 Vulkan memory allocations, and about 31 frees. Memory allocation/free APIs occupy about 30.4 ms in that capture. These API intervals overlap the operator intervals above and must not be added to them. Pipeline creation is absent in the measured steady-state ranges: recompiling shaders is not the current bottleneck.

### Controlled diagnostic

For finite inputs, the same activation can be calculated using existing GPU operations:

`y = slope*x + (1-slope)*relu(x)`

The diagnostic custom backward uses the existing threshold kernel to form the positive-input contribution, then applies the same slope. It preserves the model's layers and slope. It is **not a production replacement**: additional rounding and infinity/NaN behavior differ, and generic layout/in-place behavior has not been validated. The custom backward also avoids introducing an extra autograd accumulation fallback from the two branches of the formula.

Stock/composite/composite/stock fresh processes, eight warmup and 15 timed steps each, with an identical first forward/backward parity pass before warmup:

| Implementation | Run medians | Pooled median | Deferred submissions / step | Recorded GPU operations / step |
|---|---:|---:|---:|---:|
| stock | 132.15 / 137.69 ms | 136.62 ms | 36 | 178 |
| composite | 45.60 / 46.25 ms | 45.93 ms | 9 | 306 |

The diagnostic performs more GPU operations but batches them into fewer submissions and avoids activation round trips. Its pooled latency is 66.4% lower. Deferred-submission counters exclude eager copy submissions, so they are not the same count as the Nsight API total.

First-pass parity across all 50 parameter gradients:

| Quantity | Error |
|---|---:|
| Output maximum absolute | 2.62e-6 |
| Loss absolute | 5.96e-8 |
| All gradients relative L2 | 1.68e-6 |
| Worst gradient absolute | 3.73e-8 |

This is measured evidence for prioritizing a GPU-native LeakyReLU implementation. It is not a guaranteed speedup for other networks or a prediction of exact native-kernel latency. Long-run trajectory agreement was not tested.

## What remains after the diagnostic

The composite profile contains no LeakyReLU copy intervals. Its largest remaining copy source is tensor-to-tensor `aten::add_`: ten copy operations occupying 6.11 ms in that instrumented capture. Source inspection confirms `csrc/ops/scalar_add.cpp::add_tensor_` handles a CPU scalar natively but sends the general tensor/tensor case through `x.cpu(); other.cpu(); x.copy_(...)`. Residual gradient accumulation therefore still crosses the CPU boundary.

U-Net has 30 copy operations per profiled step (2.13 ms of nested copy intervals), mostly in the BCE-with-logits loss and its derivative: subtraction, multiplication, log-sigmoid, sigmoid, division and reductions. MLP has 16 copies (0.59 ms), largely log-softmax/NLL loss and their derivatives. These times are attribution clues, not additive estimates of removable wall time; waits and allocation behavior may change when the operations stay on the GPU.

MLP LayerNorm forward/backward also synchronizes unconditionally before native GPU dispatch. This splits otherwise batchable work. The synchronized phase capture has ten submissions/waits per step and 1.59 ms in wait APIs; the ordinary deferred operator capture records eight deferred submissions per step. Phase boundaries themselves add synchronization, and wait time includes useful GPU work.

## CUDA compute comparison and trace limits

The fresh CUDA operator traces contain summed device-event durations of about 1.83 ms for U-Net, 2.78 ms for StyleNet and 0.79 ms for MLP per instrumented training step. These sums include memory operations and exclude CPU/launch gaps; they are not normal end-to-end latency. CUDA convolution traces include Winograd and GEMM families, consistent with the previous [matched convolution investigation](convolution-register-tiles.md).

Vulkan's Nsight GPU-workload capture is incomplete here: some forward intervals appear for U-Net and StyleNet, whole submitted phases are missing, and MLP has no workload table. **GPU utilization and CPU-only residual time cannot be calculated from those captures.** Host API events and NVTX phase ranges are available. The corrected analyzer reports missing/partial Vulkan GPU time as unknown, retains observed intervals as lower bounds and recognizes versioned CUDA API names such as `cudaLaunchKernel_v7000`. CUDA GPU annotations are excluded from kernel-duration sums to avoid double counting.

The earlier isolated large-3×3 input-gradient result remains 9.3× slower in Vulkan device execution than CUDA. This investigation does not replace it with a falsely precise whole-network GPU ratio. The activation diagnostic shows why solving that convolution gap first would miss the largest current StyleNet opportunity.

## Next implementation order

| Priority | General change | Evidence / scope |
|---|---|---|
| 1 | Native LeakyReLU forward and backward | Controlled StyleNet diagnostic: 136.6 → 45.9 ms. Runtime slope and tensor geometry; test zero, negative slope, non-finite values, views and gradients. Preserve unsupported-case fallback. |
| 2 | Native tensor/tensor in-place addition | Largest copy source after the activation diagnostic; common in gradient accumulation. Must preserve broadcasting, dtype, overlap, mutation/version and in-flight buffer semantics. Gain beyond the diagnostic remains unmeasured. |
| 3 | GPU loss operations and remove redundant LayerNorm synchronization | Copy chains recur across U-Net, MLP and StyleNet; unconditional native LayerNorm waits are visible in source. Validate dependencies and lifetimes before removing waits. |
| 4 | Improve convolution algorithm coverage and tuning stability | Winograd/FFT remain promising based on matched CUDA kernels. Add reusable algorithm candidates with general guards; reprofile after eliminating CPU boundaries. No new speed estimate established here. |

Allocation reuse and barrier/descriptor overhead should be revisited after these changes: removing CPU boundaries changes where allocations live and how many commands can be batched. The present captures do not justify a separate allocation-only speedup claim. None of these priorities requires a model-name or fixed-shape branch.

## Reproduction and artifacts

Published compact results: [network-profile-795d34e](../benchmarks/results/network-profile-795d34e/). `summary.json` contains pooled and per-run medians, copy attribution, CUDA events and API totals. `manifest.json` records the core revision, capture hashes and tool/device information. Raw Nsight, SQLite, Chrome traces and parity tensors are retained under `benchmarks/results/nsys/network-795d34e/` in the development checkout and are excluded from git because they contain local process metadata.

Use the commands below with an RTX-selected CUDA environment and the same physical Vulkan device. Clear unrelated experimental overrides. For normal Vulkan latency enable `TORCH_VULKAN_GPU_RESIDENT=1 TORCH_VULKAN_DEFER_SUBMISSIONS=1 TORCH_VULKAN_CONV_AUTOTUNE=1`; do not enable timestamp instrumentation. Use `.venv-cuda/bin/python`, `--backend cuda` and `--optimizer standard` for the CUDA wall comparison.

```bash
.venv/bin/python benchmarks/compare_cuda_vulkan.py --backend vulkan \
  --model style_net --scale medium --optimizer vulkan_fused \
  --zoo ../pytorch-XDNA2-vulkan-zoo-reference --warmup 8 --repeats 15 \
  --output /tmp/style-wall.json

.venv/bin/python benchmarks/profile_network.py --backend vulkan \
  --model style_net --zoo ../pytorch-XDNA2-vulkan-zoo-reference \
  --out /tmp/style-operators.json --trace /tmp/style.trace.json

.venv/bin/python benchmarks/probe_activation_residency.py --variant composite \
  --zoo ../pytorch-XDNA2-vulkan-zoo-reference \
  --out /tmp/composite.json --parity-out /tmp/composite.pt
```

Repeat the diagnostic with `--variant stock` in ABBA order; compare the saved output/loss/all-gradient tensors before interpreting its timings. `profile_network.py --finite-leaky-composite` attributes the diagnostic's remaining costs. The script checks identical initial state and input checksums across backends through `summarize_network_profile.py`.

Nsight uses the existing `profile_matched_nsys.py` with `--scale medium --warmup 8 --samples 3`, `--trace=vulkan,nvtx --vulkan-gpu-workload=individual` or `--trace=cuda,nvtx`, and `--sample=none --cpuctxsw=none`. Export SQLite, then run `analyze_matched_traces.py` and `summarize_vulkan_stages.py`. Those synchronized/instrumented phase durations are diagnostic, not benchmark speed claims.

No backend C++ or shader files changed, so no new cross-device correctness claim is made. The previous 476-test validation applies to the unchanged core. Initial Vulkan/CUDA losses agree within 2.4e-7 across the three models; this is not full numerical trajectory validation. AMD/Intel performance remains unmeasured.
