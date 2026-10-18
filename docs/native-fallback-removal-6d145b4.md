# General CPU-fallback removal and network profile

The measured FP32 U-Net, StyleNet and MLP training steps now contain zero `aten::_to_cpu` or `aten::_to_copy` events. The changes implement general ATen operators; there are no model-name checks, fixed model shapes, replacement networks or diagnostic activation compositions. Unsupported layouts/dtypes retain guarded CPU paths. NLL still validates labels on the host, synchronizing GPU-written labels when necessary.

## Matched end-to-end measurements

RTX 2080 Ti, FP32, identical model/input seeds and original model architectures. CUDA TF32 is disabled. Each variant has two processes with eight warmup iterations and fifteen measured iterations, ordered before/after/after/before, followed by two fresh CUDA runs. Numbers below are pooled medians in milliseconds. Training includes forward, loss, backward and AdamW: Vulkan uses its fused optimizer; CUDA uses `torch.optim.AdamW(foreach=False)`. Initial placement is excluded; internal transfers and final synchronization are included.

| Model | Before training | After training | Speedup | CUDA training | After/CUDA |
|---|---:|---:|---:|---:|---:|
| U-Net | 14.671 | 12.344 | 1.19× | 9.437 | 1.31× |
| StyleNet | 144.013 | 39.770 | 3.62× | 10.300 | 3.86× |
| MLP | 3.645 | 2.817 | 1.29× | 2.189 | 1.29× |

| Model | Before inference | After inference | CUDA inference |
|---|---:|---:|---:|
| U-Net | 2.061 | 2.030 | 1.077 |
| StyleNet | 39.622 | 25.395 | 1.531 |
| MLP | 0.832 | 0.737 | 0.225 |

Training medians from the individual processes show the run-to-run variation:

| Model | Before runs | After runs | CUDA runs |
|---|---|---|---|
| U-Net | 14.704, 13.261 | 12.894, 11.751 | 9.573, 9.420 |
| StyleNet | 147.392, 136.307 | 41.232, 39.725 | 12.038, 9.728 |
| MLP | 3.524, 4.078 | 2.740, 2.962 | 2.182, 2.197 |

Shapes are U-Net `[1,3,64,64]`, StyleNet `[1,3,36,36]`, and MLP `[256,784]`. StyleNet uses the model-zoo L1 loss, not a VGG perceptual objective. These are reduced FP32 workloads, not the original full-batch benchmark. No cross-vendor performance claim is implied.

## General backend changes

- Native FP32 LeakyReLU forward/backward, in-place and out variants, with runtime slopes including negative/nonfinite values and ATen overlap checks.
- Native tensor `add_` and scalar/tensor `sub_`, `mul_`, `div_`, including broadcasting, offsets, strided destinations and mutation aliases.
- Native subtraction/division, sigmoid and its derivative, abs/sgn, log-sigmoid and its derivative, full-tensor mean and fill. Partial-axis means remain on the fallback path.
- Native 1D/2D FP32 NLL forward/backward with class weights, reductions and full-width ignore indices; bounds validation keeps GPU-produced labels safe. Int64 labels use paired 32-bit shader words, requiring no shaderInt64 feature.
- Tiny log-softmax stays native in resident/deferred execution; eager dispatch retains its previous heuristic and explicit override.
- LayerNorm no longer waits unconditionally. Packed affine tensors survive enqueue; readable views pack on GPU. Unsupported packing, including noncontiguous rank above eight, keeps the former CPU path.
- Device `copy_` uses existing GPU packing shaders for FP32/int64, distinct storage and a whole contiguous destination, including broadcast/strided sources. Partial destinations and dtype conversion retain guarded host semantics. Alias checks run before creating host mirrors; exact self-copies remain no-ops.
- `empty_strided` uses ATen’s generic allocation helper and the Vulkan allocator directly, preserving strides, dtype, overflow checks and empty/overlapping layouts.

Most new compute kernels cover FP32 and up to eight dimensions, with explicit index/storage limits and physical-device checks. NLL has its own 1D/2D contract. Unsupported cases are not claimed to be native. The allocator helper itself supports arbitrary rank and existing storage dtypes. Rebuild/validate the new portable shaders with `python scripts/compile_native_fallbacks.py`.

## Trace and correctness evidence

| Model | `_to_cpu` / `_to_copy` per step | Other copy events | Deferred submissions per step |
|---|---:|---|---:|
| U-Net | 0 | 0 `copy_` | 1 |
| StyleNet | 0 | 1 `copy_` | 1 |
| MLP | 0 | 0 `copy_` | 1 |

StyleNet retains its ordinary clone, now a device-to-device packing dispatch. A regression check verifies that this copy does not submit/wait early in deferred mode; the previous host-mirror implementation could transfer data without an `aten::_to_copy` event. Absence of these CPU-copy events is scoped to the measured steps, not every PyTorch operator or all possible inputs. NLL’s host bounds scan is still CPU work. Profiler intervals overlap, can change allocation lifetimes, and must not be added to uninstrumented timing savings.

Untouched-weight captures matched initial-state and input SHA256 hashes. Outputs, losses and **all 102 parameter gradients** (42 U-Net, 50 StyleNet, 10 MLP) passed against both the preserved old backend and CUDA, using `rtol=5e-4, atol=2e-5`. Maximum absolute gradient differences against CUDA were:

| Model | Output max absolute error | Gradient max absolute error |
|---|---:|---:|
| U-Net | 6.14e-06 | 1.35e-07 |
| StyleNet | 4.63e-06 | 5.96e-08 |
| MLP | 2.53e-06 | 9.31e-08 |

Some U-Net normalization-bias gradients are essentially zero, so per-tensor relative error can exceed one despite tiny absolute error; the raw reports retain those values rather than hiding them.

Final validation: 806 tests passed on RTX resident/deferred and V100 resident/deferred; 804 passed with two deferred-only skips on RTX eager; 805 passed with one hardware-only skip on Lavapipe; 351 focused host-visible/deferred checks passed. The three pre-existing MSE resize deprecation warnings remain. AMD and Intel hardware are untested.

## Largest remaining cost and next work

A fresh, stage-synchronized StyleNet Nsight diagnostic records **32 `vkAllocateMemory` calls and about 38.7 `vkFreeMemory` calls per measured step**, totaling **23.86 ms** of host API time. Fence waits account for another 5.85 ms in that diagnostic. These are instrumented host intervals, not Vulkan kernel durations or an additive prediction of removable normal-step time.

`convolution_overrideable` creates `mgr.tensor(std::vector<float>(p.cout, 0.0f))` whenever bias is absent. StyleNet has 16 biasless convolution layers, consistent with 32 device/staging memory allocations. This identifies a concrete candidate behind the allocation churn; attribution of every free to that path is not proven. The next general change should reuse immutable zero-bias buffers by device/dtype/channel count, or represent absent bias directly in a guarded kernel variant, while respecting in-flight lifetime. No model-specific cache key is needed.

After memory churn, investigate convolution algorithm coverage and scheduling. The matched CUDA capture still uses Winograd forward kernels and specialized dInput/dWeight kernels. Vulkan GPU-workload visibility in this Nsight setup remains incomplete, so this report does not invent a whole-network Vulkan GPU-time breakdown or claim that the entire CUDA gap is CPU overhead.

## Reproduction and provenance

Baseline revision: `6d145b40e32e875a395bb48f59c3e7224ad51c98`. Tested compiled extension SHA256: `d4981192039cf936a44d2424d03076e35f9b2828d98eee090d04d8c0251f7190`. The manifest records baseline binary hash, each changed backend/shader/test source hash, script hashes, GPU UUIDs, driver and validation summaries. The profile revision field is the pre-change base; binary/source hashes identify the tested working tree.

Compact results: [native-fallbacks-6d145b4](../benchmarks/results/native-fallbacks-6d145b4/). `summary.json` contains per-run/pooled medians, operator counters and parity summaries; raw profiles, shader hashes and validation logs are alongside it. Larger Chrome/Nsight/SQLite traces and parity tensors remain in the development checkout under ignored `benchmarks/results/nsys/native-fallbacks-6d145b4/`.

Build the baseline and this change in separate worktrees, keeping each package with its matching shader directory. Use `PYTHONPATH` to select that package; the benchmark records its actual loaded extension hash. For each model run Vulkan before/after/after/before, then fresh CUDA twice:

```bash
TORCH_VULKAN_GPU_RESIDENT=1 TORCH_VULKAN_DEFER_SUBMISSIONS=1 \
TORCH_VULKAN_CONV_AUTOTUNE=1 .venv/bin/python benchmarks/compare_cuda_vulkan.py \
  --backend vulkan --model style_net --scale medium --warmup 8 --repeats 15 \
  --optimizer vulkan_fused --zoo /workspace/pytorch-XDNA2-vulkan-zoo-reference \
  --output result.json
```

CUDA uses `.venv-cuda/bin/python`, `--backend cuda --optimizer standard` and the same physical GPU. `capture_network_parity.py` and `compare_network_parity.py` reproduce full-gradient checks. `profile_network.py` records three measured steps after eight warmups; `summarize_native_fallbacks.py` verifies matching hashes and summarizes the captures. Nsight uses `profile_matched_nsys.py`, followed by `summarize_vulkan_stages.py`; those explicit stage synchronizations are diagnostic only.
