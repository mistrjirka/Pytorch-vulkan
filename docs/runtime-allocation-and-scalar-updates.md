# Reused storage, scalar updates, and deferred AdamW

This iteration removes CPU clearing of reused tensor storage, keeps scalar
updates on Vulkan, and lets AdamW join pending GPU work. These are general
runtime/operator changes, with no model-name dispatch.

## Matched end-to-end results

RTX 2080 Ti, FP32 XDNA2 U-Net `[1,3,64,64]`, 472,833 parameters, model seed 42,
input seed 123, eight warmups and fifteen timed iterations per process. Vulkan
uses GPU residency, deferred submission and opt-in fused AdamW; CUDA uses
ordinary PyTorch AdamW, with TF32 disabled. Model source is XDNA2 commit
`c7c063101ec5209caf8022f5b4ea6948d1a8d6ed`.

Runs were ordered previous/updated/updated/previous. The table pools the 30
samples for each Vulkan configuration; CUDA has 15 samples. Training includes
forward, loss, backward and the optimizer update. Placement and compilation
are outside the warmed measurements.

| Workload | Previous Vulkan median (range), ms | Updated Vulkan median (range), ms | CUDA median, ms | Latency reduction |
|---|---:|---:|---:|---:|
| Inference | 3.281 (3.090–3.560) | 2.507 (2.282–3.047) | 1.091 | 23.6% |
| Training | 19.159 (17.730–20.865) | 14.956 (13.161–15.322) | 9.558 | 21.9% |

The individual updated training-run medians were 13.279 and 15.052 ms; previous
medians were 19.669 and 18.939 ms. The previous implementation measured 17.204 ms
in the earlier iteration. Use the freshly interleaved comparison for the
improvement claim; do not subtract measurements from different sessions.
All runs, including CUDA, have initial loss `0.7050257325172424`.

Raw measurements are in
[`benchmarks/results/runtime-overhead/`](../benchmarks/results/runtime-overhead/).

## Changes and semantics

**Reuse storage without clearing it.** The allocator previously cleared the
entire mapped CPU staging buffer whenever an allocation was reused. `empty()`
does not promise initialized contents, and its consumer initializes the values
it needs. The unnecessary memset is removed; new allocations still use the
existing initialization path. This does not change `zeros`, explicit fills,
host/device dirty tracking, or the rule that pending GPU users prevent reuse.
`TORCH_VULKAN_ZERO_REUSED_BUFFERS=1` restores clearing for A/B measurements.

**Add scalars on the GPU.** FP32 and int64 `add.Scalar`, `add_.Scalar`, and
CPU-zero-dimensional-scalar cases of `add_.Tensor` now have native dispatch.
Rank zero through eight, offsets, permutations, and non-overlapping stepped
views are supported. In-place updates retain storage aliases. Ambiguous
layouts, other dtypes, and tensor/tensor in-place addition keep the CPU path.
Type promotion and alpha validation use ATen rules. Int64 values use two
32-bit words with explicit carry and wraparound; shaderInt64 is unnecessary.
The shader checks dispatch/storage limits through the shared native utility.

This removes the ten CPU scalar additions used by U-Net BatchNorm counters,
which previously interrupted unrelated queued work. It also removes twenty
copy/conversion calls and twenty allocations per training step.
`TORCH_VULKAN_DISABLE_NATIVE_SCALAR_ADD=1` restores previous dispatch.

**Keep AdamW in the queued graph.** In GPU-resident deferred mode, updates now
follow gradient-producing commands in the same sequence, using the existing
memory barriers and buffer-lifetime rules. The eager behavior is unchanged.
`TORCH_VULKAN_ADAMW_FORCE_SYNC=1` restores the pre-optimizer CPU wait. This
removes an execution boundary; it does not change the optimizer's shader math.

## Profiling and causal checks

Latest matched CPU operator profiles (seed 2026, three profiled steps after
eight warmups) give the following self CPU attribution:

| Operation | Previous | Updated |
|---|---:|---:|
| Allocation (`aten::empty`) | 2.497 ms, 228 calls | 0.293 ms, 208 calls |
| Copy/conversion (`aten::_to_copy`) | 2.150 ms, 50 calls | 1.954 ms, 30 calls |
| Addition (`aten::add`) | 2.986 ms, 12 calls | 0.087 ms, 2 calls |
| Fused AdamW entry point | 4.675 ms | 0.946 ms |

The optimizer attribution previously included waiting for backward GPU work.
Its reduction is not a 4.9× optimizer-compute speedup. Likewise, the additions
include waits previously forced by CPU fallback. These profiles locate costs;
they must not be added to or substituted for unprofiled wall times.

Restoring one old behavior at a time, with the other fixes enabled, gives:

| Restored behavior | Inference median, ms | Training median, ms |
|---|---:|---:|
| Clear reused buffers | 3.374 | 18.086 |
| Wait before AdamW | 2.521 | 16.101 |
| CPU scalar additions | 2.488 | 17.322 |

These separate-process ablations are noisy and their differences are not
additive. Each supports retaining the corresponding change.

## Remaining CUDA gap

To avoid charging unfinished backward work to AdamW, the stage probe explicitly
synchronizes before and after each stage. Fifteen warmed samples on each backend:

| Stage | Vulkan median, ms | CUDA median, ms |
|---|---:|---:|
| Forward, training mode | 4.476 | 1.570 |
| Loss | 1.485 | 0.076 |
| Backward | 8.042 | 2.084 |
| Optimizer | 1.861 | 3.796 |

These stage boundaries alter scheduling, so their sum is not the end-to-end
benchmark. CUDA here uses ordinary AdamW with `foreach=False`, not CUDA's fused
optimizer. The Vulkan optimizer is not the largest remaining comparative gap.
Loss fallbacks and forward/backward execution remain the useful targets.

An additional three-step Nsight capture excludes warmup and records Vulkan API
time within the NVTX stage ranges. Mean fence-wait time was 1.670 ms in forward,
3.459 ms in backward, and only 0.148 ms in optimizer. Recording also issues
333/386/419 pipeline barriers in these respective stages. This suggests examining
barrier recording and dependency handling as well as convolution kernels next.
Wait time includes driver scheduling and is not exact GPU kernel duration.
This capture exposed host API events, not Vulkan GPU workload timestamps; no
kernel-time claim is inferred from it. The sanitized API summary is committed;
raw process traces remain outside the repository.

## Validation

- Full suite: **361 passed on RTX 2080 Ti; 361 on V100; 360 passed and one
  hardware-only skip on Lavapipe**.
- Full GPU-resident deferred RTX suite: **361 passed**.
- Mapped host-visible focused suite: **46 passed**.
- Scalar tests cover int64 carries/overflow, signed boundaries, invalid casts
  and alpha, storage aliasing, stepped/offset/transposed views, empty tensors,
  rank-eight native and rank-nine fallback paths, expanded-overlap rejection,
  autograd, and no submission caused by counter updates.
- AdamW regression queues several gradient/update/read cycles and compares
  parameters and both moments against CPU AdamW.
- Both new shader binaries pass `spirv-val`.

AMD/Intel hardware remains untested. Residency and deferral remain opt-in.

## Reproduce

Use the normal `compare_cuda_vulkan.py` U-Net command from the
[previous report](pooling-layouts-and-fences.md), with GPU residency and deferred
submission enabled. Set all three diagnostic flags above to `1` for the prior
implementation; leave them unset for the updated implementation.

The existing stage profiler now accepts any model in the benchmark configuration:

```sh
python benchmarks/profile_matched_nsys.py --backend vulkan \
  --model unet --scale medium --samples 15 --warmup 8 \
  --zoo /path/to/pytorch-XDNA2 --result /tmp/stages.json
```

Use the CUDA Python environment, `--backend cuda --optimizer standard`, and the
same physical GPU for comparison. For API attribution, capture with
`nsys profile --trace=vulkan,nvtx --sample=none --cpuctxsw=none`, export to SQLite,
then run:

```sh
python benchmarks/summarize_vulkan_stages.py /tmp/capture.sqlite \
  --out /tmp/api-stages.json
```
