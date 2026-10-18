# Native pooling, concatenation, and completed-fence reuse

Measured October 18, 2026, on the RTX 2080 Ti with PyTorch 2.14.1. This
iteration removes pooling and concatenation CPU round trips, packs arithmetic
views on the GPU, fixes CPU scalar multiplication, and reuses completed Vulkan
fences. All changes use general tensor dimensions and Vulkan compute; there is
no model-name or NVIDIA-only dispatch.

## Matched model result

Unchanged XDNA2 U-Net architecture, 472,833 parameters, FP32 input
`[1,3,64,64]`, model seed 42, input seed 123, eight warmups, fifteen synchronized
wall-time samples, eight CPU threads. Model source is XDNA2 commit
`c7c063101ec5209caf8022f5b4ea6948d1a8d6ed`. Training includes forward, loss,
backward, and AdamW. Vulkan uses experimental GPU residency, deferred submission,
and the existing opt-in fused AdamW; CUDA uses ordinary PyTorch AdamW with TF32
disabled. Placement and initial compilation are outside warmed timings.

| Workload | Previous Vulkan median (range), ms | Updated Vulkan median (range), ms | CUDA median (range), ms | Vulkan improvement |
|---|---:|---:|---:|---:|
| Inference | 39.202 (37.836–39.963) | 3.321 (3.072–3.637) | 1.083 (1.075–1.149) | 11.80× |
| Full training step | 86.383 (82.389–119.701) | 17.204 (17.017–17.948) | 9.480 (8.470–9.571) | 5.02× |

The initial training loss is exactly `0.7050257325172424` in all three runs.
This is a consistency check, not proof of long-run convergence. The previous
path is selected in the same rebuilt extension by disabling all five new
optimizations. Raw samples and controls are in
[`benchmarks/results/pooling/`](../benchmarks/results/pooling/).

## General operator coverage

- `max_pool2d_with_indices` and backward support FP32 CHW/NCHW, rectangular
  kernels, independent strides/padding/dilation, ceil mode, empty batches,
  channels-last, transposed, sliced, offset and broadcast views. Output indices
  use int64 storage represented by two 32-bit shader words, so shaderInt64 is
  not required. First equal maxima and last-NaN selection match CPU. Backward
  gathers overlapping contributions deterministically using saved indices;
  it needs no floating-point atomics.
- FP32 `cat` supports arbitrary axes through rank eight, empty inputs, strided
  layouts, repeated input allocations and autograd. Other dtypes, mixed-dtype
  promotion and higher ranks retain the CPU fallback.
- Arithmetic packs FP32/int64 views on GPU through rank eight. CPU zero-dimensional
  scalars in FP32 `add.Tensor`/`mul.Tensor` become push constants, with PyTorch
  scalar type promotion checked before dispatch. Both operand orders and scalar
  autograd are covered. This fixes Python floating-point multiplication that
  previously reached an FP32-only tensor check with a CPU double scalar.
- Dispatch checks tensor indexing, spatial arithmetic, storage-buffer ranges
  and physical workgroup limits. Unsupported sizes fail explicitly. Pooling
  currently requires FP32 and int32 spatial indexing; other pooling families
  still use the existing fallback.

The new shaders use 128-thread workgroups. Packing is a separate dispatch and
can remain expensive for broadcast views; it is not a stride-aware arithmetic
fusion.

## Why Vulkan submission was slower

A matched native pooling workload showed only 5.25 microseconds of Vulkan GPU
work for `[1,32,64,64]` forward, versus 3.52 microseconds for CUDA's kernel,
while Vulkan wall time was about half a millisecond. Nsight identified the
larger cost in host-side fence lifetime:

| Vulkan API | Before: calls over 60 submissions | Before: median per call | After: calls |
|---|---:|---:|---:|
| `vkCreateFence` | 60 | 217.1 µs | 1 |
| `vkDestroyFence` | 60 | 193.5 µs | 1 |
| `vkWaitForFences` | 60 | 67.0 µs | 60 |

With completed-fence reuse, an unprofiled 50-sample probe drops from **0.544 ms
to 0.091 ms** total median, after ten warmups. The new fence pool is owned by
Kompute's Manager and shared by its sequences. Each submitted sequence holds
an exclusive fence until successful completion. Timeouts keep the fence and
pending resources; `clear()` rejects in-flight sequences; destruction waits
before releasing command resources. At most 32 idle fences are retained.
The pool shuts down before the device. This is fence reuse, not command-buffer
reuse. The patch applies to the bundled Kompute dependency.

For the larger overlapping `[2,64,64,64]` pool, Vulkan forward/backward GPU
measurements were 27.72/60.47 µs; CUDA kernel times were 29.32/37.78 µs.
Vulkan backward still has a kernel-level gap, but the forward wall-time gap
cannot be explained by shader execution alone. Vulkan timestamps use eager
submission while wall-time probes use deferred submission; CUDA uses its
profiler. These measurements diagnose the layers of cost, not an exact sum.

Selected synchronized operator medians, five warmups and fifteen repeats:

| Operator and input | Previous Vulkan, ms | Updated Vulkan, ms | CUDA, ms |
|---|---:|---:|---:|
| Pool forward `[1,32,64,64]`, 2×2/stride 2 | 4.650 | 0.122 | 0.027 |
| Pool backward, same | 7.153 | 0.101 | 0.044 |
| Pool forward `[2,64,64,64]`, 3×3/stride 1/pad 1 | 48.626 | 0.457 | 0.053 |
| Pool backward, same | 54.127 | 0.264 | 0.070 |
| Channel cat, two `[1,64,32,32]` tensors | 5.193 | 0.095 | 0.026 |
| Offset view + broadcast add `[1,64,32,32]` | 2.707 | 0.130 | 0.026 |

Every operator probe checks outputs, indices and gradients against the same
seeded CPU reference before timing. Tiny isolated CHW pooling `[3,7,9]` is a
counterexample to universal speedup: the old CPU fallback takes 0.034/0.036 ms
forward/backward versus 0.070/0.070 ms natively. Keeping such tensors on GPU
can benefit a larger graph, but does not make an isolated tiny call faster.

## Validation and remaining costs

- Full suite: **324 passed on RTX 2080 Ti; 324 on V100; 323 passed and one
  hardware-only skip on Lavapipe**.
- Full RTX suite with GPU residency and deferred submission: **324 passed**.
- New pooling/cat/scalar/layout tests: **110 passed** with residency/deferred
  submission, also **110 passed** with mapped host-visible buffers enabled.
- C++ real-queue regression passed on RTX, V100 and Lavapipe, and with fence
  pooling disabled. A host-controlled Vulkan event forces zero-timeout waits
  and destruction while pending; the test also submits two concurrent leases
  and repeatedly creates new sequences to exercise reuse.
- All five new SPIR-V binaries pass `spirv-val`.

The final matched operator profile still finds 50 `_to_copy` calls per U-Net
training step. With profiler overhead, their self CPU time is 2.05 ms;
`adamw_step_batch_` takes 4.68 ms, add 2.89 ms, and allocation 2.39 ms. These
are CPU attribution figures, not additive GPU kernel times. Follow-up work
should inspect optimizer allocations, scalar integer counter updates and
remaining loss fallbacks against the saved CUDA profile. AMD/Intel hardware
has not been tested, and the experimental residency modes remain opt-in.

## Reproduce

```sh
export TORCH_VULKAN_GPU_RESIDENT=1 TORCH_VULKAN_DEFER_SUBMISSIONS=1
.venv/bin/python benchmarks/compare_cuda_vulkan.py \
  --model unet --backend vulkan --scale medium --warmup 8 --repeats 15 \
  --optimizer vulkan_fused --zoo /path/to/pytorch-XDNA2 --output /tmp/unet.json
.venv/bin/python benchmarks/probe_pool_and_layouts.py \
  --backend vulkan --out /tmp/operators.json
.venv/bin/python benchmarks/probe_submission.py
```

For the previous model path, set these before process startup:
`TORCH_VULKAN_DISABLE_NATIVE_POOL=1`, `TORCH_VULKAN_DISABLE_NATIVE_CAT=1`,
`TORCH_VULKAN_GPU_PACK_VIEWS=0`, `TORCH_VULKAN_DISABLE_SCALAR_BINARY=1`, and
`TORCH_VULKAN_DISABLE_FENCE_POOL=1`. They can be toggled independently.
For CUDA, use the CUDA-enabled Python environment, select the same physical
GPU, change backend to `cuda`, and omit the Vulkan optimizer selection.

```sh
nsys profile --trace=vulkan --vulkan-gpu-workload=individual \
  --sample=none --cpuctxsw=none -o /tmp/submission \
  .venv/bin/python benchmarks/probe_submission.py
nsys stats --report vulkan_api_sum --format csv /tmp/submission.nsys-rep
cmake -S . -B .cmake-build -DTORCH_VULKAN_BUILD_RUNTIME_TESTS=ON
cmake --build .cmake-build --target test_fence_pool
.cmake-build/test_fence_pool 0
```

CMake must first be configured with this project's normal Torch/Python build
settings (for example by `setup.py build_ext --inplace`). The C++ test accepts
an optional physical-device index. Sanitized Nsight API summaries, operator
profiles, intermediate measurements and final A/B JSON are stored with the
benchmark results; raw Nsight process traces are not committed.
