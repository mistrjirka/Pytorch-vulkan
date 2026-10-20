# Cross-operator performance audit: direct outputs and reusable execution paths

20 October 2026. Generic PyTorch Vulkan backend; code based on revision 31bc999.
No RTX 3090 was used.

## Why this applies beyond the original optimizer problem

The earlier descriptor-cache and submission investigations identified excess
CPU bookkeeping and many small GPU dispatches. Their resource-lifetime and
LRU improvements already apply to all operators using VulkanContext's
generic get_or_create_algorithm path, not merely Adam, convolution or BMM.

A separate source audit found a widespread pattern in
csrc/ops/out_adapters.cpp: NativeOut invokes the functional variant to allocate
and compute an intermediate result, then copies it into the caller's out
destination. When the functional shader already supports a separately bound
output tensor, this adds one unnecessary allocation and another GPU dispatch.

This is **not** operator fusion or shape-specific shader optimization:
direct output is part of the ordinary PyTorch out= contract.

## Implemented: 33 operation names, four general families

- 8 FP32 unary functions using the existing diffusion_unary shader:
  silu, sin, cos, exp, sqrt, rsqrt, tanh, neg.
- 23 mathematical FP32 unary functions using the existing pointwise_math
  shader: reciprocal, log, log1p, expm1, floor, ceil, trunc, round, sign,
  log10, log2, exp2, sinh, cosh, tan, asin, acos, atan, asinh, acosh,
  atanh, erf, erfc.
- General strided rank-3 bmm.out, using the existing bmm_strided shader.
- General rank-2 mm.out, using the existing tiled/register matmul kernels.

The changes register native PrivateUse1 out kernels for those operations and
remove their formerly boxed NativeOut registrations. Both ordinary functional
operations and their mathematical shaders are otherwise unchanged.

A fast direct-output path requires a compatible FP32 destination with
appropriate shape, contiguity, device, indexing and aliasing behavior.
Input views and transposes remain supported by the existing unary and GEMM
implementations where their shader already handles them. For unsupported
dtypes, output strides, casting, storage aliases, unusual layouts or other
device limits, the original full functional+copy route is retained.
Output resizing and non-overlap checks remain in place.

The user can select the previous path at process start for A/B testing:
- TORCH_VULKAN_DISABLE_DIRECT_UNARY_OUT=1
- TORCH_VULKAN_DISABLE_DIRECT_BMM_OUT=1
- TORCH_VULKAN_DISABLE_DIRECT_MM_OUT=1

There are no model-name branches, hardcoded tensor dimensions or new
GPU-specific instructions. PyTorch still rejects autograd-dependent out=
calls, as in native CPU/CUDA.

## Matched GPU measurements

FP32, GPU resident + deferred submissions, preallocated outputs, 12 warmups
and 75 synchronized iterations per fresh process, two independent processes
per configuration. Medians below are medians of process medians.
The benchmark also checks sample outputs against CPU PyTorch, with NaN
values compared as equal where mathematically expected.

| Workload | GPU | Prior out= latency | Direct-out latency | Speedup | Dispatches before/after |
|---|---|---:|---:|---:|---|
| 200 mixed diffusion unary ops | V100 | 4.766 ms | 2.090 ms | 2.28x | 400 / 200 |
| 200 mixed diffusion unary ops | RTX 2080 Ti | 10.657 ms | 2.021 ms | 5.27x | 400 / 200 |
| 230 mathematical unary ops | V100 | 5.426 ms | 2.162 ms | 2.51x | 460 / 230 |
| 230 mathematical unary ops | RTX 2080 Ti | 13.466 ms | 2.198 ms | 6.13x | 460 / 230 |
| 128 small bmm.out calls | V100 | 4.105 ms | 1.489 ms | 2.76x | 256 / 128 |
| 128 small bmm.out calls | RTX 2080 Ti | 6.300 ms | 1.408 ms | 4.47x | 256 / 128 |
| 128 small mm.out calls | V100 | 3.385 ms | 1.488 ms | 2.28x | 256 / 128 |
| 128 small mm.out calls | RTX 2080 Ti | 7.345 ms | 1.491 ms | 4.93x | 256 / 128 |

The RTX baseline timing was substantially noisier in the original
compute-then-copy cases; the faster Vulkan timings were comparatively
consistent. On V100 the improvement was also repeatable, without changing
shader mathematics.

Matched native CUDA on the same RTX 2080 Ti:

| Workload | New Vulkan | Native CUDA | Vulkan/native |
|---|---:|---:|---:|
| 200 mixed unary ops | 2.021 ms | 1.447 ms | 1.40x |
| 230 math unary ops | 2.198 ms | 1.593 ms | 1.38x |
| 128 bmm.out ops | 1.408 ms | 1.316 ms | 1.07x |
| 128 mm.out ops | 1.491 ms | 1.267 ms | 1.18x |

All four isolated operation-chain workloads are within the 1.5x native
CUDA target. These figures DO NOT imply full-network parity; normal
functional operations, backward passes and convolution algorithms still
need work. CUDA/ROCm vendor library kernels have different asymptotics
for larger GEMMs, so the small-matrix ratios are not universal.

The RX 7800 XT was not connected; no physical ROCm/RADV comparisons for
these changes were collected.

Detailed metadata and every timed sample:
- benchmarks/results/direct-unary-out-2026-10-20/summary.json
- benchmarks/results/direct-unary-out-2026-10-20/raw-timings.jsonl.gz

## Correctness

New tests cover:
- random and fixed input shapes (including empty dimensions);
- arbitrary/strided unary inputs, transposed GEMM inputs;
- noncontiguous and resized output destinations;
- matching input/output aliases, partially overlapping views;
- integer unary inputs that retain their older functional handling;
- input-domain NaNs for root, inverse and logarithmic functions;
- matching PyTorch out= autograd errors;
- disabled direct-output mode for regression comparison.

Focused suite: 168 passed on RTX 2080 Ti, V100 and Lavapipe.
Complete repository suite:
- RTX 2080 Ti: 3876 passed.
- Tesla V100: 3876 passed.
- Lavapipe: 3875 passed, one skipped.

All numerical tests used the same reference operations on CPU, with
separate fallback validation. No CPU numerical fallback occurred in
measured GPU workloads.

## Follow-on cross-operator audit

Using GPU dispatch counters under the same deferred mode, a single
representative out= call currently records:

| Out operator | Vulkan dispatches | Interpretation |
|---|---:|---|
| exp.out | 1 | Fixed |
| log.out | 1 | Fixed |
| mm.out | 1 | Fixed |
| bmm.out | 1 | Fixed |
| addmm.out | 3 | Usually functional GEMM/bias work plus generic output copy |
| _softmax.out | 2 | Functional result and copy; reduction scratch constraints matter |
| sum.out | 2 | Functional reduction result and copy |
| cat.out | 3 | May involve multiple input-copy operations and output copy |

These are actual dispatch counts, not a proof that every surplus dispatch
can be safely eliminated.

**Next candidates**, in order of potential gain and implementation risk:

1. addmm.out: write the existing fused bias+GEMM shader directly to a
   compatible output, but preserve beta/alpha semantics, bias broadcasting,
   alpha=0/beta=0 special cases and output/input alias snapshots.
2. softmax/log_softmax.out: redirect the final normalization output to
   the caller's destination, preserving scratch allocation and backward
   constraints; do not simply remove an essential reduction step.
3. Reduction out variants (sum/mean/max): use direct output where the
   internal reduction's final stage supports it, but preserve accumulator
   dtype, shape, strides, NaN behavior and intermediate scratch.
4. General copy/concat adapters: avoid temporary concatenation outputs
   by writing directly into disjoint destination regions where the
   memory-overlap analysis proves it is safe.

**Not directly transferable:** changing workspace partitioning is
appropriate to spatial Split-K convolution dWeight and possibly other
split-reduction kernels, but not unary math; increasing the bound
Algorithm cache further is not a replacement for per-submission
descriptor lifetime tracking. Neither change is a substitute for faster
native convolution kernels.

PyTorch 2.14 out semantics:
https://docs.pytorch.org/docs/2.14/notes/out.html

Khronos descriptor and command management:
https://docs.vulkan.org/samples/latest/samples/performance/descriptor_management/README.html
https://docs.vulkan.org/samples/latest/samples/performance/command_buffer_usage/README.html
