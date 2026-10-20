# Bounded Vulkan descriptor/runtime cache redesign

20 October 2026. This is a generic runtime change, not a model-specific shader or fused optimizer.

## Root cause

The Vulkan backend already shared immutable shader pipelines but cached the
mutable tensor-bound Kompute Algorithms by VkBuffer handle tuple. The prior
512-entry FIFO aggressively evicted recurring bindings; standard FP32
Adam incurred hundreds of costly descriptor/Algorithm creations per step.
Unbounded retention would improve hits at the expense of GPU-buffer lifetime
and memory usage.

## Implemented architecture

1. **Descriptor-set recycler.** The pinned Kompute patch now pools free
   VkDescriptorSets per compatible layout. Each in-use Algorithm owns
   a private descriptor set. When a recorded OpAlgoDispatch retains an
   Algorithm, that set cannot be recycled until the recorded/in-flight
   Sequence releases its final owner after GPU completion. No
   UPDATE_AFTER_BIND or descriptor-indexing features are required.
   Each layout arena retains at most 256 idle sets, frees excess idle
   sets and destroys completely empty descriptor pools.

2. **Buffer-aware LRU, separate from the immutable pipeline cache.**
   The bound Algorithm cache now uses LRU in place of FIFO and tracks
   distinct VkBuffer allocations referenced by cached descriptors,
   counting each buffer only once no matter how many Algorithms use it.
   Cache limits default to 4096 entries and 512 MiB of referenced
   buffer bytes, configurable through environment options below.
   A candidate that cannot fit executes on the GPU uncached, with
   no numerical fallback or change to operation semantics.

3. **Reproducible patch.** All Kompute changes are made by
   scripts/patch_descriptor_arenas.py during CMake setup. The script is
   idempotent. Pipeline binaries still share the existing independent
   immutable cache. No external GPU kernels were copied.

Environment options (read at process startup):

- TORCH_VULKAN_BOUND_ALGORITHM_CACHE_CAP: 128..16384; default 4096.
- TORCH_VULKAN_BOUND_ALGORITHM_CACHE_MB: 0..16384 MiB; default 512.
- TORCH_VULKAN_DESCRIPTOR_SET_REUSE: 1 (default), 0 to disable pooling
  for A/B experiments.

The cache byte count is a *unique-buffer reference budget*, not an
exact device-memory usage reading. Lifetime-safe in-flight references
and the allocator's own free lists have separate residency rules.

## Reproduced performance

FP32 Lab02 MyNet, 50 parameter tensors / 2,420,547 elements,
batch 1, 128×128 patches, standard Adam with no changes to its
mathematics. All timed Vulkan tests used GPU residency, deferred
submission and CPU fallback forbidden.

**V100 optimizer-only** (12 warmups; 80 timed synchronized steps per process):

| Configuration | Median optimizer time |
|---|---:|
| Cache 512, descriptor recycling disabled | 12.482 ms |
| Cache 512, descriptor recycling enabled | 11.942 ms |
| LRU 4096 / 512 MiB, recycling disabled | 8.172 ms |
| LRU 4096 / 512 MiB, recycling enabled | 8.204 / 8.249 ms |
| LRU 4096 / 64 MiB, recycling enabled | 11.575 ms |

The full default 4096/512 MiB policy reduces isolated Adam time by
approximately 34% compared with the 512-entry baseline. Descriptor set
recycling alone offers much less of the improvement; recurring bound
Algorithm reuse matters more. The 4096-entry cache retained about
192 MiB of distinct Vulkan-buffer references in that workload.

**V100 full-network training** (two independent processes per setting;
12 warmups, 70 synchronized steps each):

| Cache entries | Process medians | Median of process medians |
|---|---|---:|
| 512 | 68.458 / 68.440 ms | 68.449 ms |
| 4096, 512 MiB budget | 60.148 / 60.204 ms | **60.176 ms** |

The complete step is **12.1% faster** and incurs no CPU fallbacks.
Bound distinct-buffer references ended near **458 MiB**, below the
configured 512 MiB ceiling.

RTX 2080 Ti isolated optimizer runs also favored the larger cache,
but their 512-entry control was unusually noisy; this report does
not use those numbers for a precise speedup claim. The RTX 3090
was untouched. AMD RX 7800 XT could not be benchmarked because its
remote host was not connected.

Raw timing series and cache counters:
- benchmarks/results/descriptor-runtime-redesign-2026-10-20/summary.json
- benchmarks/results/descriptor-runtime-redesign-2026-10-20/raw-timings.jsonl.gz

## Correctness

- RTX 2080 Ti: **3708 passed** in the complete repository suite.
- Tesla V100: **3708 passed**.
- Lavapipe: **3707 passed, one skipped**.
- New tests cover 360 simultaneously live outputs, changing push
  constants, asynchronous chained temporaries, explicit 128-entry
  eviction, a zero-sized cache budget, 1 MiB retained-buffer cap,
  and strict bounds on cached buffer bytes. No CPU fallback was used
  in the focused lifetime and memory-limit tests.
- Focused tests also passed using the prior nonresident/eager mode.
- The Kompute patch was rerun successfully to check idempotence.

These tests establish numerical correctness on the tested devices.
They do not replace Vulkan validation-layer testing on every driver.
The design relies on Algorithm ownership lasting through completion
of the GPU commands that reference it; it does not rewrite active
descriptor sets.

## Remaining work

This is a substantive cache/resource-lifetime redesign, but not the
complete Vulkan execution-engine replacement. A cache miss still creates
a new Kompute Algorithm. A larger phase could use a per-submission
descriptor/command arena keyed to GPU completion fences to reduce
construction overhead further without retaining buffer identities in
the global cache.

Other general inefficiencies remain: functional-then-copy handling for
many out= operations, default eager/nonresident mode, redundant
barriers and missing high-performance convolution/backward kernels.
The full-model target of Vulkan within 1.5x native CUDA/ROCm is
not achieved yet.

Khronos descriptor lifetime and reuse references:
- https://docs.vulkan.org/spec/latest/chapters/descriptorsets.html
- https://docs.vulkan.org/samples/latest/samples/performance/descriptor_management/README.html
