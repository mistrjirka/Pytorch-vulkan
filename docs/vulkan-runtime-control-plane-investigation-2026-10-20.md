# Vulkan runtime architecture investigation

20 October 2026. Baseline revision f1945a2. Matched Vulkan experiments
ran on RTX 2080 Ti, Tesla V100, and Lavapipe. The RTX 3090 was not used.

## Finding

The large CUDA/ROCm versus Vulkan gap is **not solely missing fusion**.
This experimental backend pays excessive host-side control-plane costs per
PyTorch operation. With GPU residency + deferred submission enabled, the
largest measured CPU costs on V100 are:

- Constructing per-binding Kompute Algorithms and VkDescriptorSets.
- Recording hundreds of dispatches and barriers.
- Python/ATen dispatcher and other host bookkeeping.

The GPU kernels still have independent algorithm-performance shortcomings
(convolution backward, normalization, attention). Fixing runtime host
overhead alone cannot guarantee the 1.5x CUDA goal.

## Native CPU timing

Opt-in CPU timers were added around allocation, bound-Algorithm cache
lookups and misses, and deferred command recording. V100, 50 parameter
tensors with 2,420,547 elements, standard PyTorch Adam FP32, static
resident gradients, 70 measured steps after warm-up:

| Part of a standard Adam step | Measured CPU time |
|---|---:|
| Total (with probes) | 12.31 ms |
| CPU enqueue, before submit/sync | 8.99 ms |
| Submit/sync and GPU completion | 3.33 ms |
| Creating/binding Algorithms (cache misses) | 3.13 ms |
| Recording deferred dispatches/barriers | 1.92 ms |
| Allocator itself | 0.10 ms |

The accounting includes overlapping boundaries (total and subparts).
The uninstrumented standard Adam optimizer was ~11.66 ms. All tests
here forbid numerical CPU fallback; there were no host read calls.

An important initial theory was **disproved**: despite about 300
temporary tensor allocations per step, all 300 were **recycled** in
the warmed steady-state run. Zero *fresh* allocator requests per step
were observed. The allocator is not directly responsible for the
majority of this 9 ms enqueue cost. It may still affect memory pressure,
buffer identity, and descriptor reuse.

The V100 GPU timestamp trace for the same optimizer was 400 shader
dispatches and about 3.18 ms shader execution per step. It was taken
in separately instrumented eager mode: do not subtract this GPU number
from deferred wall-clock timing as exact CPU attribution.

## Verified descriptor cache cliff

The bound-Algorithm cache is keyed by shader name and actual VkBuffer
handles/offsets/ranges, with a fixed 512-entry FIFO. Immutable shader
pipelines are cached separately and generally hit. A large training
step creates short-lived buffer binding combinations that evict
potentially reusable descriptor sets each iteration.

The diagnostic option TORCH_VULKAN_BOUND_ALGORITHM_CACHE_CAP selects
a bounded capacity (128..16384) while preserving the default 512.

On V100, 70 warmed synchronized standard Adam optimizer-only steps:

| Bound-Algorithm cap | Misses/step | Miss-path CPU/step | Median optimizer step |
|---|---:|---:|---:|
| 512 (repeat) | 367 | 3.25 ms | 12.04 ms |
| 4096 | 30.4 | 0.30 ms | 8.69 ms |
| 8192 | 30.0 | 0.68 ms | 8.54 ms |

This is a ~28% lower latency at capacity 4096 **without altering a
kernel, operation or optimizer mathematical semantics**.

But larger caches pin more descriptor state and potentially underlying
buffers; 8192 retained ~6400 bindings and increased peak process RSS
from ~1.13 GiB to ~1.18 GiB in this probe. Device memory usage was NOT
measured by the RSS counter. An arbitrarily enormous static cache
is not a general solution.

Full 128x128 Lab02 MyNet V100 training (45 warmed timed steps/process),
same standard Adam and parameter shapes, same precision:

- Capacity 512: medians **67.54 and 65.14 ms**.
- Capacity 4096: medians **60.41 and 61.85 ms**.

This confirms the issue affects the complete network, with roughly
7-10% overall improvement, less dramatic than optimizer-only because
GPU convolution math still dominates some phases.

## Falsified/smaller hypotheses

### Unconditional transfer barriers

Kompute's OpAlgoDispatch::record emits a transfer->compute barrier
across **all** bound buffers, even when nothing was uploaded. The
deferred VulkanContext also emits compute dependency/output barriers.

A strictly local A/B modified this so new uploads receive an explicit
transfer barrier while non-uploaded bindings do not receive a
redundant transfer barrier. Results for the V100 Adam-only workload:

- Original: **11.99 and 11.48 ms**.
- Transfer-aware: **11.38 and 11.36 ms**.

The speed gain is modest. This mechanism is real, but does **not**
explain most of the CUDA performance gap. The experimental barrier
change was reverted; no unverified weakening of synchronization was
left in the implementation.

### Batching descriptor writes

A local experiment changed Kompute Algorithm::createParameters from
one vkUpdateDescriptorSets call **per tensor** to one call per complete
descriptor set. Default-cache V100 optimizer medians: **11.93 and
12.22 ms**, similar to the earlier ~12 ms. The change was reverted.
The expensive part is rebuilding and evicting many entire bound
Algorithms; batching a few binding writes does not solve it.

### GPU residency and deferred submissions are not enabled by default

A V100 30-operation serial FP32 add chain, 40 timed synchronized
samples, 512 elements/operator:

| Execution configuration | Median |
|---|---:|
| Default (nonresident, not deferred) | 1.768 ms |
| GPU resident, eager submit | 1.508 ms |
| GPU resident, deferred submit | **0.362 ms** |

This is a **4.9x default-mode** slowdown for the chain. It does not
explain the already-optimized training comparisons, which explicitly
enabled both options. Enabling them by default needs full numerical
and CPU-read/aliasing/unsupported-op fallback validation.

### Generic out= adapter doubles some GPU operations

The source in csrc/ops/out_adapters.cpp calls the functional operator
to a temporary tensor, then copies that tensor to the requested
output. A controlled 400-operation V100 FP32 add chain produced:

- Functional addition: **400 GPU dispatches**, ~5.10-5.16 ms.
- Preallocated out= form: **800 GPU dispatches**, ~14.55-14.75 ms.

The two versions allocate/own outputs differently, so the timing
ratio is not solely attributable to an extra copy. However, this
confirms a general suboptimal adapter design. A fast out= path should
write straight to the provided output after correct alias/stride and
dtype validation, retaining the generic copy fallback where needed.

## Independent DeepSeek v4.1 investigation

Three read-only OpenCode free DeepSeek v4.1 agents examined
different subsystems: VulkanContext/Kompute synchronization, allocator
and descriptor lifetime, optimizer and ATen dispatch. Two produced
completed audits, one broader search did not yield a coherent final
report. No agent code edits were accepted.

Their recommendations prioritized descriptor-cache churn, delayed
buffer reuse, per-op command-pool creation for eager mode and excess
barriers. Our direct A/B results refined and corrected these:
- Driver/descriptor *misses* cost ~3.1 ms per standard Adam step.
- Warmed allocations were almost entirely free-list reuse, not fresh
  Vulkan memory allocations.
- The ~12 ms optimizer run was **deferred**, not eager; one subagent
  mistakenly attributed it to per-op fence waits.
- Redundant transfer barrier removal was only a small win.

## External design comparisons

Khronos recommends minimizing unneeded synchronization and updating
descriptors outside hot loops, recycling command pools after in-flight
work completes. ncnn separates blob/workspace allocators and caches
compiled pipelines. AMD's VMA suballocates larger VkDeviceMemory blocks
and manages resource lifetime. These are architectural references,
NOT licenses to copy third-party GPU kernels without verification.

References:
- https://docs.vulkan.org/guide/latest/profiling.html
- https://docs.vulkan.org/tutorial/latest/ML_Inference/Vulkan_Compute_for_ML/06_synchronization.html
- https://github.com/KhronosGroup/Vulkan-Samples/blob/main/samples/performance/descriptor_management/README.adoc
- https://docs.vulkan.org/samples/latest/samples/performance/command_buffer_usage/README.html
- https://gpuopen.com/vulkan-memory-allocator/
- https://github.com/Tencent/ncnn/blob/master/src/allocator.cpp

## General architecture work recommended

1. Decouple cached immutable Vulkan pipelines from binding-specific
   Algorithms. Use an efficient **per-command/graph descriptor arena**
   or descriptor update mechanism, reusing sets *after GPU completion*,
   rather than a small global FIFO keyed by every transient VkBuffer.
   Do not introduce unsafe descriptor update-after-bind behavior or
   keep arbitrary temporary tensors alive indefinitely.

2. Promote correct **GPU-resident deferred operation execution** from an
   opt-in experiment to a supported default only after validating host
   observation, in-place aliasing, autograd, CPU fallback boundaries and
   all other PyTorch device contracts. Recycle command pools and buffers
   at completed submission/fence boundaries.

3. Introduce a lifetime-aware temporary-storage arena or device
   suballocator, not a hardcoded set of tensor shapes. This may reduce
   memory and descriptor pressure, but live GPU readers/writers must
   never alias reused buffers; avoid gratuitous zeroing of fully
   overwritten scratch storage.

4. Implement native direct-to-output paths for commonly composed ops
   rather than always materializing functional intermediates and
   copying them. Preserve all output alias, stride, dtype semantics.

5. Improve synchronization by tracking actual RAW/WAR/WAW resource
   hazards, not simply disabling barriers. Device-independent tests
   should run under Vulkan validation layers where available.

6. Continue algorithmic convolution backward and attention performance
   work. Host control-plane fixes alone do not erase those shader
   performance gaps versus CUDA/cuDNN or ROCm/MIOpen.

Instrumentation: TORCH_VULKAN_PROFILE_HOST_HOTPATH=1 exposes
torch_vulkan.runtime_hotpath_stats(). The algorithm-cache-capacity
override is a bounded diagnostic only, **not** a change to the
default 512. No optimizer math or shader algorithm changed.

Raw measurements and run-level metadata:
benchmarks/results/runtime-host-control-plane-2026-10-20/{summary.json,raw-profiles.jsonl.gz}.
The private source of the Lab02 model is not published.
