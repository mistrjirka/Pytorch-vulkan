# CUDA and llama.cpp Vulkan universality audit

> **Historical audit:** this document records the implementation at the initial review. Its tensor-address pipeline-cache, scalar-only convolution, and rejected-layout findings have since been addressed. The current candidate registry, tile reuse, optional tuning and remaining coverage limits are documented in [General convolution candidate dispatch](convolution-candidate-dispatch.md). Recommendations and source references below are retained as the original audit trail.

**Date:** 2026-10-18
**Project:** [mistrjirka/Pytorch-vulkan](https://github.com/mistrjirka/Pytorch-vulkan)
**Sources reviewed:** PyTorch CUDA convolution (snapshot `9664c41ccfc87a9661cb2b177985131948306876`), llama.cpp Vulkan (`5aeac37a35b44fde4bdffaec9159f2fcb253b622`), and our current native `PrivateUse1` implementation.

## Answer: what happens when 7×7 becomes 5×5?

**The same generic operation still works.** A kernel-size change alters convolution parameters and the amount of work; it does not alter the operator's mathematical contract. A fast backend selects a different algorithm, tile size, pipeline specialization, or layout when justified. Other operations in the network continue to dispatch independently.

**Universality means explicit supported semantics plus correct fallback. It does not mean that one shader is optimal for every shape.**

Our generic Vulkan FP32 NCHW convolution supports differing `KH`/`KW`, stride, padding, dilation and groups, with forward and dInput/dWeight/dBias. It is not yet a universal PyTorch training backend: FP16/BF16, non-contiguous/channel-last layouts, transposed convolution, symbolic dimensions, and some other operator families are unsupported.

## PyTorch CUDA: concrete dispatch layers

1. **ATen operator normalization and backend selection.** [`Convolution.cpp` lines 1297–1410](https://github.com/pytorch/pytorch/blob/9664c41ccfc87a9661cb2b177985131948306876/aten/src/ATen/native/Convolution.cpp#L1297-L1410) selects among cuDNN, CUDA depthwise, slow CUDA convolution, dilated/transposed fallbacks and other backends using device, convolution parameters, dtype, memory format, groups and backward requirements.
2. **cuDNN engine/plan selection.** [`Conv_v8.cpp` around lines 1152–1300](https://github.com/pytorch/pytorch/blob/9664c41ccfc87a9661cb2b177985131948306876/aten/src/ATen/native/cudnn/Conv_v8.cpp#L1152-L1300) first looks up an engine plan cache, then uses heuristic candidates (`benchmark=False`) or search and timing (`benchmark=True`), with fallback candidates, workspace constraints and determinism/precision filtering. CUDA PyTorch may use different plans for forward, backward-data and backward-filter.
3. **Cache by operation configuration, not allocation identity.** [`Conv_v8.cpp` cache keys and bounded LRU](https://github.com/pytorch/pytorch/blob/9664c41ccfc87a9661cb2b177985131948306876/aten/src/ATen/native/cudnn/Conv_v8.cpp#L167-L350) include parameter geometry, operation, alignment and precision requirements. Reusing the *execution plan* is independent of binding the current input/weight/output data.
4. **Conservative fallback.** [`ConvolutionMM2d.cu`](https://github.com/pytorch/pytorch/blob/9664c41ccfc87a9661cb2b177985131948306876/aten/src/ATen/native/cuda/ConvolutionMM2d.cu) demonstrates an im2col + GEMM route for supported 2D cases; dilation, depthwise, and transposed configurations may select other implementations. Do not assume that this single legacy kernel handles every convolution.

Crucially, training has **distinct backward algorithms** for dInput and dWeight. The backend must also respect the autograd output mask, and compute dBias (often a reduction) correctly.

## llama.cpp: Vulkan implementation details worth adapting

| Area | Verified upstream implementation | Adaptation for PyTorch Vulkan |
|---|---|---|
| Tile selection | [`ggml_vk_conv_select_shape`](https://github.com/ggml-org/llama.cpp/blob/5aeac37a35b44fde4bdffaec9159f2fcb253b622/ggml/src/ggml-vulkan/ggml-vulkan.cpp#L8655-L8688) chooses 64×32, 64×128, 32×256 or 128×128 based on output channels, spatial workload and hardware | Select algorithm separately per operation, shape and device; use a safe generic fallback |
| Pipeline identity | [`ggml_vk_op_get_pipeline`](https://github.com/ggml-org/llama.cpp/blob/5aeac37a35b44fde4bdffaec9159f2fcb253b622/ggml/src/ggml-vulkan/ggml-vulkan.cpp#L9277-L9335) keys the convolution pipeline by KW/KH, stride, padding, dilation, alignment, dtype and tile—not tensor pointer | Cache reusable compiled pipelines by shader/specialization/hardware, separate from live tensor descriptors |
| Shader specialization | [`conv2d_mm.comp`](https://github.com/ggml-org/llama.cpp/blob/5aeac37a35b44fde4bdffaec9159f2fcb253b622/ggml/src/ggml-vulkan/vulkan-shaders/conv2d_mm.comp#L68-L95) has independent specialization constants for KW, KH, dilation, padding, stride, tile dimensions, and alignment | Leave the source generic, specialize on meaningful parameters when profitable; benchmark compilation and cache hit rate |
| GPU capabilities | [`ggml_vk_load_shaders`](https://github.com/ggml-org/llama.cpp/blob/5aeac37a35b44fde4bdffaec9159f2fcb253b622/ggml/src/ggml-vulkan/ggml-vulkan.cpp#L1674) and pipeline setup consider NV/KHR cooperative matrices, vendor-dependent subgroup and shared-memory limits | Query actual Vulkan device features rather than assuming NVIDIA/AMD/Intel have identical limits |
| Descriptor pooling | [`ggml_pipeline_allocate_descriptor_sets`](https://github.com/ggml-org/llama.cpp/blob/5aeac37a35b44fde4bdffaec9159f2fcb253b622/ggml/src/ggml-vulkan/ggml-vulkan.cpp#L850-L891) reuses descriptor pools and expands capacity | Do not recreate descriptor layouts/pools for every new activation; **do not recycle in-flight descriptors** |
| Fusibility | [`ggml_vk_can_fuse`](https://github.com/ggml-org/llama.cpp/blob/5aeac37a35b44fde4bdffaec9159f2fcb253b622/ggml/src/ggml-vulkan/ggml-vulkan.cpp#L13689) checks graph dependencies, types, shapes, strides and alignment before mul-mat + add fusion | Fuse only when PyTorch aliasing, broadcasting, autograd, and storage-version semantics are preserved |
| Explicit limits | [`ggml_backend_vk_device_supports_op`](https://github.com/ggml-org/llama.cpp/blob/5aeac37a35b44fde4bdffaec9159f2fcb253b622/ggml/src/ggml-vulkan/ggml-vulkan.cpp#L15426) checks dtype/layout, buffer limits and supported operations | Have a truthful `supports(config, device)` predicate; unsupported must correctly fall back or raise |

**Do not transplant the GGML inference scheduler into training.** PyTorch autograd has saved intermediate tensors, mutation/version counters, concurrent execution and gradient dependencies that need explicit lifetime protection. Port the *separation of responsibilities*, not the inference-only assumptions.

## Antipatterns and correctness risks in our implementation

- `VulkanContext::make_cache_key` includes each tensor's `rawData()` address. Therefore a fresh activation can trigger pipeline creation even with identical dimensions. **Pipeline, specialization and descriptor lifetimes must be separated.**
- The existing `Kompute::Algorithm` combines shader pipeline and bound descriptors. Rebinding a pooled descriptor while an older command still uses it would introduce a silent data hazard. Use fence/timeline protected in-flight descriptor arenas.
- Our generic Conv2D currently dispatches a scalar direct kernel per output element, a poor competitive baseline for large channels/filters. It requires contiguous, zero-offset FP32 NCHW buffers and static dimensions. Dedicated depthwise and tiled implicit-GEMM paths are needed; all configurations still require a correctness fallback.
- Experimental GPU-residency and deferred submission have limited coherence tracking and are **not production stream semantics**. Host reads, in-place writes, CPU fallback, saved tensors and overlapping views must agree with Vulkan memory barriers and PyTorch mutation/version tracking.
- Avoid compiling pipeline variants by *input data pointer*. Include specialization constants in any pipeline key if they change compilation, but avoid recompiling a shader just because the tensor storage address changed.
- Do **not** claim `torch.compile` dynamic-shape support while native code depends on `SymInt::expect_int()`. Support guard/compilation specialization or reject symbolic dimensions explicitly.
- Workgroup bounds and buffer-size limits should come from actual queried Vulkan limits; fixed 65535 and 32-bit indexing are not universal.

## Recommended implementation order

1. **Introduce a capability matrix** for operator + dtype + layout/stride + shape + device, and keep the current scalar forward/backward path as the FP32 NCHW reference. Write user-visible errors for unsupported cases rather than unsafe casts.
2. **Decouple pipelines from descriptors** and reuse pipeline layouts/shader modules. Track buffer/descriptor resource ownership until the last in-flight command completes. Cache by operation, shader specialization, device and flags; bound memory with an eviction policy.
3. **Add a convolution algorithm registry:** `DirectGeneric`, `Depthwise`, `PointwiseGemm`, `TiledImplicitGemm`; implement and validate forward, backward-data and backward-filter separately. Choose by channel counts, batch, kernel, layout and hardware. Guard fast paths.
4. **Profile and then autotune, optionally.** Use Vulkan GPU timestamps (not CPU wall time alone) to compare shader work; use CUDA only as a hardware performance reference, not as an implementation oracle. Record compilation/warmup separately and cache winners by a stable configuration key.
5. **Continue reducing graph boundaries** for eager training without violating autograd semantics. CPU fallback and synchronization are not free, but avoiding a fence is worthwhile only after checking resource lifetime and visibility.

## Additional numerical verification performed

New `tests/test_conv_universality.py` exercises:
- Non-square 3×5 convolution, asymmetric stride/padding and dilation.
- 7×7 convolution with padding.
- Depthwise groups=4 with channel multiplier 3.
- Grouped 5×1 convolution with non-unit stride.
- Explicit rejection of a non-contiguous tensor layout.

**All five tests passed** on the RTX 2080 Ti, Tesla V100 and Lavapipe, including CPU reference comparisons for forward and dInput/dWeight/dBias gradients where applicable.

## Independent agent review

Two read-only OpenCode v2.0.32 sessions were launched using `opencode/deepseek-v4.1-flash-free`, against isolated copies of **public code**. The CUDA-focused source inspection completed, identifying the actual cuDNN plan selection/cache strategy and the explicit limitations of our `convolution.cpp`. A llama-focused source inspection timed out while navigating the large codebase; a separate short independent Vulkan design critique completed.

The model-generated observations were cross-checked against the code before being included. In particular, **do not accept claims that a cache key contains stale push constants or that a Vulkan memory barrier can replace a required lifetime fence without verification.**

## External skill audit

Skill files examined, with no blind vendoring:

- [Babysitter: vulkan-compute](https://github.com/a5c-ai/babysitter/blob/main/library/specializations/gpu-programming/skills/vulkan-compute/SKILL.md): useful as a quick GLSL/SPIR-V overview; **examples are incomplete**, hardcoded workgroups/descriptor limits must be checked on actual devices, and legacy barrier recipes are not sufficient for stream-aware PyTorch.
- [Babysitter: gpu-benchmarking](https://github.com/a5c-ai/babysitter/blob/main/library/specializations/gpu-programming/skills/gpu-benchmarking/SKILL.md): useful workflow, but its illustrated **peak FLOPS formula based on maximum resident threads is not generally valid**. Use architecture-specific hardware throughput and distinguish shader GPU time from CPU submit time.
- [Babysitter: nsight-profiler](https://github.com/a5c-ai/babysitter/blob/main/library/specializations/gpu-programming/skills/nsight-profiler/SKILL.md): CUDA/NVIDIA tool guidance, not automatically portable to AMD/Intel Vulkan.
- [CppStudio: vulkan-compute-sync](https://github.com/tarkansarim/CppStudio/blob/main/skills/cpp-cuda-vulkan-studio/modules/vulkan-compute-sync/GUIDE.md): highest relevance to command buffers, barriers, descriptor lifetime and in-flight memory; workstation-specific path claims should not be copied.
- [CppStudio: gpu-profiling-workstation](https://github.com/tarkansarim/CppStudio/blob/main/skills/cpp-cuda-vulkan-studio/modules/gpu-profiling-workstation/GUIDE.md): helpful matched-workload discipline; its machine inventory is for another host, not our Sandbox.

**Use authoritative Khronos specifications and Vulkan validation on the running driver as final arbiters.** Treat skills and independent model reviews as checklists, never substitutes for numerical tests and profiled evidence.
