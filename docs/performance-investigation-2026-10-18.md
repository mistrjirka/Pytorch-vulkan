# Vulkan backend performance investigation — 18 October 2026

This document records the **measured** performance findings on the RTX 2080 Ti and the architecture changes made after `1903130`. The original training benchmarks are in [GPU_COMPARISON.md](../benchmarks/GPU_COMPARISON.md). The numerical tests were executed against **PyTorch 2.14.1+cpu** (Vulkan), **PyTorch 2.14.1+cu130** (CUDA), and an XDNA2 model-zoo checkout.

## Executive summary

The largest deficit was not a slow GEMM kernel. It was an execution design that performed hundreds of host operations and device submissions per step. The original MLP training iteration on the RTX 2080 Ti took ~956 ms, compared with ~2.27 ms on CUDA. Recent independent MLP phase-run samples using the new opt-in fused AdamW and native LayerNorm backward measure **~50.7–51.6 ms**. This is approximately **19× faster** than the previous Vulkan baseline but remains **~22× slower** than native CUDA.

The speedups are real but this remains an experimental backend. The default `torch.optim.AdamW` is **not** replaced; the new implementation is explicitly selected with `torch_vulkan.optim.AdamW`.

## Detailed causal evidence

### CPU-side PyTorch profiler: original bottleneck

One MLP training step incurred ~273 `aten::_to_copy` calls occupying ~711 ms self CPU time, approximately 77% of the traced step. The ordinary AdamW optimizer incurred ~850–869 ms while GEMM calls occupied ~19 ms (five calls). The fallback path performed repeated CPU tensor transfers for `mul`, `lerp`, `sqrt`, `addcdiv`, and `addcmul`.

### Nsight Systems Vulkan timeline

In a captured fused-AdamW experiment (inclusive of initialization and warmup), Nsight recorded:

| Vulkan API | Calls | Total API time |
|---|---:|---:|
| `vkWaitForFences` | 162 | 155 ms |
| `vkCreateFence` + `vkDestroyFence` | 324 | 83 ms |
| `vkAllocateMemory` | 192 | 87 ms |
| `vkFreeMemory` | 192 | 63 ms |
| `vkCreateComputePipelines` | 90 | 50 ms |
| `vkQueueSubmit` | 162 | 2.6 ms |

The 254 GPU-work intervals, after merging overlap, accounted for ~23 ms of an approximately 239 ms traced timeline window. **This is only a trace-window estimate and not an OS-level GPU utilization figure.**

A later trace with native buffer reuse showed 120 allocations and 104 pipeline creations but covers a **different set of operations, durations, startup activity, and warmup**. The counts should **not** be directly treated as a controlled before/after comparison or claim of absolute overall improvement; use phase benchmarks for speed comparisons.

### Structural reasons

1. **Pipeline identity is incorrectly coupled to buffer addresses.** `VulkanContext::get_or_create_algorithm` caches whole Kompute `Algorithm` objects keyed by shader name and each bound buffer's host pointer. Fresh activations trigger full pipeline + descriptor layout + descriptor pool creation even when the shader and pipeline specialization are unchanged. A long deferred add/ReLU chain saw 560 cache misses across four trials despite only two distinct compute shaders.
2. **Every eager operation creates/submits a Kompute `Sequence` and blocks on a fence.** GPU workload becomes fragmented into many small submissions.
3. **`kp::Tensor::eDevice` holds device-local memory plus a mapped staging buffer.** `OpTensorSyncDevice` uploads input buffers and `OpTensorSyncLocal` downloads outputs for every eager op, even when all consumers are GPU kernels.
4. **Unsupported ATen ops fall back to the CPU, requiring mapped tensor copies.** Some previously handwritten kernels (especially embedding, softmax and LayerNorm affine parameters) also redundantly created temporary Vulkan buffers.
5. **Convolution kernels are scalar correctness implementations, not competitive optimized kernels.** Their compute cost will dominate after the control plane is improved.

## Optimizations implemented and tested

| Optimization | Measured impact |
|---|---|
| Fused GPU-native AdamW (one kernel per parameter) | Optimizer ~861 → 16 ms |
| Fused AdamW updates batched in one submission | Optimizer ~15.7 → 10.3 ms |
| Reuse Vulkan buffers in embedding/softmax/LayerNorm affine | Embedding inference ~162 → 12 ms, full AdamW training ~3,120 → 62.6 ms |
| Native FP32 LayerNorm backward | MLP backward ~32 → 25 ms |
| Opt-in direct mapped `kp::Tensor::eHost` memory | Training ~68.5 ms (`eDevice`) vs ~67.1 ms (20 MB mapped) — little benefit by itself |
| Opt-in deferred Vulkan command recording | 140-op GPU-only add/ReLU chain: 73.7 ms eager `eDevice`, 71.3 ms eager host-visible, 48.0 ms deferred host-visible |

Full model performance (small XDNA2 model-zoo cases, RTX 2080 Ti, FP32, complete forward+backward+AdamW):

| Network | Prior Vulkan | New fused-optimizer Vulkan | Speedup |
|---|---:|---:|---:|
| MLP | 944 ms | 53.6 ms | 17.6× |
| U-Net | 436 ms | 134 ms | 3.2× |
| StyleNet | 1,516 ms | 173 ms | 8.8× |
| Embedding network | 3,120 ms | 62.6 ms | 49.8× |

These are separate benchmark runs, not guaranteed deterministic measurements. The newest native LayerNorm/softmax changes further reduced phase-profile MLP iteration time to ~51 ms. Current CUDA MLP whole step is ~2.17–2.27 ms.

## Deferred execution experiment is opt-in

`TORCH_VULKAN_DEFER_SUBMISSIONS=1` records selected native ops into a single command buffer. It is only applicable to entirely host-visible bindings using `TORCH_VULKAN_HOST_VISIBLE_MAX_BYTES`; operations using device-local buffers and CPU-fallback boundaries flush and synchronize. The allocator defers reuse of buffers until the queued work has finished. The PyTorch benchmark scripts explicitly synchronize Vulkan at measurement boundaries.

The add/ReLU chain saw **1 Vulkan submission for 140 ops** under deferred mode, with about **1.8 ms final GPU submission/wait** in warmed runs, but **560 pipeline-cache misses** across four trials because queued outputs cannot reuse buffers. The whole MLP remains about **53.5 ms eager versus 55.6 ms deferred** in one controlled host-visible run, so **do not enable deferral by default** yet.

The configured mapped-memory size is not a generic device allocation budget: large host-visible buffers accessed over PCIe can degrade throughput.

## Next architecture (highest priority)

**First: decouple pipelines from descriptors.** Build one Vulkan compute pipeline per shader, descriptor layout and specialization constants (including relevant workgroup configuration only when shader specialization requires it). Cache the pipeline independently of which tensor buffers are currently bound. Allocate/update descriptor sets from a pooled descriptor arena. Use push constants for dispatch-specific dimensions and strides.

**Second: persistent device-resident storage with explicit coherence.** Each tensor allocation must track `HOST_DIRTY`, `GPU_DIRTY` and synchronization dependency state, ideally generation/version counters. Host-to-device copy should occur only after host modification, and device-to-host only before an actual host read, unsupported CPU fallback, serialization, or user-requested synchronization. Avoid implicit `data_ptr()` reads as synchronization surrogates.

**Third: operation-level command batching with real lifetime management.** Reuse/reset command pools and command buffers, queue dispatches, insert barriers for read-after-write/write-after-write hazards, defer fences until graph boundaries, and pin tensor allocations and descriptors until in-flight commands have completed. Separate host-visible staging buffers from device-local GPU memory. Use per-stream GPU events/timeline semaphores once stream support is introduced. Current deferred command prototype is a feasibility test, **not production stream semantics**.

**Fourth: GPU-native fallback replacements**, prioritized by profiler evidence: softmax/backward, reductions, normalization backward, optimizers, attention and common broadcast/view ops. Avoid backward breaks from `AutogradPrivateUse1`; include forward + gradient numerical checks.

**Fifth: only then optimize execution kernels** using cooperative matrices/tensor core paths, tiled GEMM, optimized backward convolution and fused attention. This should be accompanied by matching CUDA/Vulkan kernel-level GPU traces and larger-batch benchmarks to separate shader performance from driver/dispatch costs.

### Verification criteria

- Run the existing pytest suite on real RTX 2080 Ti, Tesla V100, and software Lavapipe after **every architectural edit**.
- Test same-weight forward/backward/AdamW correctness, model-zoo training loss, no stale-read and aliasing cases, buffer lifetime under large deferred chains, and repeated updates.
- Compare steady-state median, p95, first-step JIT/pipeline cost, pipeline creation count, bytes transferred, GPU busy intervals, and fences/submissions per training step.
- Keep explicit Vulkan performance experiments opt-in until all relevant correctness tests pass; preserve the default PyTorch 2.14.1 behavior.

### Tools and reproduction

- `benchmarks/profile_training_phases.py --backend vulkan --optimizer vulkan_fused --zoo ../pytorch-XDNA2-vulkan-zoo-reference`
- `benchmarks/probe_submission_costs.py --steps 70 --elements 4096 --repeats 4`
- `benchmarks/compare_cuda_vulkan.py --backend vulkan --optimizer vulkan_fused --model mlp --scale small --zoo ../pytorch-XDNA2-vulkan-zoo-reference`
- `nsys profile --trace=vulkan --vulkan-gpu-workload=individual --sample=none --cpuctxsw=none -o /tmp/report python ...`

Raw Nsight captures are preserved locally in the Development Sandbox under `benchmarks/results/nsys/`, not published by default because they may contain container-local execution metadata.
