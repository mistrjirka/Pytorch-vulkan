---
name: pytorch-vulkan-performance
description: Audit or optimize the PyTorch PrivateUse1 Vulkan backend: universal convolution/matmul dispatch, GPU residency, autograd, numerical correctness, synchronization and CUDA comparisons.
---

# PyTorch Vulkan performance engineering

Use this skill for backend changes, benchmarking, kernel specialization, and cross-vendor Vulkan correctness in this repository. Read `docs/vulkan-dispatch-universality-audit.md` and `docs/performance-investigation-2026-10-18.md` first. Source code and test results are more authoritative than this skill.

## Required workflow

1. **Define operator semantics.** Specify shape, dtype, stride/layout, device limits, grouping/dilation/padding, and whether forward/dInput/dWeight/dBias are required. Distinguish a supported-but-slow configuration from one that is unsupported.
2. **Read the dispatcher.** Find the actual ATen/PrivateUse1 registration, kernel and host-side memory path. Never infer that a CUDA or llama.cpp inference kernel automatically supports PyTorch autograd.
3. **Verify numerical parity before speed claims.** Compare forward and gradients with same-seed CPU and, when available, CUDA references. Test 1x1, 3x5, 5x5, 7x7, grouped/depthwise, asymmetric stride/padding/dilation, layout rejection and nonzero storage offsets.
4. **Measure matched workloads.** Same GPU, FP32, batch/shape, model architecture and warmup; CUDA must synchronize at timing boundaries. Separate CPU wall time, Vulkan GPU timestamp, upload, download, pipeline compilation, fence wait and operator fallback. Report medians/ranges and whether optimizer is included.
5. **Choose an implementation by capability.** Query Vulkan device features and limits. Retain a generic correctness path; specialize or fuse only behind explicit validated guards.
6. **Validate memory and command lifetime.** A pipeline can be reused but descriptor bindings, staging buffers, in-flight command buffers and saved tensors cannot be recycled until their GPU users finish. Respect PyTorch mutation and aliasing rules.
7. **Test on available devices** including NVIDIA RTX 2080 Ti, Tesla V100 and Lavapipe. AMD/Intel behavior is **unverified** until executed there. Treat test warnings as bugs to track, not proof of correctness.

## Do not repeat these antipatterns

- Cache whole compute pipelines by tensor `rawData()` address. Separate the pipeline/specialization cache from mutable descriptor bindings and keep caches bounded.
- Create one new Vulkan fence, descriptor set, command buffer, pipeline or staging allocation per trivial operation without measuring its cost.
- Treat `vkCmdPipelineBarrier` as CPU completion; GPU visibility barriers, CPU waits, queue dependencies, and buffer lifetimes are different contracts.
- Eliminate an upload/download without tracking whether the CPU or GPU has the newest data. Never read a GPU-dirty mapped staging pointer as if it were the device-local result.
- Assume any kernel or specialization is universal because it passed one 3x3 case; preserve generic fallback or explicit rejection.
- Use a simplistic `maxThreadsPerSM * SMs * clock` formula to claim peak FP32 FLOPS. Use official architecture rates and measured kernel time.
- Copy benchmark scripts' `ncu`/Nsight, driver, `vulkaninfo`, or `VK_ICD_FILENAMES` paths from another workstation without verifying installed tools.
- Report llama.cpp's graph scheduling or fusion as safe for PyTorch training unless all saved-tensor, aliasing and autograd dependencies have been proven.

## Existing checks

```bash
python -m pytest -q tests
python benchmarks/profile_training_phases.py --backend vulkan \
    --optimizer vulkan_fused --zoo ../pytorch-XDNA2
python benchmarks/probe_matmul_gpu_time.py --backend vulkan
```

For diagnostic comparison, the non-default flags `TORCH_VULKAN_GPU_RESIDENT=1`, `TORCH_VULKAN_DEFER_SUBMISSIONS=1`, `TORCH_VULKAN_MM_KERNEL=register` and `TORCH_VULKAN_FUSE_ADDMM=0` are **experimental controls**, not universal recommendations.

Review authoritative upstream code: [PyTorch Convolution.cpp](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/Convolution.cpp), [cuDNN v8](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cudnn/Conv_v8.cpp), [llama.cpp ggml Vulkan](https://github.com/ggml-org/llama.cpp/blob/master/ggml/src/ggml-vulkan/ggml-vulkan.cpp) and the [Khronos Vulkan Guide](https://docs.vulkan.org/guide/latest/).
