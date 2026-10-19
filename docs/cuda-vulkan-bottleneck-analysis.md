> Historical measurements: the small/eager log-softmax CPU crossover described below has been removed. Current FP32 eager/deferred execution and audited fallback behavior are documented in [the pointwise and log-softmax report](pointwise-optimization-2026-10-19.md).

# CUDA versus Vulkan: bottleneck investigation and optimization results

**18 October 2026.** PyTorch 2.14.1, actual RTX 2080 Ti, with additional Vulkan validation on Tesla V100. See also [the earlier investigation](performance-investigation-2026-10-18.md).

## What was slow

The original backend sent every tensor to the GPU, waited for the result and copied it back after almost every ATen operation. Unsupported operations crossed into CPU execution, requiring still more copies. These costs dominated the original MLP training step (~956 ms with standard AdamW).

A matched NVTX-marked CUDA/Vulkan Nsight Systems experiment on the **same RTX 2080 Ti** showed that warmed Vulkan forward, backward and optimizer stages performed approximately **10, 11 and 1 queue submissions**. Pipeline creation was **not** occurring in those steady-state stages, so a pipeline cache redesign should not be confused with eliminating the main current MLP bottleneck.

| Instrumented MLP stage | CUDA wall time | Vulkan wall time before this round | Vulkan `vkWaitForFences` API time |
|---|---:|---:|---:|
| Forward | 0.313 ms | 14.3 ms | 3.56 ms |
| Backward | 1.27 ms | 21.8 ms | 5.83 ms |
| Optimizer | 1.47 ms | 10.7 ms | 9.33 ms |

Vulkan fence-wait durations include GPU work, so **do not simply add the fence-wait times as CPU overhead**. CUDA kernel busy union from that matched trace was ~0.127 ms in forward, ~0.182 ms backward and ~0.333 ms optimizer. The Nsight build did **not** record corresponding Vulkan GPU-work intervals during measured steps; zero GPU time from that report would be invalid. The raw analysis explicitly marks this GPU timing as unavailable.

## Actual GPU shader timing versus CUDA

Dedicated Vulkan timestamp queries separate transfer, shader and readback phases. CUDA events measure the CUDA GEMM kernels. Instrumented results on the same RTX 2080 Ti, FP32 with CUDA TF32 disabled:

| GEMM dimensions (M×K×N) | CUDA kernel | Vulkan tiled shader | Vulkan upload + readback |
|---|---:|---:|---:|
| 128×784×1024 | 0.049 ms | 0.171 ms | 0.444 ms |
| 128×1024×1024 | 0.059 ms | 0.226 ms | 0.528 ms |
| 1024×1024×1024 | 0.237 ms | 1.586 ms | 1.930 ms |

Vulkan kernel execution is itself several times slower than CUDA GEMM, while the unconditional upload/download and fence path increases operator latency further. A stride-aware 64×64 register-tiled shader lowers the RTX 2080 Ti 1024³ Vulkan shader to **0.503 ms**, but can regress on small shapes and the V100; the selection is shape-sensitive and can be overridden by `TORCH_VULKAN_MM_KERNEL=tiled|register`.

Using host-visible Vulkan storage avoids staging copies but may cause the GPU to read system RAM over PCIe: for 1024³ the Vulkan shader slowed to ~4.33 ms. This is **not a general substitute** for GPU-resident device-local storage.

## Implemented fixes

| Change | Measured result |
|---|---|
| Opt-in fused Vulkan AdamW, batched across parameter tensors | MLP optimizer 861 ms → ~10 ms |
| Opt-in GPU-resident dirty-bit tracking | MLP AdamW step ~10 ms → ~1.1 ms |
| Native GEMM+bias for `torch.nn.Linear` | MLP training 33.94 → 25.24 ms; inference 7.68 → 4.59 ms |
| Native FP32 single-axis `sum` / bias-gradient reduction | MLP training 25.19 → 17.29 ms; backward 17.18 → 10.12 ms |
| Native row-wise FP32 log-softmax + backward for larger tensors | 256×4096 forward 22.34 → 0.824 ms, backward 41.99 → 0.928 ms |
| Proper backward kernels for ReLU, GELU, LayerNorm and convolution | Eliminated multiple expensive CPU fallback paths |

**The log-softmax crossover is important:** for only 128×10 values, CPU fallback is about 0.049 ms forward / 0.138 ms backward, whereas a Vulkan kernel takes ~0.552 / 0.762 ms. The default policy therefore uses native Vulkan for contiguous last-axis FP32 tensors with **at least 32,768 elements**, while smaller logits retain the cheaper CPU route. `TORCH_VULKAN_NATIVE_LOGSOFTMAX=0|1` overrides this for benchmarking.

## Final same-GPU model benchmark

Actual architecture: [pytorch-XDNA2 model-zoo MLP](https://github.com/mistrjirka/pytorch-XDNA2) (1.87M parameters), batch 128×784, FP32. Warmed end-to-end step, **forward + loss + backward + AdamW**; CUDA uses PyTorch AdamW (`foreach=False`), Vulkan uses mathematically equivalent opt-in `torch_vulkan.optim.AdamW`. Inputs and initial model weights are synthetic and seeded.

| Backend and hardware | Inference median | Training median |
|---|---:|---:|
| Native CUDA, RTX 2080 Ti | **0.172 ms** | **2.218 ms** |
| Optimized Vulkan, RTX 2080 Ti | **4.614 ms** | **17.166 ms** |
| Optimized Vulkan, Tesla V100 | **1.623 ms** | **6.500 ms** |

On the **same RTX 2080 Ti**, Vulkan remains ~27× slower for inference and ~7.7× slower for training. The V100 numbers are not a same-card CUDA comparison: current PyTorch 2.14.1 CUDA 13.0 builds do not support V100 SM70.

These results require `TORCH_VULKAN_GPU_RESIDENT=1` and the fused optimizer. The default PyTorch `torch.optim.AdamW` is **not automatically replaced**, and GPU-resident tracking remains **experimental and opt-in**. The full initial-to-current training reduction (~956 → 17.166 ms) is about 56×, but includes replacing CPU-heavy AdamW with the new fused implementation.

## Remaining bottlenecks

1. **Transformer attention** is still running through CPU fallback because the earlier Vulkan SDPA implementation was numerically incorrect and had no valid backward. A small real model-zoo transformer still takes ~1.35 s per Vulkan training step versus ~13 ms on CUDA; the log-softmax kernel improves a piece of it but cannot fix the missing attention operator.
2. **CNNs** use correctness-first scalar convolution kernels. U-Net, StyleNet and MobileNetV3 need tiled/implicit-GEMM convolution and native backward optimizations.
3. **Per-operation Vulkan fences and command preparation** remain expensive. Deferred execution can reduce 140 GPU-only operations to one submission, but is not faster for entire models yet: temporary buffers cannot be reused during outstanding work, causing many new cached-algorithm misses.
4. **Pipeline caching by buffer pointer** remains a cold-start/dynamic-shape concern. A future runtime should cache pipelines independently of buffer descriptors, allocate descriptor sets from a pool, reuse command buffers and implement genuine in-flight tensor-lifetime tracking.
5. **Coverage and portability:** FP16/BF16 and optimized cooperative-matrix paths are still missing; only NVIDIA RTX 2080 Ti, V100, and software Lavapipe have been tested here.

## How to reproduce

```bash
# In the Pytorch-vulkan checkout, with matching PyTorch and the XDNA2 repo alongside:
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export TORCH_VULKAN_PHYSICAL_DEVICE_INDEX=0
export TORCH_VULKAN_GPU_RESIDENT=1
.venv/bin/python benchmarks/compare_cuda_vulkan.py --backend vulkan --model mlp --scale small --mode both --optimizer vulkan_fused --zoo ../pytorch-XDNA2
# Matched CUDA comparison:
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 .venv-cuda/bin/python benchmarks/compare_cuda_vulkan.py --backend cuda --model mlp --scale small --mode both --zoo ../pytorch-XDNA2
```

All new operators were tested on NVIDIA RTX 2080 Ti and Tesla V100 with **83/83 tests passing** for default and GPU-resident modes. Lavapipe passed **82** tests, skipping the physical-GPU-only test. These are regression tests, not evidence of production-complete PyTorch coverage.

Raw sample timings: [`benchmarks/results/gpu_comparison/`](../benchmarks/results/gpu_comparison/) and [`benchmarks/results/fused_addmm/`](../benchmarks/results/fused_addmm/). The reproducible scripts live under [`benchmarks/`](../benchmarks/). Nsight .nsys-rep files remain local to the Sandbox and are not committed.
