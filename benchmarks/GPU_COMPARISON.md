# Vulkan versus native CUDA and CPU: PyTorch 2.14.1

Measured 2026-10-18 in the persistent Development Sandbox on the **same RTX 2080 Ti**.
Networks are actual architectures imported from the upstream XDNA2 model zoo; model weights and inputs are synthetic.

## Hardware and timing protocol

- NVIDIA **GeForce RTX 2080 Ti**, 22 GiB, NVIDIA driver 580.178.04. `CUDA_DEVICE_ORDER=PCI_BUS_ID` and `CUDA_VISIBLE_DEVICES=0` ensure CUDA selects the same physical device as Vulkan index 0.
- CPU: **AMD EPYC 7352, 24 cores / 48 threads**, with **8 PyTorch CPU threads** used for benchmarks.
- PyTorch **2.14.1+cpu** for Vulkan extension and **2.14.1+cu130** for CUDA and CPU, torchvision 0.29.1. CUDA **13.0**, TF32 disabled for CUDA GEMM and cuDNN.
- **FP32** throughout; synthetic data and random initial model weights. Architectures remain unchanged; only input batch/spatial sizes are reduced.
- Timings are **end-to-end Python wall-clock medians**, including CPU fallback, copies, and Python overhead within the forward/training step. Model construction, initial transfer to device, and input creation are excluded. CUDA is explicitly synchronized around each timed step.
- Training = forward + loss + backward + **AdamW (`foreach=False`)**. This is not only forward+backward. Each backend initializes models deterministically with the same seed; some models have stochastic dropout in training, which may differ by device.
- Small tests: first Vulkan batch used **2 warmups, 4 samples**; CUDA/CPU used **3 warmups, 5 samples**. MobileNetV3/Transformer Vulkan retests used 3/5. Larger-image inference tests used **2 warmups and 4 samples** for both GPU APIs. Results are indicative, not controlled statistical performance certification.
- Upstream model zoo: [mistrjirka/pytorch-XDNA2](https://github.com/mistrjirka/pytorch-XDNA2), commit `c7c063101ec5209caf8022f5b4ea6948d1a8d6ed`.

## Small inputs: inference

| Network | Input shape | CPU ms | CUDA ms | Vulkan ms | Vulkan / CUDA slowdown |
|---|---|---:|---:|---:|---:|
| MLP | 128×784 | 1.342 | 0.169 | 15.581 | 92.2× |
| U-Net | 1×3×32×32 | 2.977 | 1.072 | 27.644 | 25.8× |
| StyleNet (5 blocks) | 1×3×16×16 | 3.201 | 1.639 | 48.981 | 29.9× |
| ResNet18 | 2×3×48×48 | 7.368 | 2.092 | 64.082 | 30.6× |
| MobileNetV3-small | 2×3×64×64 | 7.923 | 3.833 | 102.561 | 26.8× |
| Embedding network | 32×64 | 0.132 | 0.136 | 161.554 | 1,187.9× |
| Transformer (4 layers) | 2×128 | 9.044 | 1.177 | 416.815 | 354.1× |
| VGG11 | 1×3×64×64 | 23.983 | 1.855 | 93.157 | 50.2× |

## Small inputs: complete training step (AdamW)

| Network | CPU ms | CUDA ms | Vulkan ms | Vulkan / CUDA slowdown |
|---|---:|---:|---:|---:|
| MLP | 9.638 | 2.170 | 944.186 | 435.1× |
| U-Net | 10.947 | 7.796 | 435.734 | 55.9× |
| StyleNet (5 blocks) | 18.380 | 10.085 | 1516.262 | 150.3× |
| ResNet18 | 50.216 | 12.504 | 5581.125 | 446.3× |
| MobileNetV3-small | 38.156 | 28.535 | 1915.589 | 67.1× |
| Embedding network | 10.848 | 2.013 | 3119.587 | 1,549.7× |
| Transformer (4 layers) | 48.109 | 13.031 | 4154.700 | 318.8× |
| VGG11 | 285.647 | 24.127 | — | — |

**VGG11 Vulkan training:** exceeded the 95-second total process budget during warmup/sampling. No stable per-step median was obtained. This is a timeout, **not** a claim that one training step takes 95 seconds.

**ResNet18 training correctness** requires further investigation: small spatial BatchNorm layers can amplify tiny errors. Its training-time result is informative but not validated as equivalent parameter updates. MobileNetV3 and VGG11 include stochastic layers, so first-step losses can differ by device.

## Medium inputs: inference (more realistic spatial resolution)

| Network | Input shape | CUDA ms | Vulkan ms | Vulkan / CUDA slowdown |
|---|---|---:|---:|---:|
| MLP | 256×784 | 0.231 | 18.208 | 78.8× |
| U-Net | 1×3×64×64 | 1.091 | 76.355 | 70.0× |
| StyleNet (5 blocks) | 1×3×36×36 | 1.532 | 97.240 | 63.5× |
| ResNet18 | 4×3×64×64 | 2.102 | 106.907 | 50.9× |
| MobileNetV3-small | 2×3×96×96 | 3.832 | 116.343 | 30.4× |
| Embedding network | 128×64 | 0.138 | 201.265 | 1,458.4× |
| Transformer (4 layers) | 4×128 | 1.283 | 706.455 | 550.6× |
| VGG11 | 2×3×128×128 | 2.225 | 323.937 | 145.6× |

## Measured cause of the MLP training slowdown

**Model:** 1.87 million parameter MLP, input 128×784, FP32. Medians from a separate phase-resolved experiment. Stage-by-stage CUDA synchronization can introduce additional overhead versus the whole-step benchmark.

| Stage | CUDA ms | Vulkan ms |
|---|---:|---:|
| Forward | 0.226 | 43.660 |
| Loss | 0.056 | 0.208 |
| Backward | 0.912 | 50.290 |
| AdamW | 1.015 | 861.194 |
| Total | 2.269 | 955.872 |

AdamW alone took **861 ms** on Vulkan and **1.02 ms** on CUDA. The Vulkan backend still runs large parts of its optimizer via CPU fallback, staging and synchronizing tensors repeatedly. Optimizing GEMM alone will not fix this training gap.

## Correctness and support limits

- **All 42 backend regression tests passed** on both real Vulkan GPUs (RTX 2080 Ti and Tesla V100); Lavapipe passed 41 with the physical-GPU-only test skipped.
- **MLP, U-Net, StyleNet**: earlier direct CPU comparisons validated forward and gradients for reduced test workloads; new benchmarks record first output statistics and first loss for reference, not full statistical accuracy.
- **Transformer**: an old custom scaled-dot-product-attention shader produced incorrect outputs and lacked an autograd kernel. It was disabled. The current safe CPU fallback plus proper `_reshape_alias` view produced a maximum forward difference ≈2.9e-6 and worst per-parameter gradient difference ≈1.7e-8 versus CPU for batch 2, sequence 128. Its Vulkan timings therefore include substantial CPU execution.
- **Native Conv2D** kernels are correctness-first scalar implementations. They are not comparable to cuDNN tensor-core-optimized kernels at large channel counts.
- **BF16/FP16**, GPU-resident optimizer, optimized attention forward/backward, and general asynchronous dispatch are not yet implemented. CUDA PyTorch 2.14.1+cu130 does not support the V100 SM70, so same-version CUDA/Vulkan benchmarking was performed on the RTX 2080 Ti only.

## Reproduction

```bash
# Install matching CPU (Vulkan) and cu130 (CUDA) PyTorch 2.14.1
# and check out the upstream XDNA2 repository next to this project.
git clone https://github.com/mistrjirka/pytorch-XDNA2.git ../pytorch-XDNA2
git -C ../pytorch-XDNA2 checkout c7c063101ec5209caf8022f5b4ea6948d1a8d6ed
export VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json
export TORCH_VULKAN_PHYSICAL_DEVICE_INDEX=0
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0
.venv/bin/python benchmarks/compare_cuda_vulkan.py --backend vulkan --model style_net --scale medium --mode inference --zoo ../pytorch-XDNA2
.venv-cuda/bin/python benchmarks/compare_cuda_vulkan.py --backend cuda --model style_net --scale medium --mode inference --zoo ../pytorch-XDNA2
```

Original per-run data, including all measured samples, driver/version, selected backend, and input shapes: [`results/gpu_comparison/`](results/gpu_comparison/).
