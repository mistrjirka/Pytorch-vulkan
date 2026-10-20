# cuDNN Winograd vs PyTorch Vulkan: measured algorithms (20 October 2026)

## Setup and numerical basis

RTX 2080 Ti (SM75), PyTorch 2.14.1+cu130, cuDNN version 92400 (9.24), FP32 convolution, CUDA TF32 off, GPU synchronization after every timed operation, same seeded CPU input/weights/bias, float64 CPU convolution oracle. CUDA comparisons turn ON torch.backends.cudnn.benchmark to measure the fastest selected candidate. Vulkan forbids CPU fallback. Runtime excludes initialization, first-call compilation, and cuDNN algorithm search. Benchmarks: benchmarks/probe_winograd_forward.py and benchmarks/profile_cuda_conv_kernels.py.

## What CUDA executes

Torch CUDA profiler after cuDNN warmup shows:

- Dense FP32 64ch, 64x64 and 128x128: _5x_cudnn_volta_scudnn_winograd_128x128_ldg1_ldg4_relu_tile148t_nt_v1, plus cudnn::winograd::generateWinogradTilesKernel and an elementwise CUDA bias addition.
- Grouped 64ch 128x128: same Winograd kernel, launched separately for groups.
- 128ch/8-group 64x64: _5x_cudnn_volta_scudnn_128x32_relu_medium_nn_v1, plus computeOffsetsKernel.
- 128ch/4-group 64x64: benchmark=False selected Winograd plus a large genericTranspose_kernel and ran at ~1.61ms. After benchmark=True cuDNN selected a different 128x32 scudnn family at ~0.123ms.
- These are observed cuDNN kernel family names, not proof of exact cuDNN backend API algorithm enum or internal tile structure. The "relu" substring is an internal name, not an activation requested by the benchmark.

Nsight Compute on the actual dense64 cuDNN Winograd kernel reports 0 sm__inst_executed_pipe_tensor.sum and 137,101,312 smsp__sass_thread_inst_executed_op_ffma_pred_on.sum in one kernel launch. Thus the measured CUDA Winograd kernel is FP32 FFMA CUDA-core execution, not Tensor Core HMMA. CUDA still benefits from optimized, closed-source scheduling and vector loads, but equivalent arithmetic / workgroup communication are possible in Vulkan. No hidden tensor instruction is required for this observed FP32 path.

NVIDIA documents multiple convolution families: implicit GEMM, precomputed implicit GEMM, direct, FFT and Winograd. The actual observed selection differs by convolution geometry.

References:
- https://docs.nvidia.com/deeplearning/performance/dl-performance-convolutional/index.html
- https://docs.nvidia.com/deeplearning/cudnn/backend/v9.18.1/api/cudnn-ops-library.html
- https://docs.vulkan.org/features/latest/features/proposals/VK_KHR_cooperative_matrix.html
- https://github.com/Tencent/ncnn/blob/master/src/layer/vulkan/shader/convolution_pack4_3x3s1d1_winograd_gemm.comp (BSD-3-Clause reference, no source copied)

## Matched steady-state RTX 2080 Ti FP32 operator latency

All values milliseconds, CUDA cuDNN autotuned; explicit synchronization with resident input tensors. Lower is better.

| Case | CUDA | Vulkan | Vulkan / CUDA |
| --- | ---: | ---: | ---: |
| Dense 64ch, batch2, 16x19 | 0.0509 | 0.1526 | 3.00x |
| Dense 64ch, batch2, 64x64 | 0.0832 | 0.3139 | 3.77x |
| Dense 64ch, batch1, 128x128 | 0.1395 | 0.5469 | 3.92x |
| Grouped 64ch/4, batch2, 64x64 | 0.1045 | 0.1852 | 1.77x |
| Grouped 64ch/4, batch1, 128x128 | 0.1328 | 0.2872 | 2.16x |
| 128ch/8 groups, batch2, 64x64 | 0.1581 | 0.2874 | 1.82x |
| 128ch/8 groups, batch1, 128x128 | 0.2285 | 0.4690 | 2.05x |
| 128ch/4 groups, batch2, 64x64 | 0.1230 | 0.4334 | 3.52x |
| 96ch/4 groups, batch1, 64x64 | 0.0852 | 0.1880 | 2.21x |

Raw timing and profiler JSON: benchmarks/results/winograd-cuda-20261020/ (untracked; not copied into git). CUDA profiling includes slow and fast candidates and demonstrates that an unoptimized CUDA path is a misleading comparison baseline.

## New Vulkan GEMM K16 experiment

The new independent FP32 shader conv2d_winograd_gemm_output_k16.comp expands the per-workgroup K reduction tile from 8 to 16 channels while preserving the 8x8 output workgroup, 16 transform planes, generic convolution shapes, inverse transform and bias. Shared memory rises from 8 KiB to 16 KiB. The original K8 remains the default.

Within-process ABBA/BAAB controlled K8 versus K16, GPU synchronized, float64 CPU oracle, no CPU fallback:

| Case | RTX K8 | RTX K16 | V100 K8 | V100 K16 |
| --- | ---: | ---: | ---: | ---: |
| Grouped medium | 0.1867 | 0.1747 | 0.1883 | 0.2340 |
| Grouped large | 0.2795 | 0.2565 | 0.2861 | 0.3611 |
| Grouped small | 0.1059 | 0.1028 | 0.1011 | 0.1169 |
| Forced Winograd dense small | 0.1192 | 0.1042 | 0.1452 | 0.1419 |

K16 is a **cross-device regression** on V100 despite its RTX 2080 Ti speedup. Preserve K8 as universal production default. Optional environment variable TORCH_VULKAN_CONV_WINOGRAD_GEMM_K16=1 enables experiment, falling back safely if the GPU reports under 16 KiB shared memory. It is not a model/shape-specific hack. The V100 and RTX GPUs both report 32-wide subgroups and 48 KiB shared memory, so those two simple capabilities are insufficient to choose between K8 and K16. Do not assume AMD prefers either variant without real benchmarks.

To replicate K8-vs-K16: benchmarks/compare_winograd_k_tiles.py. The CUDA profiler: benchmarks/profile_cuda_conv_kernels.py. The three GPU focused Winograd suites with K16 pass: 9 tests each on RTX 2080 Ti, V100 and Lavapipe.

## Independent DeepSeek audit availability

OpenCode 2.0.16 is installed on the connected hub, but currently configured models do not include DeepSeek. Another computer with the expected configuration is disconnected. No DeepSeek subagent was run; provider credentials and settings remain untouched.

## Best next engineering investments

1. Profile Winograd transform, GEMM, and inverse stages individually with Vulkan GPU timestamps. Do not infer kernel efficiency solely from the full operator latency.
2. Optimize M/N tiling and global-memory coalescing, plus register pressure. K16 changes reduction blocking but does not address the main difference with cuDNN's larger output tile family.
3. Try F(4x4,3x3) with strict accuracy guards and workspace limits, comparing against F(2,3), direct and implicit-GEMM. Avoid unbounded intermediate allocation.
4. Make device selection robust by benchmarking supported kernels and amortizing first-call cost; avoid exact-model names or blind vendor heuristics.
5. Support FP16/cooperative matrices where capabilities allow, but keep a portable FP32 path. The measured cuDNN FP32 kernel did not use Tensor Cores.
6. Before claiming a practical CUDA/ROCm alternative, test whole-model inference and training, numerical gradients, CPU fallback, and Radeon hardware.
