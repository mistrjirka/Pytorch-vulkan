# Diffusion operators: license verification, implementation, benchmarks

18 October 2026 · Project: mistrjirka/Pytorch-vulkan · Baseline commit 6b06790.

## License review — performed before implementation

| Reference code | Pinned source revision | Verified license | Integration |
| --- | --- | --- | --- |
| Tencent ncnn Vulkan | 44b02dabdcf97499281a4692ebd65bad73dcf8b6 | BSD-3-Clause, with file-level copyright headers | Inspected GroupNorm reduction, Swish and interpolation; no source/SPIR-V bytes copied |
| ggml/llama.cpp Vulkan | 5aeac37a35b44fde4bdffaec9159f2fcb253b622 | MIT | Inspected generic unary and FlashAttention; no code copied |
| Alibaba MNN Vulkan | dbf9ba61b08106a3b43426e295d1a29cb41cc8fe | Apache-2.0 | Inspected tiled attention; no code copied |
| This repository | Own LICENSE | MIT, copyright Peter Clemente III | MIT applies to new original implementation |

BSD-3-Clause and MIT are compatible with the project license, subject to preserving upstream notices and conditions when copying/adapting source. Apache-2.0 has attribution, applicable NOTICE and patent-related obligations. **All new Vulkan shaders and C++ code were written from mathematical operator definitions; no third-party shader implementation was copied.** If source is copied in the future, add the corresponding license notices, attribution, and applicable NOTICE text before redistribution.

## New general PyTorch Vulkan operations

- Native GroupNorm forward with saved mean and reciprocal standard deviation and backward for input, gamma and beta. Operates on user-specified N/C/spatial dimensions and groups with optional affine parameters and supports GPU-packed strided views. The implementation uses centered two-pass FP32 statistics reduction.
- Native FP32 unary family with general strided-layout support: SiLU forward/backward, sin, cos, exp, sqrt, rsqrt, tanh forward/backward, negation.
- Native stable axis-aware softmax and matching backward. This corrects an older non-last-axis numerical error.
- Native FP32 batched matrix multiplication with independent strides, storage offsets and batch views; standard PyTorch backward uses this implementation for both gradients.

All implementations remain runtime-geometry driven, not tailored to U-Net or another specific model. CPU fallback remains for unsupported memory layouts/capabilities, or unsupported dtypes may raise explicit errors. These are not complete FP16/BF16 training implementations.

## Controlled performance (RTX 2080 Ti, FP32)

Same small denoising U-Net architecture, batch 1, 16x16 inputs, 4 warmups + 8 measured steps, end-to-end Python wall time; GPU residency + deferred execution + opt-in native fused AdamW. The disabled version switches the new GroupNorm, unary, and BMM native operators to their explicit CPU fallback implementations, while keeping the corrected softmax. It is an ablation, not a replay of the old revision. CUDA uses standard AdamW, foreach=False.

| Workload | Vulkan, new operators disabled | Vulkan, new operators enabled | CUDA |
| --- | ---: | ---: | ---: |
| Basic U-Net inference | 2.845 ms | 1.869 ms | 1.214 ms |
| Basic U-Net training | 16.704 ms | 10.729 ms | 8.948 ms |
| Self-attention U-Net inference | 4.885 ms | 3.665 ms | 1.603 ms |
| Self-attention U-Net training | 23.572 ms | 14.594 ms | 9.080 ms |

Thus the native operators improve tiny diffusion-style U-Net training by approximately 1.56x, and the self-attention variant by 1.62x. The Vulkan/CUDA training ratios are approximately 1.20x and 1.61x, respectively. **These are small synthetic networks, not full Stable Diffusion or any performance guarantee at realistic batch/image sizes.** Timings show noticeable variability, especially in the attention cases.

CPU reference forward/backward parity measured after the implementation:

| Model | Max prediction absolute error | Max input-gradient absolute error | Max parameter-gradient absolute error |
| --- | ---: | ---: | ---: |
| Basic U-Net | 4.77e-7 | 1.86e-9 | 4.47e-8 |
| With self-attention | 6.56e-7 | 1.88e-9 | 3.73e-8 |
| Bilinear interpolation | 4.77e-7 | 1.86e-9 | 5.96e-8 |

## Verification

- 23 GroupNorm targeted tests across dimensions/groups/affine masks/layouts.
- 49 unary forward/backward tests across layouts and special float values.
- 32 axis-softmax forward/backward tests, including a regression for non-last axes.
- 31 BMM and attention decomposition gradient tests across transposed, sliced, offset and batch-broadcast matrices.
- On the original implementation baseline: 941 passed on RTX 2080 Ti and V100 with GPU residency/deferred enabled, 938 passed with 3 skips on Lavapipe, 939 passed with 2 skips in default RTX mode.
- **After rebasing onto the separate constant-buffer commit ef6aaf3 and rebuilding from source:** 963 passed on RTX 2080 Ti and V100 in GPU-resident/deferred mode; 960 passed with 3 skips on Lavapipe. Three existing PyTorch resizing warnings remain.

The instrumented CPU fallback trace counts only the generic boxed fallback, **not** explicit CPU fallback inside registered operators when an ablation flag disables a native path; do not compare its total event count as if it measured all CPU activity.

## Remaining work before general diffusion training

SDPA still uses a composed attention path rather than a fused training kernel, and masking/check operations can remain on CPU. General dropout with RNG, bilinear/bicubic resize backward, average pooling, transposed convolution, half/bfloat compute, and some scalar/timestep operations are still incomplete. Larger real checkpoints, long attention sequences and AMD/Intel hardware require dedicated numerical/performance validation.

Pinned source code: https://github.com/Tencent/ncnn/tree/44b02dabdcf97499281a4692ebd65bad73dcf8b6/src/layer/vulkan ; https://github.com/ggml-org/llama.cpp/tree/5aeac37a35b44fde4bdffaec9159f2fcb253b622/ggml/src/ggml-vulkan ; https://github.com/alibaba/MNN/tree/dbf9ba61b08106a3b43426e295d1a29cb41cc8fe/source/backend/vulkan .

### Post-integration timing cross-check (same model, later source revision)

The subsequent constant-buffer integration was also rebuilt and tested. This is a separate run, not the same opt-out/operator ablation. Medians in milliseconds from eight samples each (RTX 2080 Ti, FP32, GPU-resident/deferred Vulkan):

| Configuration | Vulkan inference | CUDA inference | Vulkan training | CUDA training |
| --- | ---: | ---: | ---: | ---: |
| Basic U-Net | 2.547 | 1.226 | 8.723 | 9.494 |
| Self-attention U-Net | 4.030 | 1.423 | 20.831 | 9.452 |

**Important variability:** Vulkan basic inference samples ranged from 1.503 to 3.173 ms, and self-attention training from 16.478 to 25.910 ms. Do not treat a single run or the apparent Vulkan-over-CUDA basic training advantage as a settled performance ranking. For attention, the post-integration result was slower than the earlier 14.594 ms measurement. Causes (GPU clocks/host scheduling/allocation effects) have not been isolated. These raw samples are preserved alongside the first ablation so later profiling can test regressions.

The original custom diffusion-style benchmark and raw, sample-level JSON results are tracked at benchmarks/probe_diffusion_train.py and benchmarks/results/diffusion-native-2026-10-18/. The figures above were recorded on baseline 6b06790 before integrating the independent constant-buffer commit ef6aaf3. The benchmark runner now stamps future reports with the active git revision and opt-in backend controls.
