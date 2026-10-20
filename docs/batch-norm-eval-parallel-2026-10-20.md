# Parallel native BatchNorm evaluation (20 October 2026)

## Motivation and implementation

The existing numerically stable two-pass BatchNorm shader launches one 128-thread workgroup **per channel**, even during inference. Evaluation does not need mean/variance reductions: saved running statistics are already supplied by the caller. For large image tensors, the per-channel execution leaves most GPU workgroups idle and forces every thread to walk many spatial elements serially.

A new vendor-neutral `batch_norm_eval.comp` shader instead dispatches **one GPU invocation per output element**. The channel index is computed from the contiguous packed NCHW-like tensor layout, and running mean, variance, weight and bias use the existing contiguous/staging normalization from the shared operator implementation. Training and both gradient implementations remain unchanged. The new evaluation path is automatic for nonempty FP32 inputs with **at least 32,768 elements**; small inputs retain the previous channel kernel to avoid overhead where both paths tie. The optional `TORCH_VULKAN_BN_EVAL_LEGACY=1` flag is for performance diagnostics only and never needed to obtain the optimized path.

The fast shader retains PyTorch's exact evaluation math `((x - mean) * rsqrt(var + eps)) * weight + bias`, including optional affine parameters and originally supported tensor ranks/layouts. GPU dispatch bounds are validated using the same device capability checks as the original kernel.

## Same-process RTX 2080 Ti benchmarks

GPU-resident deferred Vulkan execution, FP32, zero numerical CPU fallback, identical resident inputs/parameters and CPU reference outputs. The benchmark alternates legacy and optimized kernel selection inside **one process** using ABBA/BAAB scheduling; the independent subprocess run initially showed unreliable GPU clocks and was not used to determine the speedup. Each measured invocation synchronizes the GPU.

| Workload | Previous median | New median | New / previous |
|---|---:|---:|---:|
| Native BN evaluation [1,64,640,959] | 1.977 ms | **0.773 ms** | **0.391×** |
| TorchBench original U-Net eval [1,3,640,959] | 327.490 ms | **323.953 ms** | **0.989×** |
| Native BN [4,64,96,96] | 0.190 ms | **0.113 ms** | **0.597×** |
| Native BN [1,128,320,479] | 0.628 ms | **0.387 ms** | **0.616×** |
| Native BN [2,32,16,17] | 0.0723 ms | 0.0710 ms | 0.981× |
| Tiny [1,256,1,1] | 0.0749 ms | 0.0765 ms | 1.022× |

The small-tensor measurement motivates retaining the historical kernel below the general work threshold. The TorchBench speedup is **approximately 1.1%**, much smaller than the isolated operation's 2.56× improvement because convolution is still the primary end-to-end bottleneck. Full model remains far from the <=1.5× CUDA target.

Machine-readable paired blocks: `benchmarks/results/bn-eval-20261020/within-process-isolated.json`, `within-process-torchbench.json` and per-shape `within-process-*.json`. Source: `benchmarks/compare_batch_norm_eval_modes.py`.

## Investigated and rejected in this iteration

An experimental portable fused Winograd F(2×2,3×3) shader was built and numerically checked against the existing seeded double-precision operator oracle. On `conv3x3_large` it took **0.403 ms forward** against approximately **0.19–0.24 ms** for existing register3 kernels; no win, so the shader and experimental algorithm registration were discarded. The prototype repeatedly transformed filter coefficients for each spatial workgroup, unlike a pretransformed-weight GEMM-based design. Do **not** enable speculative Winograd or add exact-model routing. A future attempt should amortize transforms, benchmark scratch/workspace and GPU timings per phase, and retain generic fallback.

## Final verification and limitation

- RTX 2080 Ti: full ordinary repository suite, **3,923 passed, 14 warnings**.
- RTX 2080 Ti, V100 and Lavapipe: **92/92 native BatchNorm tests passed on each**, with `TORCH_VULKAN_FORBID_CPU_FALLBACK=1`; 5 of these tests explicitly compare the parallel and legacy evaluation algorithms to CPU output for small, medium, large and degenerate shapes.
- Raw full-model ABBA/BAAB TorchBench results: median-of-block-medians **327.490 ms legacy / 323.953 ms default**. Both use the *same unmodified TorchBench implementation and the same model weights/input in each process*, and have zero numerical CPU fallback. The comparison shows a real gain but does not close the ~8× CUDA gap.
- Note on other GPUs: V100 and software Lavapipe were checked for correctness, **not performance**. Radeon hardware was unavailable in this environment.
