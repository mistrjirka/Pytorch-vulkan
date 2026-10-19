# FP32 convolution specialization and 1.5× goal — 19 October 2026

## Summary

**The overall target remains unmet.** Fresh matched RTX 2080 Ti tests show **0/32 isolated convolution phases at or below 1.5× CUDA**. The complete large U-Net training step is **52.155 ms Vulkan vs 10.757 ms CUDA** (4.85×); inference is 8.993 vs 2.279 ms (3.95×). The previous release's large U-Net Vulkan median was 55.200 ms, an approximately 5.5% cross-run improvement; it is not a controlled interleaved old/new comparison.

## Maintainable implementation

- Adds **one reusable, explicitly guarded** compiled-FP32 3×3 variant of the existing 32×32 register tile for forward/input-gradient/weight-gradient. Filter dimensions are compiled constants, while batch/channels/spatial dimensions, stride, dilation, padding, groups and tensor addresses remain runtime values.
- `applicable()` rejects the compiled variant unless `kh=kw=3`. For other filters, groups or small workloads the existing generic candidate selection remains. The normal heuristic uses the new variant only for dense, unit-stride/dilation, ungrouped 3×3 filters with Cin/Cout≥32 and batch×output-spatial≥4096. No network-name or exact tensor-shape dispatch exists.
- The existing scalar/direct and tiled implementations remain accessible; explicit overrides and optional per-phase autotuning continue to work. Shader binaries are compiled with `scripts/compile_conv_tiles.py` and validated by `spirv-val`.
- The new candidate changes only integer indexing specialization and code generation; the FP32 accumulation order, semantics, autograd interface and workspace algorithm remain unchanged.
- Two larger-tile experiments (64×64 register blocking; reduction-depth 32) were **not integrated** because they regressed other shapes or showed no dependable gain. A separate explicit im2col+GEMM experiment measured 0.374 ms versus 0.201 ms native for the same large 3×3 forward input, so it was also rejected. No external kernel was copied or vendored.

## Isolated kernel evidence (exploratory interleaved processes)

The RTX 2080 Ti, strict GPU execution, unchanged seeded FP32 input and CPU float64 forward/gradient oracle were used. Wall values are milliseconds, medians of fresh-process medians. The run order was previous/register3/a separate 64×64 experiment/register3/previous. The selected 3×3 shader binaries are unchanged in the final build; the experimental candidate menu was subsequently simplified.

| Shape / phase | Previous 32×32 register | Compiled 3×3 register | Reduction |
|---|---:|---:|---:|
| 2×32×64×64 forward | 0.2189 | 0.2052 | +6.2% |
| 2×32×64×64 dinput | 0.2740 | 0.2372 | +13.4% |
| 2×32×64×64 dweight | 0.2957 | 0.2699 | +8.7% |
| 2×64×16×19 forward | 0.1465 | 0.1437 | +1.9% |
| 2×64×16×19 dinput | 0.1838 | 0.1630 | +11.4% |
| 2×64×16×19 dweight | 0.1497 | 0.1434 | +4.2% |

## Convolution goal matrix

The acceptance suite uses `benchmarks/probe_conv_dispatch.py` for **eight general convolution geometries × four phases**: 1×1, rectangular/dilated, grouped, depthwise, large/small 3×3, and 10×10/11×11 filters. Both runs are serial on the same RTX 2080 Ti and have the same input checksum. CUDA uses FP32 with TF32 off. Each phase has 8 warmups and 25 timing samples; timer includes the ordinary Python/backend operator call and final device sync, not preprocessing/parity downloads. Vulkan strict numerical fallback rejection is enabled. CUDA cuDNN benchmarking remains disabled for the comparison. This suite does not prove all possible shapes are covered.

| Geometry | Forward / CUDA | dInput / CUDA | dWeight / CUDA | dBias / CUDA |
|---|---:|---:|---:|---:|
| conv3x3_large | 3.22× | 3.81× | 2.10× | 2.65× |
| conv3x3_small | 3.04× | 3.32× | 2.68× | 2.22× |
| depthwise | 3.27× | 2.87× | 2.82× | 2.18× |
| filter10x10 | 3.24× | 4.32× | 4.62× | 2.96× |
| filter11x11 | 3.50× | 4.05× | 5.42× | 3.78× |
| grouped | 1.85× | 1.67× | 3.04× | 2.82× |
| pointwise | 2.06× | 2.36× | 1.86× | 2.09× |
| rect3x5 | 2.38× | 3.11× | 1.98× | 2.21× |

Run `benchmarks/check_convolution_target.py <vulkan-probe.json> <cuda-probe.json> --max-ratio 1.5 --require` to fail if any tested phase misses the target. A readable `goal.json` and compressed complete raw inputs (`raw.jsonl.gz`) are retained. The `--require` flag is not added to correctness CI because the performance goal is not yet met.

## Numerical verification after cleaning candidate menu

- Full ordinary repository suite: **3,683 passed on RTX 2080 Ti**, **3,683 passed on V100**, **3,682 passed and one skipped on Lavapipe**. Seven existing warnings per run.
- Strict convolution suite (`test_conv_dispatch.py`, `test_conv_universality.py`, `test_convolution.py`): **137 passed on each of RTX, V100, Lavapipe**, in both eager and deferred modes, CPU fallback forbidden. Forward and each requested gradient compare against the unchanged float64 CPU oracle.
- The full strict repository suite still reports the same **11 pre-existing intentional FP64 fallback tests as failures** (on each device); they are not kernel/shape correctness regressions. Unsupported precision remains unsupported.
- Original pinned, unchanged upstream PyTorch `nn.functional.conv2d` OpInfo selection: on both RTX and V100 **11 passed, one historical error**, identical to `be9c29d` before the change. The error arises while constructing a test Double random tensor, before the convolution test; no fallback calls in those 12 cases. Neither the original test bodies nor expected-failure metadata were modified.
- Every new matched model process passes CPU output/loss/input-gradient/all-parameter-gradient parity (`rtol=2e-3`, `atol=8e-5`); numerical Vulkan training records zero CPU fallback and zero host reads. CPU float64 operator oracle tolerances remain the original benchmarks'.
- Performance was measured only on RTX 2080 Ti. AMD/Intel physical GPUs and FP16/BF16 performance are not verified. GPU timestamp profiles from earlier commits cannot be substituted for these wall timings.

## Remaining bottlenecks and follow-on work

1. Generic algorithm families need broadening. CUDA/cuDNN uses Winograd on the measured 3×3 shapes and FFT on measured 10×10/11×11 cases. Our current core remains implicit GEMM and direct/reduction kernels. Introduce general filter/stride/dilation/capability-gated Winograd or FFT implementations **only when numerical gates and matched performance validate them**; retain generic fallback.
2. Weight-gradient split partials allocate significant scratch and require a second reduction. Evaluate reduction split sizes, workspace traffic, occupancy and reduction kernel design on multiple sizes.
3. The 1.5× target across small operations also requires reducing host dispatch/binding overhead. One-off tile tweaks alone cannot close the measured gap.
4. Rerun the full CUDA/Vulkan 32-phase target matrix, original upstream OpInfo cases, strict CPU fallback audit and multi-device numerical suite after every algorithmic change. Attention optimization is intentionally deferred.

Evidence in [`goal.json`](../benchmarks/results/convolution-specialization-2026-10-19/goal.json) and [`raw.jsonl.gz`](../benchmarks/results/convolution-specialization-2026-10-19/raw.jsonl.gz).
