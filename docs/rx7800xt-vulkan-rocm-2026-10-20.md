# AMD RX 7800 XT validation: Vulkan versus ROCm (20 October 2026)

## Hardware and reproducibility

Remote computer: Connect to mini "mypc", Linux host "advantage".
- GPU: AMD Radeon RX 7800 XT (Navi 32, PCI device 1002:747e); Vulkan device "AMD Radeon RX 7800 XT (RADV NAVI32)", Mesa RADV 26.2.x.
- ROCm: PyTorch 2.14.1+rocm7.2, HIP 7.2.53211, GPU reports "AMD Radeon Graphics", gfx1100, 16,368 MiB, 30 compute units.
- Vulkan: PyTorch 2.14.1+cpu with native torch_vulkan compiled against main 4bc8ce0. Runs on RADV (no Mesa software Vulkan device used).
- Source checkout: ~/programovani/Pytorch-vulkan-rx7800xt-20261020, isolated from the previous ~/programovani/Pytorch-vulkan-benchmark (local shader changes were preserved).
- Script: benchmarks/probe_winograd_forward.py, --backend rocm (uses torch.cuda HIP compatibility API) vs --backend vulkan. 7 warmups, 24 samples per case, two runs each in separate processes, seeded identical inputs and filters; results checked against CPU FP64 reference. Vulkan ran with TORCH_VULKAN_FORBID_CPU_FALLBACK=1. ROCm used --cuda-autotune to toggle torch.backends.cudnn.benchmark, which maps to native convolution algorithm selection in a ROCm build; do not assume the selected MIOpen algorithm was profiled or is the global optimum.
- Measurements are synchronized per-call wall-clock latencies in microseconds, excluding input upload, validation, warmup and setup. They are *not* isolated GPU execution times, and include Python/backend overhead. GPU occupancy and per-stage Vulkan timings were not gathered.

## Default algorithm, RX 7800 XT

Median of the medians across 2 independent batches of 24 repeats. Microseconds, lower is better.

| Geometry | ROCm | Vulkan default | Vulkan/ROCm |
|---|---:|---:|---:|
| Dense 64ch, 16x19 batch2 | 71.2 | 288.1 | 4.05x |
| Dense 64ch, 64x64 batch2 | 84.1 | 400.1 | 4.76x |
| Dense 64ch, 128x128 batch1 | 97.8 | 576.1 | 5.89x |
| Grouped 64ch, 4 groups, 32x35 batch2 | 63.1 | 228.0 | 3.61x |
| Grouped 64ch, 4 groups, 64x64 batch2 | 80.9 | 311.2 | 3.85x |
| Grouped 64ch, 4 groups, 128x128 batch1 | 89.1 | 448.4 | 5.03x |
| Grouped 128ch, 8 groups, 64x64 batch2 | 95.2 | 392.0 | 4.12x |
| Grouped 128ch, 8 groups, 128x128 batch1 | 120.7 | 538.7 | 4.46x |
| Grouped 128ch, 4 groups, 64x64 batch2 | 86.7 | 604.4 | 6.97x |
| Grouped 96ch, 4 groups, 64x64 batch1 | 128.9 | 318.2 | 2.47x |

ROCm's 96ch/4-group measurements varied substantially (about 68.4 vs 189.3 us across the two runs). Treat that particular ratio as unstable until a longer alternating-run study. Other cases had smaller but nontrivial GPU thermal/clock/allocator variation; isolated timings do not establish model-level speed.

## Winograd shape selection and kernel variants

Comparing forced Winograd F(2,3) K8 to forced legacy tiled convolution on RX 7800 XT:

| Case | Tiled | Winograd K8 | Winograd speedup |
|---|---:|---:|---:|
| 64ch/4 group, 64x64 batch2 | 426.4us | 326.6us | 1.31x |
| 64ch/4 group, 128x128 batch1 | 675.0us | 444.8us | 1.52x |
| 128ch/8 group, 64x64 batch2 | 667.4us | 434.5us | 1.54x |
| 128ch/8 group, 128x128 batch1 | 1147.3us | 553.4us | 2.07x |
| 128ch/4 group, 64x64 batch2 | 1235.8us | 550.4us | 2.25x |

K8 Winograd improves grouped large workloads versus the old tiled path, including on AMD. The current automatic grouped-convolution heuristic selects it for the eligible shapes and preserves alternative shaders for other shapes. This is a portable semantic optimization, but not yet within the 1.5x-ROCm total-performance goal.

Controlled within-process ABBA/BAAB K8 versus optional K16 (identical allocations and tensors, 3 rounds, 14 repeats) on RX 7800 XT:

| Case | K8 | K16 | K16 regression |
|---|---:|---:|---:|
| Grouped medium | 302.1us | 368.0us | 21.8% slower |
| Grouped large | 388.4us | 533.1us | 37.3% slower |
| Grouped small | 199.8us | 235.0us | 17.6% slower |
| Forced Winograd dense small | 229.4us | 238.5us | 4.0% slower |

Keep K8 by default. K16 was faster on RTX 2080 Ti but slower on Tesla V100 and now on RX 7800 XT. Do not enable K16 on AMD merely because a capability such as subgroup width or shared-memory limit is advertised. Prefer a small geometry/device-aware benchmark when autotuning is practical, otherwise a conservative portable default.

## Correctness findings

- Strict no-fallback Winograd test subset on AMD: 10 passed, 111 deselected.
- Full current 4bc8ce0 AMD suite before the sgn fix: **3,927 passed, 5 failed, 1 skipped, 14 warnings** out of 3,933 tests (48.80s).
- Four failures are FP32 parity differences for special Chebyshev V, Laguerre L, Legendre P and shifted Chebyshev V polynomial implementations. These are *real* test failures under the repository's tolerances; do not dismiss them as harmless or claim complete correctness. Investigate recurrence rounding, input boundaries, instruction contraction and reference/algorithm semantics in a separate issue.
- Fifth failure: NaN input in real FP32 sgn. The original shader computed float(v>0)-float(v<0), which produced the wrong result for NaN on RADV. PyTorch torch.sgn(float('nan')) expects 0. A portable shader fix uses isnan(v) ? 0.0 : (float(v>0)-float(v<0)); this was validated on AMD, RTX 2080 Ti and V100 with the targeted unary tests. This fix is committed independently from the benchmark experiments.
- Full AMD suite after the sgn shader fix: **3,928 passed, 4 failed, 1 skipped, 14 warnings** (37.12s). Remaining failures are exclusively the four special polynomial cases listed above.
- The user-visible previous benchmark checkout was not reset. Fresh checkout builds from isolated clean Kompute source and does not require carrying the 183 local changes from the old tree.

## Suggested next experiments

1. Instrument Vulkan GPU timestamps to separate shader execution from host dispatch, three-transform scratch allocation, barriers and fences. Same wall-clock comparison alone cannot establish whether driver overhead or kernel occupancy dominates.
2. Profile M/N tile blocking, access coalescing, register pressure and subgroup cooperation on RDNA3. K16 enlarges 8x8 K blocking at the cost of more shared memory and loses performance on RDNA3/V100. Optimize M/N/loads rather than blindly increasing K.
3. Benchmark mixed train/eval models and backward convolutions using existing NIS lab02 and official TorchBench suites. The present table is forward-only.
4. Fix the four remaining AMD FP32 polynomial special-function discrepancies without weakening global numerical tolerances.
5. Record full workspace use, autotune startup cost, GPU temperatures/clocks and compiler revision before adopting broader dispatch changes.

Raw host-local JSON files are in ~/programovani/Pytorch-vulkan-rx7800xt-20261020/benchmarks/results/amd-rx7800xt-20261020 (not in git); scripts tracked under benchmarks/. Re-run using the exact RX 7800 XT, not the RTX 3090 connected as homer.

Validation of the cross-GPU NaN-sign correction on RTX 2080 Ti: **3,933 passed, 14 warnings** in the full repository suite (69.65s). This full-run improvement in count reflects the existing 3,932 tests plus one recently added test on main; the four AMD polynomial parity failures remain open. The IEEE NaN sign correction is a portable behavior change, not a GPU-specific shader branch.
