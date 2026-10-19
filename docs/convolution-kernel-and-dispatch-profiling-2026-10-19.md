# FP32 convolution CUDA/Vulkan profiling and bottleneck experiments

October 19, 2026 · RTX 2080 Ti, matched architecture, PyTorch CUDA / custom Vulkan.

## Conclusion

**GPU computation and host/command overhead are both major limitations; transfers
between CPU and GPU are not the bottleneck in the production GPU-resident path.**
Simply accelerating the final weight-gradient reduction is not worthwhile:
the *partial-gradient production* shader consumes 94–98% of the GPU execution
time for the sampled split-gradient workloads.

All timed Vulkan operators prohibit CPU numerical fallback and pass the existing
CPU-float64 oracle comparisons. CUDA TF32 is disabled. CPU→GPU setup transfers
and explicit oracle downloads happen outside timed calls.

### Matched individual phases

Times in microseconds. Vulkan GPU data are Vulkan timestamp spans in **eager,
resident** mode; CUDA GPU data are the **sum of traced CUDA kernels**
from an independent profiler. Wall includes host dispatch, Vulkan/CUDA overhead,
allocation and final synchronization, and is not additive with the GPU time.
Five warmups and 15 steady-state wall samples per phase; five separate CUDA
kernel-profile iterations. Exclusive RTX 2080 Ti access; each run is a separate
process. Do not treat GPU sums as directly equivalent to whole-step wall time.

| Shape and phase | Vulkan wall | Vulkan GPU shader | CUDA wall | CUDA summed kernels |
|---|---:|---:|---:|---:|
| Large 3×3 forward | 206 | 141 | 74 | 41 |
| Large 3×3 dInput | 238 | 157 | 72 | 29 |
| Large 3×3 dWeight | 276 | 184 | 139 | 93 |
| 11×11 forward | 421 | 357 | 166 | 118 |
| 11×11 dInput | 621 | 538 | 176 | 132 |
| 11×11 dWeight | 443 | 351 | 108 | 61 |
| Depthwise forward | 73 | 11 | 31 | 7 |
| Grouped dInput | 115 | 39 | 109 | 94 |
| 1×1 forward | 84 | 22 | 40 | 11 |

The CUDA kernel trace identifies the actual techniques in use:

- Dense large **3×3 forward and input gradient**: cuDNN
  `scudnn_winograd_128x128` (plus small tile-generation kernels).
- Dense large **3×3 weight gradient**: `volta_sgemm_64x64_nt`
  plus `winogradWgradData4x4`, `winogradWgradDelta4x4`, and
  `winogradWgradOutput4x4`.
- **11×11 forward**: `fft2d_r2c_32x32`, `fft2d_c2r_32x32`,
  complex-domain multiplication. The **11×11 input gradient** instead uses
  a cuDNN `dgrad_engine` kernel, and **weight gradient**
  a `wgrad_alg0_engine` kernel. CUDA does *not* use FFT in every phase.
- **Depthwise gradients**: purpose-built `conv_depthwise2d_backward_kernel`
  and `conv_depthwise2d_grad_weight_kernel`.
- **Grouped dWeight**: `wgrad2d_grouped_direct_kernel`, not a generic
  dense 3×3 Winograd plan.

The selected kernel varies substantially across forward/dInput/dWeight,
so Vulkan must preserve **phase-specific algorithms and generic fallbacks**.
Full kernel names and timing samples are saved in
[`raw-profiles.jsonl.gz`](../benchmarks/results/convolution-root-cause-2026-10-19/raw-profiles.jsonl.gz).

## Explicit copy/no-copy experiment

Using the **same resident, eager, instrumented Vulkan kernel**,
with or without GPU residency (no algorithm changes):

| Operation | GPU-resident wall | Nonresident wall | Nonresident upload span | Nonresident download span |
|---|---:|---:|---:|---:|
| Large 3×3 forward | 206 µs | 956 µs | 92 µs | 650 µs |
| Large 3×3 dInput | 238 µs | 746 µs | 171 µs | 329 µs |
| Large 3×3 dWeight | 276 µs | 550 µs | 244 µs | 13 µs |
| Depthwise forward | 73 µs | 211 µs | 20 µs | 111 µs |

Nonresident work is much slower due to unconditional input/output copies.
Resident mode's recorded upload slots are roughly 1–2 µs; these are often
GPU dependency/no-op synchronization slots rather than actual transfers.
GPU-resident Vulkan model timing records **zero CPU fallback calls and zero
host-read calls/bytes**. CUDA's per-operator profiled memory-copy events are
empty in the selected forward/dInput cases. These observations do **not**
rule out device-to-device copies. In the separately profiled large U-Net,
Vulkan `cat_copy` uses ~0.170 ms, `device_copy` ~0.051 ms and
`native_pack` ~0.272 ms per step, much less than convolution compute.

## Split-gradient anatomy (new internal GPU timestamps)

The Vulkan `GpuTiming` reporting was extended, with no changes to shader
arithmetic, to measure the first kernel, dependency barrier, and final kernel
in `run_compute_pair`. Six warmups and 25 timed repetitions with Vulkan
timestamp profiling enabled:

| dWeight shape | Partial kernel | GPU barrier | Final reduction | Entire GPU sequence |
|---|---:|---:|---:|---:|
| Large 3×3 | 177.0 µs | 1.2 µs | 6.3 µs | 184.5 µs |
| Small 3×3 | 90.0 µs | 1.0 µs | 4.3 µs | 95.3 µs |
| 10×10 | 303.0 µs | 1.1 µs | 4.8 µs | 308.9 µs |
| 11×11 | 345.7 µs | 1.0 µs | 5.0 µs | 351.7 µs |

The first kernel is **94–98% of the execution span**. Optimize its image
gathers, row/column arithmetic, memory reuse and reduction parallelization,
not the finalizer first. This does not imply its *temporary buffer traffic* is
free; the partial shader writes the buffer that the finalizer later reads.

## Whole-network attribution: identical large U-Net FP32 network

- Normal GPU-resident, **deferred** Vulkan training median: **52.01 ms**
  (25 samples, 8 warmups). Inference median: 8.84 ms. Model output and
  gradients pass reference parity.
- Separately instrumented **eager** Vulkan GPU shader time: **25.53 ms**
  per training step, of which **18.96 ms** is convolution shader work,
  **3.14 ms** is batch normalization, and the remainder is other shaders.
  These are aggregate GPU timestamps from 15 steps with instrumentation.
- CUDA CPU/GPU tracing (8 warmups, 3 traced steps)
  totals **10.79 ms of device events per step**, including ~2.83 ms of
  Winograd forward/dInput kernels and ~2.58 ms of SGEMM kernels.
- A separate deferred Vulkan CPU trace records **552 queued operations in
  three steps**, **3 queue submissions**, **493 bound-algorithm cache
  misses and 493 evictions**, **493 pipeline-cache hits** and **0 new
  pipeline compilations**. That is ~164 new algorithm bindings per step
  under this instrumented run. Input/outputs typically use new buffers,
  while the shared Vulkan pipeline is reused. The profiler's
  `network/final_sync` CPU interval is ~25.5 ms; **this contains
  GPU queue execution/waiting**, not 25.5 ms of pure CPU overhead.

These are **different execution modes and profiled/unprofiled instruments**:
do not subtract Vulkan eager 25.53 ms from deferred 52.01 ms and call the
difference purely CPU time. Nonetheless, the large difference between
kernel and end-to-end timings, the ~30–45 µs outside the command-eval span
per eager isolated operation, and binding-cache churn indicate material
host/driver/dispatch overhead in addition to slower GPU kernels.

### New geometry-aware dispatch experiment

An alternating forced-tile experiment compares the existing base 8×16 tile
with the compiled 3×3 register tile, requiring CPU-float64 forward and
gradient parity on both Turing (RTX 2080 Ti) and Volta (V100).
The register tile was **slower** for small 32-channel cases with
up to ~576 spatial/batch positions, **faster** with 64 input/output
channels at 1024 positions, and **faster** with 32 channels at 2048
positions. Consequently the default branch now considers those
general throughput regimes while continuing to use the base path on
low-work cases. No model name or hardcoded tensor dimensions are used
for dispatch. Other filter geometries keep the original candidates.
Two further explicit test cases verify the modified default dispatch.

## Priorities

1. Implement a **general, capability- and geometry-guarded Winograd
   backward-input and backward-weight algorithm** for dense 3×3 FP32;
   compare the *whole transform+GEMM+inverse* path, not isolated compute
   kernels, with cuDNN. Preserve the generic implicit GEMM fallback.
2. Improve the **gathering and addressing within the partial weight-gradient
   shader**; retain the deterministic final reduction unless an actual
   split-size or workspace experiment demonstrates improvement.
3. Reduce per-call Vulkan overhead with safe descriptor binding/command reuse
   (potentially descriptor-buffer/push-descriptor, after capability checks).
   Kompute's per-buffer `Algorithm` binding currently churns even though
   underlying pipelines are cached. In-flight descriptor lifetime must
   remain correct; do not merely increase cache size without reuse evidence.
4. Introduce distinct algorithms for depthwise and large-filter shapes
   where CUDA kernels conclusively use them. Test against diverse shapes,
   groups, stride, dilation and workspace budgets, both GPUs and Lavapipe.

The original **1.5× CUDA goal remains unmet**. The earlier
32-phase operator acceptance benchmark is documented in
[`convolution-specialization-2026-10-19.md`](convolution-specialization-2026-10-19.md).
Use `benchmarks/check_convolution_target.py` against paired fresh runs
after meaningful performance changes.

## Numerical verification

- After the geometry change and profiling instrumentation: full local
  repository tests **3,685 passed** (RTX), **3,685 passed** (V100),
  **3,684 passed + 1 skipped** (Lavapipe), all with zero unanticipated failures.
- **139 focused convolution tests passed** on **each device**, in **both
  eager and deferred execution** with CPU fallback forbidden.
- Original pinned upstream OpInfo broad audit of 8,696 cases, run with the
  general 3×3 candidate before the latest geometry threshold/profiling-only
  refinements: outcomes exactly match the earlier baseline, with zero
  changed statuses, **7,831 passed**, 313 errors, 446 skipped, 32
  unexpected successes and 74 expected failures. Unsupported/xfail/skipped
  cases are not counted as passes. The new geometry refinement has fresh
  local numerical tests but **not** a fresh second 8,696-case rerun.
  [Raw upstream outcomes](../benchmarks/results/convolution-specialization-2026-10-19/upstream-full-results.jsonl.gz).
- All newly collected isolated timed operator cases passed their unchanged
  float64 CPU oracle checks, including gradients. Every altered shader
  was compiled using `scripts/compile_conv_tiles.py` and `spirv-val`.

Reproducibility evidence: [summary.json](../benchmarks/results/convolution-root-cause-2026-10-19/summary.json)
and [raw GPU/CPU profiler captures](../benchmarks/results/convolution-root-cause-2026-10-19/raw-profiles.jsonl.gz).
