# Bounded-workspace convolution gradient dispatch

20 October 2026. Generic FP32 Vulkan backend; no network-specific kernel.

## Root cause

The old tiled dWeight algorithm always used spatial Split-K chunks of
256 positions. Its scratch requirement was

    4 * Cout * (Cin/groups) * Kh * Kw * ceil(N*Hout*Wout/256).

If scratch exceeded the configured limit (default 64 MiB), the dispatcher
removed this algorithm entirely and selected a much slower generic
per-weight reduction. A dense 128→256 3×3 128×128 convolution requires
exactly 72 MiB under the fixed-split scheme: the old system had a large
71→72 MiB performance discontinuity.

## General implementation

csrc/ops/convolution.cpp now constructs a WeightSplitPlan from convolution
geometry, tile-K, available scratch and Vulkan limits (buffer range,
uint32 indexing and dispatch-z). The plan is used consistently by
candidate eligibility, dispatch grid, scratch allocation and final reduction.

- When possible, retain the exact original 256-position split and reduction.
- Otherwise use fewer **longer tile-K-aligned** splits. Recompute the actual
  count after aligning the span; allocate only the needed number of
  weight-gradient partials, and never exceed the requested scratch limit.
- The tiled dWeight shader and its finalize shader receive the new split
  span as a push constant; they no longer assume a fixed 256 split.
- A budget smaller than **two complete partial weight gradients** still
  uses the existing safe fallback. The original zero-budget behavior
  is preserved and tested; the default budget remains 64 MiB.

No shape-specific code, CUDA instructions or numerical test relaxations.

## Real split plans

One 128→256, 3×3, 128×128 convolution:

| Budget | Split span | Partial count | Scratch |
|---|---:|---:|---:|
| 64 MiB | 304 | 54 | 60.75 MiB |
| 71 MiB | 272 | 61 | 68.625 MiB |
| 72 MiB | 256 | 64 | 72 MiB |

All three are verified from actual dispatch counters on RTX 2080 Ti,
selected the register3 tiled shader, and had zero CPU fallbacks.

## Matched performance evidence

Lab02 MyNet, FP32 batch 1 128×128, seeded synthetic identical model and
inputs, training (forward/MSE/backward/Adam), 12 warmups, 100 synchronized
steps per process. Inputs/weights have matching SHA256 hashes across
legacy/adaptive runs.

| RTX 2080 Ti variant | Run medians | Median of process medians |
|---|---|---:|
| Previous, 64 MiB | 104.71 ms | 104.71 ms |
| Previous, 96 MiB | 63.18 ms | 63.18 ms |
| Adaptive, 64 MiB | 63.75 / 61.47 ms | **62.61 ms** |
| Adaptive, 71 MiB | 63.04 ms | 63.04 ms |
| Adaptive, 72 MiB | 63.63 / 61.61 ms | **62.62 ms** |
| Adaptive, 96 MiB | 62.54 ms | 62.54 ms |

The default configuration is **40.2% faster** than the previous default
and the former scratch-limit discontinuity is eliminated.

On Tesla V100, adaptive 64 MiB medians were 58.29 / 57.79 ms; adaptive
72 MiB medians 59.57 / 59.62 ms. This does NOT establish V100 speedup
over the old implementation, which was not run for that particular
network. Physical AMD/RADV remains to be benchmarked for this patch.

Full raw timing series (100 timed steps per process), per-process
metadata, geometry and dispatch-counter evidence:
benchmarks/results/adaptive-weight-split-2026-10-20/{summary.json,timings.jsonl.gz}.

## Numerical verification

- New generic tests verify bounded scratch and split arithmetic at 1 MiB
  and 64 MiB for dense 3×3 and grouped, strided, dilated 3×5 cases.
  All requested gradients compare to CPU float64 references.
- **143 focused convolution tests per device and execution mode** passed
  on RTX 2080 Ti, V100 and Lavapipe; both eager and deferred; CPU fallback
  explicitly forbidden.
- Entire repository regression suite: **3689 passed** on RTX 2080 Ti;
  **3689 passed** on V100; **3688 passed, 1 skipped** on Lavapipe.
- Full Lab02 MyNet CPU FP32 oracle at both 64 and 72 MiB: all 50 gradient
  tensors passed; maximum forward output discrepancy 2.4437904e-6.
- glslangValidator compilation, spirv-val validation for generated shaders.

## Remaining architectural work

With space for fewer than two complete gradient partials, this algorithm
still falls back; a future streaming tiled accumulation mechanism can
handle that lower-memory regime. Extreme split spans also need a measured
cost model instead of assuming longer tiles always win. Descriptor binding
churn, optimizer fusion, and faster dWeight/dInput kernels remain
independent performance bottlenecks. The full-network 1.5× CUDA goal is
not yet met.
