# General FP32 pointwise fusion and device-only log-softmax — 2026-10-19

Matched FP32 training latency is 6–35% lower across the six main networks on the
RTX 2080 Ti, with zero CPU fallback in the measured runs. An uncounted eager
log-softmax CPU route is also removed.

The backend now writes FP32 `lerp`, `addcmul`, `addcdiv`, `maximum` and
`minimum` results directly to their destinations. Functional, `out=` and
in-place calls use the same operator implementation. Standard PyTorch optimizers
and other callers benefit without model-specific wiring or a custom optimizer.

## Scope and dispatch

The contiguous path uses offsets and scalar-broadcast flags without constructing
expanded tensor views or uploading indexing metadata. The general path supports
broadcasting and nonnegative strided views, including transposes, slices, nonzero
storage offsets, scalars, empty tensors and ranks through eight, within the
backend's existing checked storage/index limits.

Selection checks FP32 storage, destination independence, alias mappings and Vulkan
limits. The fused shader needs five storage-buffer bindings and a 128-thread
workgroup. Unsupported fusion cases retain the existing composed GPU path.
Mutation obtains the writable buffer before binding inputs, preserving PyTorch
copy-on-write; deferred commands retain their buffer references. Partial overlap
and internally overlapping destinations receive PyTorch overlap errors.

`TORCH_VULKAN_DISABLE_FUSED_OPTIMIZER_PRIMITIVES=1` is a startup-only diagnostic
control for the pre-change composition. Published-before measurements below use
the actual previous checkout, rather than this switch. The preserved composition
has its original arithmetic order; numerical validation of the new path uses
PyTorch CPU references.

Scalar/tensor lerp uses ATen's stable branch around weight 0.5. Addcmul/addcdiv
preserve CPU scalar-multiplication order, use the existing precise division
helper, and preserve infinity times a nonzero subnormal on hardware that flushes
subnormals. Integer selection preserves the chosen extrema input's bits and
propagates NaNs. Mixed positive/negative zero ties can differ across PyTorch
backends: the CPU scalar and SIMD paths themselves disagree, and CUDA uses
different minimum/maximum tie rules. Tests require zero-sign parity where both
extrema inputs agree, plus numerical and nonfinite classification parity.

For ordinary AdamW with AMSGrad, the old code performed four destination copies
per parameter tensor, plus intermediate pointwise dispatches. These calls now
write directly in four dispatches total. This is operator fusion: the backend
does not recognize network names, parameter names, module classes or training
graphs.

## Eager log-softmax and the fallback audit

Profiling exposed a pre-existing blind spot: the old small/eager and non-last-axis
log-softmax implementation directly copied tensors to CPU without going through
the fallback audit. Normal resident/deferred benchmark runs selected its GPU
path, but timestamp diagnostics disable deferral and exposed those CPU copies.
Thus the old diagnostic fallback counter alone was insufficient evidence.

The final build removes that size/execution-mode heuristic. Supported FP32
log-softmax forward/backward now use Vulkan for arbitrary axes, nonnegative
strides/offsets, expanded and lazily negated views, scalars, empty tensors and the
backend's checked rank/index limits. Axis normalization and packing stay on the
device. Both shaders use 128-thread workgroups with two-dimensional row dispatch;
70 added tests include ranks above eight and more than 65,535 rows. Explicit CPU
diagnostic requests and unsupported precision now use the common audited fallback
helper and are rejected by the strict guard before downloading data.

## Correctness

| Check | RTX 2080 Ti | Tesla V100 | Lavapipe |
|---|---:|---:|---:|
| Full repository suite | 3,675 passed | 3,675 passed | 3,674 passed, 1 skipped |
| Focused deferred suite | 276 passed | 276 passed | 276 passed |
| Focused eager suite | 270 passed, 6 counter-test skips | same | same |
| Relevant unchanged upstream OpInfo cases | 88 passed, 2 expected failures, 26 known errors | same | same |

The focused suite enforces device execution on normal paths, checks that explicit
CPU diagnostic routes are audited/blocked, and covers all API variants, broadcast and
strided destinations with backing-storage checks, gradients, overlap errors,
lazy-clone copy-on-write, nonfinite arithmetic, scalar multiplication order, and
12 consecutive deferred AdamW updates with/without AMSGrad, checking parameters
and optimizer states.

The 116 upstream test IDs and outcomes match the published baseline exactly.
The 26 errors are the existing deferred-precision cases: four request FP64 random
tensors and 22 exercise explicit dtype-conversion variants. They remain visible
with their original scope-ledger explanations; tests, sample sets, tolerances and
expected-failure declarations were not changed.
The full suites' seven warnings are existing PyTorch beta/deprecation,
reference-LSTM, complex-cast and sparse-invariant warnings, retained in the logs.
Kompute also reports that requested Vulkan validation layers are unavailable on
this host. The new SPIR-V passes `spirv-val --target-env vulkan1.1`; these runs
are numerical/regression validation, not validation-layer-clean certification.

Validation covers available NVIDIA devices and software Vulkan. AMD and Intel
remain unverified. This change targets FP32; it adds no FP16/BF16/FP64 claims.

## Matched training measurements

All 72 final-build processes passed their unchanged CPU output/gradient parity
gate. The optimized backend recorded **zero CPU fallback calls** over 600 timed
training steps and 600 timed inference passes. Host reads were at most four bytes
each for validation/status; logical read counters are not PCIe traffic meters.

Training milliseconds, median of three process medians:

| Workload | Input shape | Before | Optimized | Reduction | CUDA | Optimized / CUDA |
|---|---|---:|---:|---:|---:|---:|
| mlp / medium | 256×784 | 9.65 | 6.28 | 34.9% | 2.34 | 2.68× |
| unet / medium | 1×3×64×64 | 31.46 | 27.68 | 12.0% | 10.05 | 2.75× |
| resnet18 / medium | 4×3×64×64 | 71.62 | 67.38 | 5.9% | 15.34 | 4.39× |
| mobilenet_v3 / medium | 2×3×96×96 | 169.19 | 134.15 | 20.7% | 34.77 | 3.86× |
| embedding / medium | 128×64 | 8.24 | 5.84 | 29.2% | 2.36 | 2.47× |
| transformer / medium | 4×128 | 77.80 | 73.35 | 5.7% | 16.17 | 4.54× |
| unet / reference | 8×3×96×96 | 105.97 | 93.23 | 12.0% | 10.14 | 9.20× |
| transformer / reference | 16×128 | 165.38 | 149.95 | 9.3% | 16.25 | 9.23× |

All eight workload/shape medians improve. MobileNet has substantial process
variation:

- before: 160.82–211.16 ms across process medians.
- after: 134.02–135.65 ms across process medians.

Inference distributions and all samples are retained in `final/wall/summary.json`
and `final/wall/raw.jsonl.gz`; this patch primarily targets training.

| Workload / medium | Compute dispatches before → optimized | Binding misses/step before → optimized |
|---|---:|---:|
| mlp | 201 → 121 | 158.1 → 14.8 |
| unet | 853 → 517 | 852.0 → 433.2 |
| resnet18 | 1368 → 872 | 1368.0 → 872.0 |
| mobilenet_v3 | 3329 → 2193 | 3329.0 → 2193.0 |
| embedding | 114 → 74 | 18.8 → 3.5 |
| transformer | 1134 → 718 | 1134.0 → 714.0 |

Every measured Vulkan run has zero warmed pipeline-cache misses. Binding-object
misses are separate from pipeline compilation. Fusion reduces allocations and
descriptor churn without changing the cache design.

The larger ResNet batch-32 case remains excluded from speed ratios because the
unchanged CUDA gradient-parity gate failed in the previous broad investigation.
This round repeats the two larger reference workloads that passed that gate; no
parity thresholds were relaxed.

An additional twelve feature networks passed the same CPU output/gradient gate
on Vulkan and CUDA, with one process per API and 25 measured steps each. The
optimized Vulkan runs recorded zero CPU fallback calls over another 300 training
steps and 300 inference passes. These exercise pooling, bilinear/bicubic and
trilinear interpolation, reductions, attention and causal masking, Conv3D and
transposed Conv3D, reflection padding/antialiasing/grid sampling, bidirectional
LSTM, EmbeddingBag/index selection/sorting/cumulative sums, FFT and linear solves.
Their one-process latency distributions are in `final/features/summary.json` and raw
records; they are coverage checks, not before/after speedup estimates.

## Profiles and remaining gaps

All 46 final-build captures passed: 16 phase, 16 GPU-timestamp and 14 normal-
batching Nsight captures, with seven paired trace analyses. Every Vulkan capture
records zero CPU fallback and host reads no larger than four bytes.

The fixed timestamp-mode MLP reads eight status bytes per step, compared with
30,728 bytes in the published pre-fix diagnostic; the removed 30,720 bytes were
bulk log-softmax forward/backward tensor downloads. Native GPU log-softmax
dispatches now appear in the shader captures. The prior fallback counter missed
that inline CPU route; the final results use the corrected implementation.

GPU `device_copy` shader dispatches per step:

| Workload / medium | Before | Optimized | Fused pointwise dispatches |
|---|---:|---:|---:|
| mlp | 40 | 0 | 40 |
| unet | 178 | 10 | 168 |
| resnet18 | 285 | 37 | 248 |
| mobilenet_v3 | 645 | 77 | 568 |
| embedding | 20 | 0 | 20 |
| transformer | 252 | 44 | 208 |

These are GPU shader copies, rather than CPU transfers. Some remaining copies
perform layout packing; their presence alone does not establish redundancy.

Normal-batching observed host API calls per step:

| Workload | Submits / fence waits | Barriers | Descriptor allocations | CUDA completion syncs |
|---|---:|---:|---:|---:|
| mlp / medium | 5 / 5 | 224 | 29.0 | 1 |
| unet / medium | 1 / 1 | 978 | 435.0 | 1 |
| resnet18 / medium | 5 / 5 | 1552 | 872.0 | 1 |
| mobilenet_v3 / medium | 5 / 5 | 3984 | 2193.0 | 1 |
| transformer / medium | 9 / 9 | 1317 | 714.0 | 1 |
| unet / reference | 1 / 1 | 986 | 439.7 | 1 |
| transformer / reference | 9 / 9 | 1317 | 714.0 | 1 |

Fusion reduces command/barrier and descriptor work but does not remove the
existing validation waits. The MLP still has five fence waits and the transformer
nine; U-Net already has one. Fence duration includes necessary GPU completion
and is not a measure of pure CPU overhead. Pipeline compilation is absent from
the warmed primary timings, while binding-object churn remains.

The largest remaining general kernel gaps are:

- Larger U-Net: convolution forward/dInput/dWeight contribute 55.34 ms of 64.58 ms instrumented Vulkan shader time (85.7%).
- Larger Transformer: SDPA contributes 71.14 ms of 92.74 ms (76.7%). CUDA attributes about 1.76 ms to efficient-attention forward/backward on the same FP32 workload.

These are diagnostic GPU attributions, not normal training latency or occupancy
measurements. GPU timestamp mode disables Vulkan batching; CUDA device-event
totals can include memsets/copies. Their sums must not be subtracted from normal
wall time to calculate CPU overhead.

The [SDPA shader](../csrc/shaders/sdpa_training.comp) performs a shared reduction
for every query/key pair in backward and recomputes the row-invariant gradient
dot product inside that loop. A tiled backward plus one dot product per query
is a concrete general candidate. Any replacement must preserve mask/causal,
dropout/RNG, rectangular shapes, gradients and capability guards. Convolution
forward/dInput/dWeight reuse is the other large kernel priority. For smaller
graphs, remaining work includes descriptor/binding reuse, reviewing redundant
barriers with full hazard/lifetime tracking, and combining pending computation
with status readback before a fence wait.


## Reproduction and provenance

The comparison rotates the published Vulkan backend, optimized Vulkan backend
and CUDA in separate processes, serially under an exclusive GPU lock. Both use
the RTX 2080 Ti (`GPU-30acbf2b-a41a-8b88-e298-fe66c4138b29`), NVIDIA driver
580.178.04, fixed input/initial-weight hashes, FP32, disabled TF32, four CPU
threads, eight warmup steps and 25 measured steps. Training includes standard
`torch.optim.AdamW(lr=1e-4, foreach=False, amsgrad=True)`. Reported latency is
the median of three process medians; per-process ranges and every sample are
preserved. Each timed process first checks outputs, loss, input gradients and
parameter gradients against CPU at the unchanged network-harness tolerances.
Dropout is disabled only for this parity check and restored for timed training.

The baseline is main `183070e5f4bac3350571f3b3bd05cf38572ad924`.
Both environments use PyTorch git `5c4886908584029761b579af026dcfb627c84070`
(2.14.1 CPU/Vulkan and 2.14.1+cu130); Torchvision git is
`1f1a920d685d46dfbe187934b785b7ecb7a182f2`.
The reference zoo is `c7c063101ec5209caf8022f5b4ea6948d1a8d6ed`.
The final backend source fingerprint is
`c0139cd9d74f92d016db9477968df1d38ab6f067f441cfaa4efccfd95bf04c21`;
the tested development-build extension SHA256 is
`24d88aa00a0d90a3ef5f2fb7b42d89ad11b35ff7d0a0b9dd6461b095a53a3a88`.
Final results are under `final/`. The earlier pointwise-only build is preserved
at commit `e943441fb7e8ed39bc55899ca0fcb755526b48a3`, with source fingerprint
`ce27f6fc64fdac42713340fde1ed1afdaf3ed684ae2ffe9e3c621d2ad606562e`;
the initial root-level correctness/wall/feature records refer to that build.
Complete before/after file hashes and binary identities are preserved alongside
the measurements, and checked unchanged before/after the wall comparison.

```bash
python benchmarks/compare_pointwise_optimization.py \
  --before /workspace/Pytorch-vulkan-baseline \
  --out benchmarks/results/optimizer-pointwise-2026-10-19/final/wall \
  --reference-cases unet transformer
python benchmarks/summarize_pointwise_optimization.py \
  benchmarks/results/optimizer-pointwise-2026-10-19/final/wall
python benchmarks/run_network_profiles.py \
  --cases mlp unet resnet18 mobilenet_v3 embedding transformer \
  --out benchmarks/results/optimizer-pointwise-2026-10-19/final/diagnostics
```

Create the before directory as a separate worktree at `183070e` and build it
using the repository build instructions. It must retain that baseline and its verified build;
run the optimized checkout separately. Wall measurements use ordinary deferred
batching and only boundary synchronization. Phase profiles deliberately
synchronize between phases. GPU timestamp diagnostics disable deferral and must
not be interpreted as normal end-to-end latency. Nsight full-step traces retain
normal batching; their phase ranges describe host enqueue time. Vulkan Nsight
GPU/API coverage is partial, so missing copy commands do not establish zero
copies or full device utilization.

Raw results, summaries, correctness logs, unchanged-test inventories and source
identities are under
[`benchmarks/results/optimizer-pointwise-2026-10-19/`](../benchmarks/results/optimizer-pointwise-2026-10-19/).
The previous broad investigation remains in
[the network performance report](network-performance-2026-10-19.md).
