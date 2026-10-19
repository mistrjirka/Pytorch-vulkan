> Historical audit note: the inline log-softmax CPU route was later found to bypass fallback counting. The numerical outcomes below remain the recorded results, but their zero counters alone are not exhaustive residency proof. The [current report](pointwise-optimization-2026-10-19.md) documents the fix and new eager/deferred residency checks.

# Final verification of the agreed correctness and fallback scope

The agreed single-device FP32 correctness/native-execution gate passes for
published implementation `be9c29d90e2c23367d7ccded946ee1bf83bddf25`.
There are **zero unresolved in-scope failures** and **zero numerical CPU fallback
calls or attempts in successful or upstream expected-failure cases**. This is
completion of the selected audited scope, not a claim that every PyTorch/CUDA
API, precision or possible workload has been implemented.

## Exact upstream run

The full pinned selection contains 8,696 original cases from 16 test families,
covering FP32, supporting int32/int64/bool and test-defined dtypes. Each original
case ran exactly once across six disjoint shards on RTX 2080 Ti and V100; this
is not 8,696 cases per device. No cases timed out or required worker restarts.
The coordinator verified complete duplicate-free IDs, source identity, extension
identity and immutable backend source before and after execution.

| Original raw outcome | Cases |
|---|---:|
| passed | 7831 |
| error | 313 |
| skipped | 446 |
| unexpected success | 32 |
| expected failure | 74 |

The raw error outcomes are retained exactly. They have these independently
explained scope blockers:

| Scope blocker | Cases |
|---|---:|
| deferred precision | 272 |
| cross device rng mask comparison | 1 |
| cuda source jit gate | 40 |

The precision entries require actual Double/Half/BFloat16/ComplexDouble/
ComplexHalf/FP8 tensors or Float64 weighted-bincount output. They remain raw
errors, and later samples behind such blockers are not assumed to pass.
The 40 Jiterator cases call upstream's CUDA/ROCm-only literal-source JIT API;
this is outside the audited ATen/default `torch.compile` contracts.
The single alpha-dropout case compares CPU and GPU random masks exactly.
Its exact captured fixture also differs from CPU on matched CUDA: CUDA and
Vulkan both satisfy the affine output rule and mask-gradient checks. Upstream
already marks this comparison as an expected failure for CUDA/XPU.

Skipped, expected-failure, unexpected-success and unvisited samples are never
counted as passing cases. All-status fallback counts and every original trace
are available in the [scope ledger and compressed raw shard results](../benchmarks/results/upstream-opinfo-final-be9c29d/).
Remaining fallback attempts are confined to the explicitly deferred precision
blockers. Successful and expected-failure cases have no fallback attempts.

## Independent verification

| Check | Result |
|---|---|
| Strict native regressions, RTX 2080 Ti | 1,664 passed |
| Strict native regressions, V100 | 1,664 passed |
| Strict native regressions, Lavapipe | 1,664 passed |
| New regressions, strict eager mode on Lavapipe | 76 passed |
| Ordinary repository, RTX / V100 | 3,408 passed on each |
| Ordinary repository, Lavapipe | 3,407 passed, one skipped |
| Full strict repository, V100 | 3,397 passed; same 11 unchanged explicit FP64 failures |
| Original complex64-only OpInfo samples | All 31 pass across all six entries, zero fallback attempts |
| Composite backward / forward-AD probes | All 10 pass |
| Matched finite-value CUDA / Vulkan controls | All 14 pass on each, identical serialized inputs and PyTorch commit |
| Focused original narrow-copy cases | 15 passes, two upstream unexpected successes; zero fallback attempts |

The unchanged 11 repository FP64 failures are matched by exact test IDs and
source-file hashes against the prior ledger. The 76 earlier upstream expected
failures were also executed as original built-in CPU/CUDA test instances:
both controls retained 72 expected failures and four skips. These are controls,
not passes. Four compare-CPU cases are skipped on CUDA upstream; installed CUDA
LU kernels cannot execute on the available GPU, and CUDA lacks narrow_copy.out.
Raw control traces and these limitations are preserved in checkpoint 12.

## Fixes and generality

The earlier checkpoints fix general activation, loss, arithmetic, reduction,
indexing, sorting, random, convolution, pooling, sparse, embedding, spectral and
linear-algebra paths. Later fixes include exact Int64 convolution/pooling,
typed factories/padding, native general FP32 eig/eigvals and complex64 GEMM/LU
for their gradients. Checkpoint 12 adds complex SiLU/exp/neg/power, gradient
accumulation, complex-alpha addition, complex fold/unfold, general overlapping
window-gradient broadcasting, and typed alias-safe narrow_copy.out. Checkpoint
13 adds correct complex-power zero/nonfinite handling, a scaled complex square
root, and CPU-compatible sqrt/rsqrt/reciprocal/-2 specialization contracts.
Its 14 boundary tests check 49 input combinations each on all three devices;
real and imaginary components are compared independently. Vulkan matches all
14 CPU cases. Matched CUDA controls match six and differ in eight zero/nonfinite
component cases; the complete raw differences are retained.

Dispatch depends on operator geometry, dtype, strides and device capability.
There are no network-name checks, U-Net exceptions or model-specific shape lists.
Integer arithmetic uses uint32 words where shaderInt64/Float64 is unavailable.
JAMA 1.0.3 eigen reference provenance and its public-domain notice are retained.

The immutable af259ae audit found complex SiLU and a later all-status inspection
found narrow_copy.out fallbacks hidden by upstream expected-failure decorators.
Both paths now execute on the GPU. The complete d43d267 run passed its selected
gate before an independent complex-power pole probe found an additional bug.
After the boundary fix, the entire pinned selection was rerun on the final
source. Upstream test_out_warning may choose a dtype
from an unordered set when FP32 is unsupported; independent checks of all 31
original complex64-only samples now cover that selection blind spot explicitly.
An initially malformed new internal unfold regression was corrected to use a
proper expanded window; original upstream tests were never edited.

## Reproduction and limits

PyTorch source: `5c4886908584029761b579af026dcfb627c84070`.
Pinned test_ops.py SHA256:
`17ffae61a1654f23d7c2c8f73879bee99c56a80482d061c96975c0fc4720020c`.
Backend source SHA256:
`718f6b58fba20e0659af08feaca01dbe4b29c83a77b139e95507422709771b30`.
Final main extension SHA256: `0a6cca6ad0a8aa8cb5b8e41f7980e8900e108a0961277cf69ffff3c15927fa95`.
The build manifest records every backend source hash; plan.json.gz records
all original IDs and the scheduling partition. Strict runs use
TORCH_VULKAN_GPU_RESIDENT=1, TORCH_VULKAN_DEFER_SUBMISSIONS=1 and
TORCH_VULKAN_FORBID_CPU_FALLBACK=1. Numerical tensors remain on the GPU;
validation/status scalars and dynamic-size metadata may synchronize to CPU.

Original upstream test bodies, samples, tolerances, skips and expected failures
were left unchanged. No performance claim follows from concurrent correctness
runs. FP16/BF16/FP64, other precision variants, AMP and multi-GPU remain deferred.
AMD and Intel hardware remain unverified; Lavapipe is software portability
evidence. New evidence/report commits must preserve the audited backend hash.
