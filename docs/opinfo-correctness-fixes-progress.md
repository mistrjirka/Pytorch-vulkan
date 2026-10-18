# Upstream correctness fixes: work in progress

The agreed FP32 correctness and native-execution task is **not complete**. The
baseline audit exposed thousands of blocked or failed checks. Passing the local
regression suite does not establish universal PyTorch coverage.

This checkpoint fixes the confirmed unfused `addmm` coefficient error, ignored
NaN bias when beta is zero, empty matrix cases, and scalar/vector `t()` views.
It adds GPU copies/casts for FP32, int32, int64 and bool, strided Boolean writes,
exact int64 comparisons and integer-to-float rounding, comparison/logical/bitwise
operators, Boolean and FP32 integer RNG, sequence/permutation factories,
metadata `set_`/`unfold`, vector norms/extrema/dot, and `out=` adapters for existing
native kernels. The GPU SVD uses one-sided Jacobi iteration with a convergence
check, supports batches, rectangular/empty/rank-deficient matrices and full or
reduced singular-vector bases. It is a correctness implementation, not a speed
parity claim.

Copy-on-write now distinguishes read-only buffers from shader outputs. Storage
materialization copies on GPU instead of using potentially stale host bytes.
Native shader output bindings request writable storage. The remaining legacy
read-only convolution/layer-normalization/log-softmax accesses avoid unnecessary
materialization.

Inline CPU computation paths now participate in the strict fallback gate, not
just the central dispatcher fallback. Explicit transfers and scalar validation
or shape/status downloads remain distinct. Some inline CPU implementations
remain and are now reported as failures when strict mode is enabled.

## Verification at this checkpoint

- RTX 2080 Ti: 1,971 repository tests passed, one existing CPU-reference LSTM warning.
- RTX 2080 Ti, Tesla V100, Lavapipe: 252 targeted checks passed on each device,
  with CPU fallback forbidden. Numeric primitives assert no host reads; SVD
  permits a four-byte convergence/status download.
- Shader compilation and SPIR-V validation target Vulkan 1.0.
- AMD and Intel hardware remain unverified.

The pinned 8,696-case upstream suite is unchanged. Intermediate sweeps before
this checkpoint progressed from 1,433 passes to 1,713 and then 1,916. The latter
still had 6,236 errors, 438 upstream skips, 102 upstream expected failures and
four unexpected successes. Those intermediate sweeps preceded the stricter
inline-fallback gate and the SVD/bitwise additions, so they are not results for
this checkpoint. A new full upstream sweep is required. The runner now records
source-file hashes as well as the loaded extension hash to identify dirty builds.

Advanced indexing and many remaining ATen operations are still open. FP16/BF16,
AMP and multi-GPU remain deferred by agreement. No completion or CUDA performance
parity is claimed here.

## Second checkpoint: indexing, selection, byte storage and portable FP32 math

The task remains **in progress**. This checkpoint adds general GPU advanced
indexing and indexed assignment (including duplicate accumulation), nonzero,
GPU NLL label validation, typed selection/concatenation, masked fills, clamp
variants, Boolean reductions, and integer arithmetic. GPU copy, packing, RNG,
bitwise and selection primitives now also support uint8/int8/int16. Raw Boolean
bytes survive identity copies, while numeric conversions normalize truth values.
This also supplies the byte storage cloning used by upstream backward-compliance
tests. There is no model-specific dispatch.

New numerical fixes include empty broadcast arithmetic, zero-order norms with
NaN entries, and inverse trigonometric accuracy on Lavapipe. FP32 division uses
an approximate hardware quotient followed by an exact integer mantissa rounding
check. It requires neither FP64 nor native int64 nor a fused-FMA guarantee. Tests
cover 1.2 million values around integer quotient boundaries and random bit
patterns across the FP32 range, including subnormals and exceptional values.

Verification for the second checkpoint:

- RTX 2080 Ti: **2,063 repository tests passed**, one existing CPU-reference LSTM warning.
- Tesla V100 and Lavapipe: **344 targeted tests passed on each**, with CPU fallback forbidden.
- Both division-specific tests passed independently on RTX and Lavapipe.
- Edited shaders compiled and passed SPIR-V validation for Vulkan 1.0.
- AMD and Intel hardware remain unverified.

The preceding complete upstream sweep (before the byte/selection and final math
fixes) had **3,437 passes, 4,711 errors, 442 skips, 100 expected failures and six
unexpected successes** out of 8,696 cases. That is not a completion result. A
subsequent sweep was deliberately stopped after Lavapipe exposed a division bug;
its partial results must not be presented as a completed audit. A fresh full
sweep is running against this checkpoint. The build now records source and
extension hashes immediately after linking; the audit uses this manifest so
later working-tree edits cannot misidentify the tested binary.

Remaining work includes scatter/index reductions, additional linear algebra and
special functions, and other gaps reported by the unchanged upstream suite.
Explicit CPU transfers and small validation/status downloads remain distinct
from numeric CPU fallback. FP16/BF16, AMP and multi-GPU are still deferred.
