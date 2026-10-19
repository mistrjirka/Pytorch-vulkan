# General native primitives, checkpoint 9

1,199 regression checks pass with CPU fallback forbidden on RTX 2080 Ti,
Tesla V100 (included in the full strict run), and Lavapipe. The ordinary RTX
repository suite passes 2,943 tests. The full strict V100 repository run has
2,932 passes and 11 raw failures, all explicitly using deferred FP64. The raw
tracebacks and source excerpts remain in the evidence; none became a pass or skip.

Adds general rank/stride FP32 iteration up to 64 axes within device buffer and
32-bit indexing limits, including high-rank gradient accumulation, broadcasting,
negative views, and alias-safe mutation. The existing eight-axis ABI remains
unchanged for shaders that require fixed-size metadata. This is shared packing
and arithmetic infrastructure, not a claim of 64-axis support in every operator.

Adds multinomial sampling with independent device generators, batched and
concatenated/padded CTC loss forward/backward, and real FP32 Bunch-Kaufman LDL
factorization/solve. Extends sigmoid and integral power for supported integer and
boolean tensors and fixes mean's invalid-dtype validation. Dispatch depends on
operator semantics and device limits, without model names or model shapes.

Focused50 reruns 275 unmodified upstream cases: 241 pass, six errors, 26 upstream
skips, one xfail and one xpass, with no CPU fallback calls or attempts. Its six
errors explicitly require deferred FP64 (integer-weight bincount and reference
triangular indices). Focused51 rechecks all LDL cases after the portable rounding
correction. Earlier focused48/49 retain the intermediate failures and fixes.

Identical cached LDL inputs are compared on the same RTX through CUDA and Vulkan,
using the same pinned PyTorch commit. FMA corrects solve roundoff; portable
single-rounded multiply-add also fixes Lavapipe's separate-rounding discrepancy.
The new helper uses 32-bit integers and is checked through the actual GPU solve
against host C99 fmaf over directed cancellation, overflow, subnormal and random
exponent cases. A host translation additionally matches two million fmaf results
bit-for-bit; that host experiment alone is not counted as GPU coverage.

CUDA and Vulkan both have large factor-entry differences from CPU LAPACK at
129x129, with matching pivots/status. Their independent normalized equation
backward errors are about 7-8e-8. Larger local factor tests therefore check an
independent CPU FP64 solve residual, while small factors retain direct CPU checks
and solving from CPU factors retains the original default tolerance. No upstream
bodies, samples, skips or tolerances were changed. Raw earlier failed local tests
remain in diagnostics, including the corrected CPU singular-info exception oracle.

LAPACK's compatible permissive license and attribution are retained in source
and wheel license data. No tensor numerical computation is downloaded for these
paths; lengths, validation/status scalars and explicit test comparisons may read
host memory. FP16/BF16/FP64, AMP and multi-GPU remain deferred. AMD/Intel are
untested. Eigenvalue and sparse coverage gaps remain: overall completion is open.
These are correctness implementations; no current CUDA performance parity is claimed.
