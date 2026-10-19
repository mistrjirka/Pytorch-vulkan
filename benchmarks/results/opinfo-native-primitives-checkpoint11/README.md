# Native correctness checkpoint 11

This checkpoint adds exact Int64 2D/3D average/max pooling, Int64 convolution,
typed reflection/replication padding, integer factories and addcmul, and native
FP32 general eig/eigvals with complex64 GEMM/LU solve for gradients. Dispatch
uses geometry, dtype, strides and device limits; no model names or shape lists.

The public-domain JAMA 1.0.3 reference is vendored with release provenance.
Integer linspace follows the pinned CPU endpoint casts and binary64 FMA
interpolation using uint32 word arithmetic, without shaderFloat64/Int64 support.
FP64 tensor storage/computation has not been enabled.

Strict native regressions: 1,588 passed each on RTX 2080 Ti, V100 and Lavapipe.
Ordinary full repository: RTX 3,332 passed; Lavapipe 3,331 passed, one skipped.
Full strict V100: 3,321 passed, with the exact same 11 unchanged explicit FP64
fixtures retained as raw failures. No upstream test bodies, sample sets,
tolerances, skips or expected failures have been edited.

All 368 errors from the prior full b7aa287 audit were rechecked: 54 now pass;
314 remain raw errors. The case ledger identifies 273 deferred-precision cases,
40 upstream CUDA/ROCm-only Jiterator gates, and one CPU/GPU random-mask equality
comparison already xfailed upstream on CUDA/XPU. The latter's exact captured
fixture was checked on matched CUDA and Vulkan: both obey the alpha-dropout
output rule and mask gradients, and both differ from CPU masks.
No passing recheck case recorded a CPU fallback call or attempt.

Matched RTX controls use identical serialized inputs and the same PyTorch
commit on CUDA/Vulkan. Eigenvalue multisets, eigenvector residuals, invariant
gradients and all four complex LU solve modes pass their stated CPU checks.
These are correctness controls, not speed or CUDA-parity measurements.

The full current-source upstream audit is pending. This is a checkpoint, not
an overall completion claim. Later samples behind deferred fixtures remain
unvisited in the raw upstream run and are not assumed to pass. FP16/BF16/FP64,
other precision variants, AMP and multi-GPU remain deferred. AMD/Intel hardware
has not been tested; Lavapipe is software portability evidence only.
