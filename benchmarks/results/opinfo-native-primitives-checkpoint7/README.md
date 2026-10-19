# General native primitives, checkpoint 7

834 strict regression checks passed on RTX 2080 Ti, Tesla V100 and Lavapipe;
2,578 repository tests passed on RTX. Focused38 combines the affected unmodified
upstream cases in one fresh run. Exact counts and source/build identities are
in summary.json and the inventories. Earlier focused32-37 preserve debugging
failures. Skips, xfails and xpasses remain separate. This is not a full audit.

Six raw errors remain in the FP32-tagged torch._refs.logspace cases: the Python
reference explicitly creates FP64 linspace/pow intermediates. Their IDs and
source evidence are in deferred-reference-fp64.json. These are retained as errors,
not passes. Native FP32 logspace cases pass. FP64 arithmetic remains deferred.

Adds exact integer shifts and GCD/LCM, extra device RNG distributions, native
pseudoinverse/rank and least squares, incomplete gamma, unfold/fold and unfold
backward, matrix square root/exponential, legacy triangular solve, heaviside,
logspace and quantiles. Implementations dispatch by operator semantics, precision
and device limits; there are no model-specific shape rules. These correctness
paths include serial work and composed kernels that still need speed profiling.

Quantile rank products reproduce the wider rank precision used by pinned ATen
using portable uint32 arithmetic, without shaderFloat64 or shaderInt64. 100,015
random/boundary rank-mode comparisons match the CPU FP64 control exactly on all
three devices. Quantile gradients preserve PyTorch autograd and tensor subclass
contracts; range validation is inside the opaque native primitive.

The initial local quantile gradient check incorrectly required identical NaN tie
indices. Identical 80-case CPU/CUDA/Vulkan controls found 17 CUDA and 12 Vulkan
gradient cases differing from CPU, exclusively at tied NaNs. The final local
regression compares finite gradients exactly within default tolerances and total
NaN gradient mass per reduction; no upstream tests or tolerances were changed.
Raw diagnostic logs intentionally contain the earlier failures.

FP16/BF16/FP64, AMP and multi-GPU remain deferred. AMD/Intel are untested.
No CUDA speed parity or overall completion is claimed. Remaining general
operators still require a full strict audit and further implementations.
