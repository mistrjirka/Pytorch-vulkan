# Native correctness checkpoint 12

The immutable af259ae upstream audit found a real complex64 SiLU fallback.
An earlier run of the same test ID selected complex128 instead: upstream
`test_out_warning` chooses from an unordered dtype set when FP32 is unsupported.
The old deferred-complex128 outcome did not prove the complex64 path worked.
The full failed gate is retained in `../upstream-opinfo-fixes-af259ae/`.

A second inspection found two forbidden fallback attempts inside upstream
expected failures for narrow_copy.out. A typed native out adapter now handles
strides, empty output, negative start, input/output aliases and CPU dtype rules.
All 17 unchanged focused cases pass (15 passes, two upstream unexpected successes)
with zero fallback attempts. The original af259ae raw outcomes remain retained.

General fixes add GPU complex64 SiLU/exp/neg and scalar power, complex gradient
accumulation, complex-alpha addition and complex fold/unfold. The general
unfold-backward implementation now accepts broadcast batch axes, actual partial
frame counts, strided windows and non-overlap copy broadcasting. It validates
the overlapping window width; upstream's internal CPU kernel indexes that
width directly and does not expand a malformed singleton window.
An initially incorrect new regression was corrected to use a proper expanded
zero-stride window. Original upstream test code, samples, tolerances and skips
were not changed. Diagnostic failures are retained.

Strict native regressions: 1,650 passed each on RTX 2080 Ti, V100 and Lavapipe.
Ordinary repository: RTX and V100 3,394 passed; Lavapipe 3,393 passed, one skipped.
Full strict V100: 3,383 passed and the same 11 unchanged explicit FP64 failures.
All six complex64-only OpInfo entries were independently checked with their 31
original CPU-generated complex64 samples, avoiding unordered dtype selection;
all pass at the stated tolerances with zero fallback attempts. This supplements
rather than rewrites the original upstream audit. Ten original composite
backward/forward-AD compliance checks also pass.

Matched RTX CUDA/Vulkan controls use identical serialized inputs and PyTorch
source commits. Individual raw outcomes and tolerances are retained, including
any CUDA-unsupported cases. These checks establish correctness, not performance.
The previous 76 upstream expected-failure cases were also mapped to the
original CPU and CUDA test instances, with all bodies, samples and decorators
unchanged. Both controls report 72 expected failures and four skips; they are
not passes. CUDA LU controls cannot execute with this installed binary's kernel
support, and CUDA lacks narrow_copy.out. Four compare-CPU controls are skipped
on CUDA upstream. Full raw traces and these limitations are retained.

GPU dispatch depends on dtype, geometry and strides, with no model-specific
routing. The final full immutable upstream rerun remains pending at this commit.
FP16/BF16/FP64, other precision variants, AMP and multi-GPU remain deferred;
AMD and Intel hardware have not been tested.
