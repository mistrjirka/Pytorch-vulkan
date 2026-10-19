# General native primitives, checkpoint 6

725 strict regression checks passed on RTX 2080 Ti, Tesla V100 and Lavapipe;
2,469 repository tests passed on RTX. Focused31 combines the affected upstream
cases in one fresh run, using unmodified pinned test bodies and tolerances.
Counts and exact source/build identities are in summary.json and the inventories.
Skips, expected failures and unexpected successes are separate from passes.
Earlier focused25-30 runs preserve the debugging history, including failures.
Focused25 accidentally selected broad noncontiguous families and is not a clean
affected-operator result. This is not a new full 8,696-case audit.

Adds native order statistics and mode, isin, repeat-interleave indices, renorm,
addr, smooth-L1/Huber/BCE loss and backward, batch-normalization entry points,
and general special functions and orthogonal polynomials. Fixes normalization
empty inputs, copy-on-write mutation, output aliasing, reduction contracts,
complex scalar masked-fill validation and FP32 error-message formatting.
The formatter follows the existing PyTorch policy for devices without FP64;
explicit FP64 arithmetic is still unsupported.

Precision controls compared identical failing inputs against CPU and CUDA.
Portable FP32 expansion arithmetic, integer-based subnormal classification,
and a coefficient table fix cover errors exposed on NVIDIA and Lavapipe.
After validation, trailing whitespace was removed from the special-function
include; recompilation produced byte-identical SPIR-V. The before/after source
hashes and equality check are in whitespace-equivalence.json.
The saved diagnostic logs precede the final version and are not final results.
PyTorch/Cephes/Faddeeva and Sun/OpenLibm license notices are in third_party.
No model-specific shape rules were added. Polynomial degrees beyond the shader's
signed integer representation still need explicit general-range handling.

FP16/BF16/FP64, AMP and multi-GPU remain deferred. AMD/Intel are untested.
This checkpoint establishes neither CUDA speed parity nor overall completion.
Remaining operators and contracts must still pass the full strict audit.
