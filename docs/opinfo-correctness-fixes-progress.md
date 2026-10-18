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
