# Embedding, sparse storage and sparse matrix correctness, checkpoint 10

1,425 strict regression checks pass on RTX 2080 Ti, Tesla V100 and Lavapipe.
The ordinary RTX repository run passes 3,169 tests. The full strict V100 run has
3,158 passes and the same 11 explicit-FP64 raw failures as checkpoint 9. Their
IDs and test-source hashes are verified unchanged; no failure became a pass/skip.

Adds GPU embedding renormalization with strided alias-preserving writes,
independent handling of duplicate/negative indices, zero and infinite norm
orders, non-finite values and empty-feature validation. Enables sparse
embedding-bag gradients through the existing generic sparse embedding backward.

Adds exact boolean, signed integer and complex64 COO value storage, duplicate
coalescing, dense conversion and masking. Large Int64 values and wraparound are
checked without converting their values to FP32. Subword output uses disjoint
atomic masks. Extends compressed storage to CSR, CSC, BSR and BSC, including
batches, dense blocks, transfers and sparse-axis transpose, using shared GPU
primitives. Adds real FP32 CSR sum/mean/extrema matrix reductions, saved indices,
backward gradients and sampled multiplication. Sparse index validation executes
on GPU and downloads only its status scalar. Several general composed paths
still use dense GPU workspaces and need performance optimization.

Corrects the pinned ATen invalid-input contracts for concatenation, identity
factories, extrema, triangular operations, boolean negation, pooling, cross
products, embedding, where and uniform RNG. Generic out handling now enforces
aminmax dtype matching and concatenation's full alias restrictions.

Focused55: 255 passed, 16 upstream skips, 14 errors and two xfails. Every passed
case has zero CPU fallback calls or attempts. The 14 errors explicitly require
Half/Double fixtures: cat, complex, dot/vdot, convolutions, scatter and all five
sparse-mul layouts. Generator hashes and raw details are retained, and those
errors are not counted as passes. Focused54 adds seven sparse passes/five skips.
The raw intermediate focused52/53 runs remain preserved.

Matched controls use identical cached tensors, the same RTX and the pinned
PyTorch commit. Sampled multiplication and all gradients match CPU within the
original tolerance on both Vulkan and CUDA. Vulkan also matches the CPU
renormalization controls; CUDA differs from CPU for zero, negative and infinite
norm orders on this extreme-value fixture. The raw disagreements are retained,
not silently treated as Vulkan requirements. Installed CUDA does not implement
sparse.mm with mean reduction; Vulkan is compared with CPU for that operation.
These are correctness controls, not evidence of CUDA performance parity.

Local test corrections are preserved in diagnostics: dense-to-sparse with zero
sparse dimensions is rejected by CPU for non-scalar input (valid zero-sparse-dim
constructors are tested separately); batched compressed conversion requires the
same nnz per batch, so valid fixtures now guarantee a common nonzero pattern;
sparse-gradient references and GPU leaves are detached to keep their graphs
independent. No upstream bodies, samples, tolerances or skips were changed.

FP16/BF16/FP64, AMP and multi-GPU remain deferred. AMD/Intel are untested.
Int64 average pooling and real nonsymmetric eigenvalue coverage remain open.
Overall correctness and numerical-fallback completion is not claimed.
