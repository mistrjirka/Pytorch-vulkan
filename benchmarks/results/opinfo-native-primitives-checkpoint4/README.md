# General native primitives, checkpoint 4

The latest compiled code passes 580 strict targeted tests each on RTX 2080 Ti,
Tesla V100, and Lavapipe. Strict tests forbid CPU fallback. The full RTX repository
suite passes 2,324 tests; it also includes deliberate fallback tests.

The build started from main at 5f9d488 with local changes. The build manifest
records the exact extension and source hashes. These hashes identify the tested
code; the parent commit alone does not.

The focused13 run covers 1,209 unchanged upstream cases. Later focused runs
recheck fixes; summary.json retains their source identities and latest per-case
results. Raw errors remain in every results file. This combination is a debugging
record across successive builds, not a full audit of one clean commit. The full
previous 8,696-case audit remains under upstream-opinfo-fixes-5f9d488.

Changes include exact integer division/remainders/powers and matrix products;
gamma functions and gradients; scans, unique rows and consecutive unique;
adaptive max pooling/unpooling; reusable activations and gradients; complex
construction/reduction/sign; fixed-size nonzero and flattened indexing;
determinants, symmetric eigensystems, Householder QR, and matrix polar factors.

Correctness fixes cover determinant saved-factor transpose conventions, Boolean
products, gather/output overlap, empty indices, negative index_fill indices,
zero-tensor reads, full-range FP32 trigonometry, and cancellation in reductions.
The captured masked-product forward-AD failure passed CPU/CUDA with the same
input and tangent; it now passes Vulkan after compensated reduction. Diagnostic
scripts record inputs without changing the upstream bodies or tolerances.

Cephes-derived gamma routines retain their BSD license in third_party. Other
new shaders are original implementations. The generic QR/eigen/integer matrix
kernels and quadratic unique sort are correctness paths, with no CUDA speed
parity claim. AMD and Intel hardware have not been tested. FP16/BF16, FP64, and
multi-GPU remain outside this checkpoint's scope. Remaining raw failures and
other operators mean the overall completion condition is not met.
