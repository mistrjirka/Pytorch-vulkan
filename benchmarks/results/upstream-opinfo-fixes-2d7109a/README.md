# Upstream correctness checkpoint: 2d7109a

This is the full, unchanged first-suite selection from PyTorch commit
`5c4886908584029761b579af026dcfb627c84070`, on an RTX 2080 Ti.
The 8,696 cases produced 4,661 passes, 3,485 errors, 444 skips,
102 expected failures, and four unexpected successes. This is not a completed
correctness or no-fallback gate.

The audit started from b33207e with uncommitted changes. Every recorded backend
source hash was checked against published commit 2d7109a and matches it; see
source-identity.json. The archived inventory retains the original identity.
A further 24,078 generated cases were outside the first-suite selection.

Of the errors, 3,222 first stopped at a forbidden CPU fallback. The summary
includes inline fallback paths as well as dispatcher fallbacks. Removing one
missing operation can expose another failure later in the same case.

The single raw numerical-comparison error is alpha dropout's exact CPU/device
sample comparison. The included CUDA control also produces different CPU/CUDA
masks with the same seed (14 of 25 elements differ), with matching possible
output values. Independent Vulkan distribution and derivative checks were
subsequently added and passed on RTX, V100, and Lavapipe. The original upstream
error remains in these results; it has not been relabeled as a pass.

The division/truncation and masked zero-order norm comparison failures found
in the preceding run pass unchanged in this run. Repository verification for
this source checkpoint: 2,063 RTX tests and 344 strict targeted checks on each
of V100 and Lavapipe. Newer local changes are not represented by these counts.
