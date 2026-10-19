# Full upstream correctness audit at 5f9d488

Unmodified PyTorch test_ops.py, pinned source and backend hashes in the inventory.
All 8,696 selected cases completed: 5,836 passed, 2,310 errors, 444 skipped,
99 expected failures, and 7 unexpected successes. Of the errors, 2,052 first
stop at forbidden CPU fallback. Completion of the runner is not an all-pass result.

The one numerical-comparison failure is alpha_dropout, whose CPU and device RNG
sequences differ; the prior CUDA control is retained in the round6 evidence.
No upstream test bodies, samples, tolerances, or result classifications changed.
FP32-tagged tests can still invoke deferred FP64 helpers.
This selected suite excludes 24,078 generated cases and is not all PyTorch coverage.
