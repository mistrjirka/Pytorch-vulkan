# Full immutable upstream audit at af259ae

All 8,696 original selected cases executed once across six disjoint shards on
RTX and V100. Raw results: 7,834 passed, 310 errors, 446 skipped, 76 expected
failures and 30 unexpected successes. No passing case used or attempted CPU
fallback. All original IDs were present exactly once; source/build identity
was verified before and after. Original samples, tolerances and skips are intact.

The original error-only completion gate failed on complex64 SiLU. A subsequent
all-status fallback check also found two narrow_copy.out fallback attempts
inside upstream expected failures. Checkpoint 12 fixes all three paths.
The other raw errors were 268 deferred-precision cases, 40 upstream CUDA/ROCm-only
Jiterator gates, and one CPU/GPU random-mask equality case also known upstream
on CUDA/XPU. No error, skip, xfail, xpass or later unvisited sample is a pass.
Checkpoint 12 fixes the SiLU issue and additional complex64 intermediate paths;
its later full rerun must be assessed independently.
