# Full immutable audit at d43d267

All 8,696 original selected cases executed exactly once: 7,830 passed, 314 errors,
446 skipped, 74 expected failures and 32 unexpected successes. The selected scope
gate passes: zero unresolved in-scope failures; successful and expected-failure
cases have zero fallback attempts. The 314 raw errors comprise 273 deferred
precision blockers, 40 CUDA/ROCm Jiterator gates and one CPU/GPU random-mask
comparison with matched CUDA evidence. Raw outcomes are preserved as recorded.
Unordered dtype selection in upstream warning tests can vary the precision
blocker counts between processes. No blocked later sample is assumed to pass.

This was not overall completion: a supplemental complex-power zero/nonfinite
probe subsequently found a real bug beyond those original samples. Checkpoint
13 fixes it and adds component-wise boundary regressions. Its new source must
pass the complete immutable audit again before the task is declared finished.
