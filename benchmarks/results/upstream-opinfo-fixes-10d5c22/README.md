# Full pinned OpInfo audit at 10d5c22

8,696 selected cases from 16 first-suite families; 24,078 generated cases outside
that selection. The upstream source is unmodified and pinned by SHA256.
Raw: 7,586 passed, 552 errors, 444 skipped, 77 xfails, 29 xpasses, eight timeouts.
All eight timeouts passed isolated rechecks with a 180-second watchdog, producing
7,594 resolved passes. Original timeouts remain preserved. The initial default
watchdog was 30 seconds. Skips, xfails and xpasses are not counted as passes.

421 cases first hit forbidden CPU fallback, versus 1,319 in the previous full
audit. Cases stop at their first failing sample; this is neither complete
operator coverage nor evidence that every remaining error is in FP32 scope.
Checkpoint 8 changes were implemented after this immutable main audit and are
covered separately by focused rechecks, not silently substituted into these rows.
