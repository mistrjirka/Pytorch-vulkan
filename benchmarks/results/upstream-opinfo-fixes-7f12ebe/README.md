# Full pinned OpInfo audit at 7f12ebe

8,696 selected cases from 16 first-suite families; 24,078 generated cases outside
that selection. Unmodified upstream source SHA256 is recorded in the inventory.
Raw results: 7,707 passed, 437 errors, 446 skips, 77 xfails and 29 xpasses.
No timeouts or worker restarts; per-case watchdog is 300 seconds.

304 cases first hit forbidden CPU fallback, down from 421 in the prior full
audit. This includes deferred dtypes as well as actual native coverage gaps.
Cases stop at their first failing sample. Skips, xfails and xpasses are separate
from passes. Checkpoint 9 changes were made after this immutable main audit;
its focused rechecks do not replace these raw results or establish a new full
coverage total. Eigenvalue, sparse and other residual gaps remain open.
