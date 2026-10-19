# Full immutable upstream audit at a6c59e9

8,696 selected cases from 16 first-suite families, with unmodified pinned source.
24,078 generated cases are outside this first-suite selection.
Raw: 7,748 passed, 396 errors, 446 skips, 76 xfails and 30 xpasses.
No timeouts or worker restarts, with a 300-second per-case watchdog.
257 cases first hit forbidden CPU fallback, compared with 304 at 7f12ebe.

These include explicitly deferred dtypes and actual coverage gaps. Cases stop
at their first failing sample; later samples are not assumed to pass. Checkpoint
10 changes were implemented after this immutable main audit and are verified
separately. Raw errors and outcomes are not overwritten by focused rechecks.
