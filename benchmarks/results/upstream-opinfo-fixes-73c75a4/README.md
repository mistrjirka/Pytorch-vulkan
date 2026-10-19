# Clean strict upstream audit at 73c75a4

All 8,696 selected upstream cases completed, with no worker restarts: 6,648
passed, 1,498 errors, 444 skipped, 96 expected failures, 10 unexpected successes.
The previous 5f9d488 audit had 5,836 passes and 2,310 errors. These are case counts,
not distinct operators, samples, or a claim of complete PyTorch coverage.

The strict gate stopped CPU fallbacks in 1,319 cases. Several FP32-tagged cases
explicitly request deferred FP16/BF16/FP64/complex128 fixtures. Raw errors remain
errors here; summary categories do not count these as passes. The one numerical
comparison category is alpha-dropout: independent CPU/CUDA controls recorded
previously show different random masks for equal seeds. This remains a raw error.

The source and build identities are retained in inventory.json.gz. The tested
main tree was clean. No upstream samples, bodies, or tolerances were changed.
The completion condition remains unmet. AMD/Intel hardware is unverified.
