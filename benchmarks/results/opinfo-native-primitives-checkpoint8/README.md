# General native primitives, checkpoint 8

1,110 strict regression checks passed on RTX 2080 Ti, Tesla V100 and Lavapipe.
The ordinary repository suite passed 2,854 tests on RTX. Running that entire
suite with CPU fallback forbidden exposed 20 failures (2,834 passed), including
high-rank/in-place coverage gaps and tests explicitly using deferred FP64.
Those raw failures are retained and remain open; ordinary-suite success does
not prove fallback elimination.

Focused47 is a fresh run of all affected unmodified upstream cases: 121 passed,
16 upstream skips and six errors. No fallback calls or attempts occurred.
The six errors are four Python reference triangular-index cases that explicitly
create FP64 arange, and two bincount cases with integer weights which ATen
promotes to FP64 output. They remain raw errors, not counted as passes.
Earlier focused39-46 preserve intermediate failures and fixes.

Adds lengths/offsets segment reductions with gradients, margin losses, ATen
hashing and the supporting UInt64 storage/copy/comparison/fill/bounded RNG,
bincount, histograms, triangular index factories, pairwise distances and their
gradients, fractional pooling with gradients, log-sigmoid output adapters and
the remaining CPU-flash ATen attention entry point through Vulkan training
kernels. Dispatch uses operator semantics and device limits without model rules.

The valid matched RTX CUDA control uses the exact pinned PyTorch commit and
checks hash values, UInt64 storage/fill/bounded RNG, empty pairwise gradients
and histc boundaries. CPU/CUDA UInt64 ordering is unsupported; Vulkan ordering
is independently checked against Python integers. An earlier V100 CUDA control
is retained as INVALID: the installed cu130 wheel lacks sm70 kernels.

Pinned CPU pdist backward crashes for empty inputs with p<1, independently
reproduced in a CPU-only process. The regression uses the zero derivative for
empty/zero-feature cases; CUDA confirms the empty gradient. CPU triangular
index count arithmetic also overflows at extreme signed offsets, so the local
boundary regression uses the exact integer inequality as its oracle. Upstream
test bodies, samples, tolerances and skips have not been changed.

FP16/BF16/FP64, AMP and multi-GPU remain deferred. AMD/Intel are untested.
These are correctness paths and still require matched performance profiling.
Remaining native gaps mean overall completion is not claimed.
