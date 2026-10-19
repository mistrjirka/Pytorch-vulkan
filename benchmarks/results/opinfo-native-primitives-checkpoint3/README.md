# Native primitive correctness checkpoint 3

Repository suite on RTX: 2,199 passed. Strict targeted tests with CPU fallback
forbidden: 480 passed on each of RTX 2080 Ti, V100, and Lavapipe. These are
regression results, not proof that the complete upstream suite passes.
The build source manifest identifies the compiled code and runtime shaders.

New generic paths include integer scatter reductions, index copy/fill,
masked selection/scatter, complex copy and magnitude helpers, extrema,
Cholesky/inverse/LU unpacking, cumulative reductions, unique/sort, spatial
NLL loss, and FP32 remainder and binary math. Several paths prioritize
correctness and use serial reductions or quadratic sorting; these results
make no speed-parity claim. Integer rounded division and some other integer
math variants remain explicitly unsupported.

The unchanged 457-case focused upstream subset improved from 355 passes and
90 errors to 443 passes and two errors in the preceding build. Both remaining
errors arise while constructing FP64 test fixtures. Raw skips and unexpected
successes are retained; the two errors are not counted as passes.

Extreme-remainder checks use exact comparisons over 30,000 random FP32 bit
patterns plus IEEE edge cases. The archived CUDA control predates the final
subnormal fixes. It shows that CUDA agrees with the FP64 remainder reference
on all 30,000 pairs, while CPU vectorized FP32 fmod/remainder differs on 3,725.
For floor division, CPU and CUDA agree on all 30,000 pairs. The final strict
tests compare fmod/remainder against that independent reference and floor
division against CPU, with zero tolerance. All pass on all three devices.
No upstream test body, sample, or tolerance was modified.
