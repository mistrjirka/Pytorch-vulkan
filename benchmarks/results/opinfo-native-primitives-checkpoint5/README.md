# General native primitives, checkpoint 5

619 strict tests passed on RTX 2080 Ti, Tesla V100, and Lavapipe; 2,363 repository
tests passed on RTX. Focused24 is one clean 221-case upstream run of the tested
sources; 219 passed and 2 skipped. Earlier focused22/23 raw results retain the
noncontiguous sorter lifetime failure and its verification. Every source/build
identity is preserved. This is not a new full 8,696-case audit.

Adds device binary search (searchsorted/bucketize, exact signed integer ordering,
optional sorter, batched boundaries), atan2/angle, frac/sinc/entr/xlog1py, frexp,
and reusable 1D/2D/3D nearest/exact-nearest and 1D linear resampling with gradients
and output adapters. Correctness fixes preserve subnormal input classification,
normalize tiny atan2 ratios and entropy, and retain packed sorter storage through
dispatch. Original implementations; no third-party kernels were copied.

FP16/BF16/FP64 and multi-GPU remain deferred. AMD/Intel are untested. These changes
do not establish CUDA performance parity or fulfill the overall completion condition.
