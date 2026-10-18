# FP32 single-GPU coverage

The target is general FP32 compute on one Vulkan device. FP16/BF16/AMP and
multi-GPU execution are deferred. Integer indices and boolean masks are included
because FP32 networks need them.

The operator and runtime gaps from the
[CUDA universality audit](cuda-universality-audit-b80a1f6.md) are implemented and
verified within the contracts in the
[completion report](fp32-coverage-completion.md). That report records the test
matrix, strict fallback/host-read checks, feature-network training, matched CUDA
measurements and remaining API/performance boundaries.

Completed work includes general Conv3D/transposed convolution, LSTM primitives,
independent RNG/dropout, indexing and dense embedding gradients, embedding bags,
sort/top-k/scans/variance, FFT and dense solve, fused training attention,
ordinary AdamW/AMSGrad primitives, additional spatial operators, sparse COO,
logical streams/events and default Inductor integration. The kernels are
selected by operator contracts, not model identity.

The earlier contract fixes at `f66bcb5` remain covered: mixed-dtype/device checks,
exact large embedding indices, allocation layouts and invalid-device rejection,
view-preserving zeroing, and exact integer/boolean copies. Historical validation
for that batch is retained under
[`benchmarks/results/fp32-contracts`](../benchmarks/results/fp32-contracts/).

Future work is performance-guided specialization/fusion, additional API families,
AMD/Intel execution testing, and the explicitly deferred precision/multi-device
features. GPU-native execution must continue to be distinguished from CPU-backed
execution, and each speed claim needs a matched workload comparison.
