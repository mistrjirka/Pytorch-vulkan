# FP32 coverage completion

> Latest native correctness evidence: [checkpoint 13](../benchmarks/results/opinfo-native-primitives-checkpoint13/README.md).
> The final full upstream recheck remains pending; this page is the earlier feature milestone.

> **Subsequent upstream audit:** [OpInfo testing at this revision](upstream-opinfo-audit-863fc0b.md)
> found missing dispatches and an `addmm` scaling bug beyond these selected tests.
> This report records the narrower feature milestone, not general FP32 correctness.

This completes the single-device FP32 gaps identified by the
[CUDA universality audit](cuda-universality-audit-b80a1f6.md), within the operator
contracts below. FP16/BF16/AMP and multi-GPU remain deferred. This is a tested
milestone, not a claim to implement every CUDA/PyTorch operation or match CUDA
performance.

## Implementations and evidence

All implementations dispatch by operator geometry, dtype, strides and device
capabilities. No network names, model-specific shapes or U-Net branches are used.
New shader sources are original project code; they use Vulkan 1.0 operations,
including pairs of uint32 words for exact integer indices and arithmetic.

| Audited gap | Native implementation | Verification |
|---|---|---|
| Conv3D and transposed convolution | General grouped, strided, padded and dilated FP32 convolution; forward, input, weight and bias gradients; transposed 1D/2D/3D | `test_complete_spatial_recurrent.py`, volume autoencoder |
| LSTM | Fused cell and gate backward primitives composed with existing GEMM; multilayer, bidirectional and projected cases | Spatial/recurrent tests and recurrent regressor |
| Dropout/device RNG | Philox4x32-10; independent generators, seed/offset, clone/save/restore; FP32 uniform/normal/Bernoulli, exact int32/int64 random ranges; dropout masks and gradients | `test_complete_random.py`, known Philox vector, statistics, aliases and CPU-RNG independence |
| Indexing and embedding | Gather, scatter/scatter-add, index-select/index-add, dense embedding backward; sum/mean/max embedding bags with per-sample-weight gradients | `test_complete_indexing.py`, training-primitives tests, retrieval network |
| Sorting/scans/variance | Stable FP32 sort, top-k, cumulative sum, variance; exact int32/int64 add/multiply/sum and boolean sum | Indexing/integer tests and retrieval network |
| FFT | FP32/complex64 real/complex transforms, multidimensional axes, normalization, Hermitian reconstruction and gradients; complex arithmetic/copies needed by spectral training | `test_complete_fft_linalg.py`, learned spectral filter |
| Dense solve | Batched FP32 partial-pivot LU, left/right/adjoint solve, error reporting and autograd; native identity-matrix factories | FFT/linalg tests and learned solve regressor |
| Attention | Fused forward and backward; causal/bool/additive masks, mask gradients, dropout, broadcast batches, GQA and fully masked rows | `test_complete_attention.py`, causal-attention network, compiled attention |
| Loss | FP32 MSE forward and both input gradients, reductions, strided output and broadcasting | `test_complete_loss.py` and original audit |
| Ordinary optimizers | Reusable lerp, addcmul, addcdiv and minimum/maximum; standard AdamW and AMSGrad | Training-primitives tests; two CPU-matched steps for all seven new networks, including moments |
| Extra spatial operators | Average/max/adaptive 3D pooling; antialiased bilinear/bicubic and trilinear resize; reflection/replication padding; 2D/3D grid sampling and gradients | `test_complete_spatial_extra.py`, volume and spatial-transformer networks |
| Sparse COO | Real SparsePrivateUse1 storage, dense round trip, hybrid blocks, duplicate coalescing, invariant checks, shape inference and gradients | `test_complete_sparse.py` |
| Default compile | Dynamo/AOTAutograd/Inductor external-kernel integration; device-specific random factories, cached execution and backward | `test_complete_runtime.py`: matmul/activation, 3D convolution, fused attention and dropout |
| Streams/events | Logical streams, contexts, event recording/query/wait/synchronize, GPU timestamp elapsed time and in-flight allocation retention | Runtime ordering and lifetime tests |

`benchmarks/feature_networks.py` adds seven stock-PyTorch feature networks to the
existing five: volume autoencoder, spatial transformer, recurrent regressor,
retrieval regressor, spectral filter, solve regressor and causal-attention block.
They exercise these features together, rather than relying only on isolated
operator probes. `tests/test_complete_networks.py` checks outputs, input and
parameter gradients, parameters and all AMSGrad moment tensors against CPU.

## What “no CPU fallback” means

`TORCH_VULKAN_FORBID_CPU_FALLBACK=1` raises on generic ATen CPU fallback.
`torch_vulkan.execution_stats()` also counts explicit host materializations,
which catches CPU paths inside registered Vulkan operators. Reset counters
with `reset_execution_stats()` after setup/compilation and inspect them before
explicit result downloads. Counts are process-wide and host byte counts include
the enclosing allocation, not just the requested view.

The network residency checks allow only four-byte validation/status reads.
General index validation, probability validation, sparse nnz counts and LU
pivot validation synchronize to report errors or allocate dynamic outputs.
LU error checks read four bytes per batch matrix; inferred sparse sizes read
up to eight int32 dimensions. These are control/status reads, not CPU execution
of the numeric operator. Output/gradient/weight data stays on the GPU in
resident mode. Eager mode deliberately transfers intermediates through host
storage, while still executing the new numeric kernels on Vulkan.

These checks caught and fixed partial-storage corruption, a conjugate-copy bug
that produced wrong spectral gradients, hidden boolean-mask conversion copies,
and an empty-input eager synchronization error.

## Validation matrix

PyTorch 2.14.1, source `5c4886908584029761b579af026dcfb627c84070`; CPU reference
wheel `+cpu`, matched CUDA wheel `+cu130`. Results, execution counters, input and
weight hashes, shader/source hashes and extension provenance are under
[`benchmarks/results/fp32-completion`](../benchmarks/results/fp32-completion/).

| Mode | Full regression | Strict completion suite |
|---|---|---|
| RTX 2080 Ti, resident/deferred | 1744 passed, 1 warning in 26.95s | 430 passed, 1 warning in 8.37s |
| Tesla V100, resident/deferred | 1744 passed, 1 warning in 23.05s | 430 passed, 1 warning in 7.19s |
| Lavapipe, resident/deferred | 1743 passed, 1 skipped, 1 warning in 22.77s | 430 passed, 1 warning in 7.00s |
| RTX 2080 Ti, eager | 1741 passed, 3 skipped, 1 warning in 24.50s | 430 passed, 1 warning in 7.85s |

The MSE implementation also removes the three earlier functional-loss
output-resize warnings. The projected LSTM CPU reference emits an expected
oneDNN-support warning. Kompute reports
that no requested validation layer is installed in this environment. AMD and
Intel hardware have not been tested; Lavapipe is software portability evidence,
not a substitute for those devices.

Reproduce the new strict checks with:

```sh
python scripts/compile_coverage.py
python -m pip install -e . --no-build-isolation --no-deps
TORCH_VULKAN_GPU_RESIDENT=1 TORCH_VULKAN_DEFER_SUBMISSIONS=1 \
TORCH_VULKAN_FORBID_CPU_FALLBACK=1 python -m pytest -q tests/test_complete*.py
```

The compile script rebuilds each new SPIR-V module and runs `spirv-val` with a
Vulkan 1.0 target. The full regression command is `python -m pytest -q tests`
with the chosen execution-mode/device variables.

The original 39-case audit now passes **all 32 in-scope Vulkan cases** with
zero generic CPU fallback and at most four-byte status reads. The seven remaining
failures are FP16/BF16/FP64 add/matmul and complex64 matmul, outside this milestone.
The matched CUDA reference passes all 39. RNG independence, device seeding,
independent generators, LSTM, ordinary AdamW and second derivatives also pass;
autocast remains deferred.

## Matched CUDA measurements

| Training workload | Vulkan wall ms | CUDA wall ms | Ratio | Vulkan GPU ms | CUDA GPU ms |
|---|---:|---:|---:|---:|---:|
| volume autoencoder | 3.226 | 1.566 | 2.06× | 1.425 | 0.240 |
| spatial transformer | 1.859 | 1.617 | 1.15× | 0.522 | 0.120 |
| recurrent regressor | 31.094 | 6.516 | 4.77× | 3.506 | 0.447 |
| retrieval regressor | 9.814 | 2.221 | 4.42× | 0.597 | 0.163 |
| spectral filter | 3.640 | 1.787 | 2.04× | 0.564 | 0.102 |
| solve regressor | 6.124 | 2.038 | 3.01× | 0.744 | 0.142 |
| causal attention block | 5.312 | 2.436 | 2.18× | 1.130 | 0.170 |

Wall columns are medians of 50 samples across two runs. GPU columns are summed
device activity from separate instrumented runs, not a decomposition of wall time.

The largest wall-time gap is the recurrent network: **31.094 ms versus 6.516 ms**,
or **4.77×**. Its instrumented Vulkan work contains many small matrix products,
strided copies, additions and packing dispatches. The much smaller summed GPU
time suggests dispatch and host orchestration are the first optimization targets;
this is an inference across different scheduling modes, not an exact CPU-time
measurement. The volume network is more directly compute limited: generic
convolution accounts for roughly 0.9 ms of its 1.425 ms of Vulkan GPU work.

CUDA kernel totals exclude profiler user annotations such as the enclosing
`Optimizer.step` range, which would otherwise double-count its child kernels.

The timing harness uses identical initialized models and inputs, FP32 with TF32
disabled, four CPU threads, eight warmup steps and 25 measured steps per wall
run. Each complete step includes forward, loss, backward and ordinary
`torch.optim.AdamW(foreach=False, amsgrad=True)`. Two interleaved Vulkan/CUDA wall
runs use the same physical RTX 2080 Ti. Compilation and explicit result downloads
are excluded. These are small feature workloads, not full production models.

A separate instrumented run records Vulkan GPU dispatch/upload timestamps with
residency enabled and deferral disabled, and CUDA kernel durations using
`torch.profiler`. Instrumented per-dispatch fence waits change scheduling and
are not production wall-time measurements. Event elapsed time alone would
include idle gaps and cannot establish GPU busy time. Raw per-kernel timings
are included so optimization priorities can be checked against evidence.

## Boundaries that remain

- Most new numeric kernels support packable FP32 views up to rank 8, uint32
  addressing and the selected device's storage, dispatch, descriptor and shared
  memory limits. Complex64 support is the FFT/arithmetic subset, not complex GEMM.
- Linalg here means the audited dense solve/LU family, not QR, SVD or eigensolvers.
  Sparse support here means COO round trips/coalescing, not every sparse layout
  or sparse matrix multiplication. Embedding gradients are dense.
- Sort uses pairwise ranking; non-power-of-two FFT uses direct DFT; grid-sampling
  input gradients use deterministic gather accumulation. These are general
  correctness paths with known scaling costs, not CUDA-performance substitutes.
- Logical streams share one ordered Vulkan queue. They provide ordering and
  lifetime semantics, but no concurrent queues; stream query/synchronize are
  conservative across the queue. Priorities, IPC events and CUDA graphs are not
  implemented.
- Default Inductor schedules external native ATen kernels. It does not generate
  fused SPIR-V or promise compilation of arbitrary Python/PyTorch programs.
  The integration is pinned to this project's PyTorch 2.14.1 dependency. CPU and
  CUDA retain their own lowering functions. CPU compiled forward/backward was
  separately checked after importing the backend.
- Vulkan RNG is reproducible and independent, but does not promise bitwise CUDA
  RNG sequences. Other distributions and dtypes remain outside the tested set.
- FP16/BF16/AMP, wider FP64 compute and multi-GPU/distributed execution remain
  outside this FP32 single-device milestone.

### PyTorch accelerator-coexistence limitation

CUDA autograd in the same process as an imported PrivateUse1 backend hits
`opt_ready_stream && opt_parent_stream` in PyTorch 2.14.1. The failure reproduces
with the previous `f66bcb5` extension and this completion build; CUDA eager and
compiled training work in a fresh process without importing Vulkan. The pinned
[DeviceAccelerator.cpp](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/DeviceAccelerator.cpp)
prioritizes registered PrivateUse1 globally, while the
[autograd engine](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/torch/csrc/autograd/engine.cpp)
expects accelerator stream metadata for CUDA tensors. This is a pre-existing
framework integration limitation, not a newly introduced CUDA lowering change.
The matched comparison therefore uses separate processes. Vulkan execution with
the CUDA-enabled PyTorch wheel passes 39 runtime, network-training and RNG checks
in `vulkan-cuda-wheel.log`.
The baseline/current/without-import probes and the unsuccessful initial CUDA
compiler isolation log are retained; this coexistence case is not labeled passed.
