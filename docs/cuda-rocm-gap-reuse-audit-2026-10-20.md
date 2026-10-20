# CUDA / ROCm gap audit — 20 October 2026

**Baseline:** 9148518, PyTorch revision 5c4886908584029761b579af026dcfb627c84070.
CUDA and Vulkan were profiled on the **same RTX 2080 Ti**, with FP32/TF32
disabled and GPU synchronization. RTX 3090 not used. No active ROCm host
was available (Connect_to_mini exposed only hub). ROCm statements here
are public architecture research, **not fresh AMD benchmark measurements**.

## The three main problems are fundamentally different

1. **Missing semantic coverage of an optimization we already built.**
   The large network suite uses standard torch.optim.AdamW with
   amsgrad=True and foreach=False. Our opt-in native Vulkan fused AdamW
   supports neither AMSGrad's maximum-second-moment state nor all
   optional PyTorch optimizer features. Using it without extending
   semantics would make the benchmark unfair.
2. **Partial LSTM fusion.** Vulkan fuses _thnn_fused_lstm_cell and its
   backward but still launches many separate small GEMMs, packs and
   pointwise kernels across an LSTM sequence. CUDA/cuDNN processes
   recurrence more coherently using RNN_blockPersist forward/backward.
3. **Inefficient attention kernel.** Vulkan uses a real native
   fused FP32 forward/backward attention shader, but it loops across
   keys/queries with repeated dot products and workgroup barriers.
   CUDA uses tiled CUTLASS SM75 memory-efficient attention kernels.

GPU residency, deferred submission, bounded descriptor LRU, optimizer
**primitive** fusion, and the direct-output unary/GEMM optimization
are already active where eligible. Re-enabling them does not repair
these three independent shortcomings.

## Experiment A — LSTM sequence fusion

Identical two-layer bidirectional LSTM (input 2×4×5, hidden size 7,
FP32, eval mode). 15 warmups, 80 synchronized forward iterations,
two alternating runs per configuration:

| CUDA cuDNN | Run 1 | Run 2 |
|---|---:|---:|
| Enabled | **0.2030 ms** | **0.2037 ms** |
| Disabled | 0.8136 ms | 0.8102 ms |

Output parity max absolute difference across backend selection was
under 1e-6. **The sequence fusion is approximately 4× faster.**
CUDA profiler identifies RNN_blockPersist_fp_LSTM and
RNN_blockPersist_bp_LSTM kernels.

The Vulkan recurrent model uses **474 GPU dispatches per complete
training step** in the matched full-network suite. Its own LSTM cell
kernel only accounts for ~0.092 ms of ~2.728 ms instrumented GPU
shader time, versus mm ~0.535 ms, optimizer_pointwise ~0.393 ms,
native_pack ~0.309 ms, loss_mutate ~0.242 ms. The remaining deficit
cannot be resolved by tuning the per-cell kernel alone; it requires
reducing sequence-level launches and intermediate memory movement.

## Experiment B — attention backward algorithms

Reference Transformer: batch 16, 128 tokens, four 256-wide
TransformerEncoder layers and eight heads; three profiled training
steps after warmup:

| Vulkan shader family | Medium ms/step | Reference ms/step |
|---|---:|---:|
| sdpa_training | 21.39 | **70.86** |
| Sum of all Vulkan shader timings | 33.12 | **92.38** |
| SDPA fraction | 65% | **77%** |

CUDA profiler for reference Transformer uses CUTLASS FP32
memory-efficient attention kernels sm75, with ~0.380 ms
forward and ~1.439 ms backward **summed across four attention
calls each per step**. CUDA's complete profiler GPU-kernel sum was
~12.20 ms/step. Vulkan shader timestamp mode runs eager, whereas
the primary matched wall suite uses deferred submissions: these
diagnostic sums identify dominant kernels but should not be
subtracted from whole-step wall times or treated as exact ratios.

Independent standalone SDPA forward+backward experiment:
shape [16,8,128,32], dropout 0, FP32, 50 timed
iterations × 2 processes, strict CPU output and all Q/K/V gradient
parity:

| Algorithm | Run medians |
|---|---|
| CUDA automatic memory-efficient attention | **0.674 / 0.680 ms** |
| CUDA forced mathematical SDPA | 1.076 / 1.090 ms |
| Vulkan fused sdpa_training | **19.456 / 19.359 ms** |

CUDA optimized SDPA is only ~1.6× faster than its own math
variant here. Vulkan remains ~18× slower even compared with CUDA
math; *lack of a fused operator is not the root cause*.

Source review of csrc/shaders/sdpa_training.comp:
- Forward traverses query/key scores twice, once for log-sum-exp,
  once for output, and repeatedly barriers between reductions.
- Backward serializes the opposite sequence dimension for each
  query/key workgroup, recalculates score terms and repeatedly
  computes row-wise delta = dot(grad_output, output).
- The shader lacks tile-level Q/K/V reuse between neighboring
  queries and keys. General tile-level reductions, precomputing
  per-row delta, and optimized backward block partitioning would
  help without introducing model-name dispatch.

## Experiment C — previously implemented AdamW fusion not applied

The full-network suite uses **AMSGrad**, excluded from the previously
implemented Vulkan native fused AdamW. Matched optimizer-only
parameters: 50 heterogeneous MyNet parameter tensors, 2,420,547
FP32 elements, fixed gradient tensors, 12 warmups + 75 timed
synchronized updates × 2 processes:

| GPU + implementation | Process medians |
|---|---|
| CUDA standard per-tensor AdamW, AMSGrad | 4.911 / 4.956 ms |
| CUDA fused AdamW, **AMSGrad maintained** | **0.426 / 0.428 ms** |
| Vulkan standard AdamW, AMSGrad | 16.261 / 11.364 ms |

This demonstrates **~11.5× acceleration from CUDA fusion**
with the optimizer semantics unchanged. An additional 40-step
CUDA numerical check compared fused and unfused parameters,
exp_avg, exp_avg_sq, and max_exp_avg_sq; maximum absolute error
was **5.07e-7**.

This is not a direct benchmark of Vulkan fused AMSGrad; that
implementation does not exist yet. The native Vulkan fused optimizer
already handles ordinary AdamW but is opt-in, lacks AMSGrad and
other features (maximize, mixed precision, graph capture, etc.).
To make it generally useful, extend it with the max-second-moment
buffer and exact bias correction, state_dict compatibility,
in-place tensor version behavior and numerical tests. Keep the
benchmark's optimizer choice explicit: do not silently replace
torch.optim.AdamW with a different-semantic custom optimizer.

## Other already-known gaps and new reuse candidates

- **topk:** order_statistics.cpp::topk_native fully sorts each
  axis and only then narrows to k. This wastes work when k is
  small relative to the axis. CUDA/HIP have dedicated selection
  paths. Need a capability/shape-safe partial-selection kernel.
- **gather.out/index_select.out:** indexing.cpp still computes a
  temporary functional result and copies into the provided
  destination, unlike the recent direct-output fixes to unary
  and GEMM. General alias-safe direct output can avoid copies.
- **Embedding validation:** the Transformer and embedding network
  observe four logical host reads of four bytes each per
  training step, to validate indices. Those require GPU-to-CPU
  synchronization even with zero registered CPU numerical fallback.
  We previously proposed batched device validation, but it has
  not been implemented. Preserve synchronous error semantics.
- **Convolution dispatch:** Vulkan has a bounded geometry/capability
  algorithm selector, but not the cuDNN/MIOpen style measured
  solution cache keyed by shape, dtype, layout and device.
  Target forward, input gradient and weight gradient separately;
  retain generic capability-checked fallbacks and memory limits.
- **Native optimizer primitives:** the latest direct-destination
  fusion and bounded descriptor cache *are already applied*; do
  not claim more improvements simply by turning them on again.

## ROCm: architecture facts, not physical benchmark results

ROCm PyTorch runs through HIP and supports architecture-dependent
SDPA backends such as AOTriton and Composable Kernel (CK). Backend
selection depends on availability, GPU, dtype, layout and
configuration; we did **not** verify which backend the user's
RX 7800 XT would select at these shapes.

MIOpen can benchmark, select and cache forward, backward-data
and backward-weight convolution solvers, including workspace
requirements and hardware-specific tuning parameters. The
underlying idea to reuse is **measure candidate algorithms once
during warmup, cache selection by validated geometry/capability**,
not copy model-specific source or use unlimited workspaces.

Primary references:
- https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW
- https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention
- https://docs.pytorch.org/docs/stable/notes/hip.html
- https://docs.pytorch.org/docs/stable/backends.html
- https://rocm.blogs.amd.com/software-tools-optimization/comfyui-fa-backends/README.html
- https://rocm.docs.amd.com/projects/MIOpen/en/latest/how-to/find-and-immediate.html
- https://rocmdocs.amd.com/projects/MIOpen/en/latest/conceptual/perfdb.html

## Reproduction and interpretation

Reproduction code:
- benchmarks/probe_cuda_cudnn_lstm_ab.py
- benchmarks/probe_sdpa_backend_selection.py
- benchmarks/profile_optimizer_only.py --optimizer adamw --amsgrad
  with optional --fused on CUDA
- benchmarks/check_cuda_amsgrad_fused_parity.py
- benchmarks/run_network_profiles.py for existing stage/GPU profiles

Machine-readable evidence:
- benchmarks/results/native-backend-gap-audit-2026-10-20/summary.json
- benchmarks/results/native-backend-gap-audit-2026-10-20/raw-experiments.jsonl.gz

The raw records retain per-run latency samples, data source names,
CUDA kernels and Vulkan shader timings. Distinguish measured
control-plane wall timing from instrumented GPU time.

## Priorities

1. Add native Vulkan **fused AMSGrad AdamW** with complete state
   behavior and test variants; keep it opt-in and compare matched
   semantics with standard AdamW and CUDA. Lowest implementation
   risk and a clear already developed kernel foundation.
2. Tile/restructure **attention forward+backward**, especially
   reused per-query statistics and reduction barriers; largest
   measured full-Transformer GPU shader deficit.
3. General LSTM **sequence-level** operation scheduling/fusion,
   reducing per-timestep GEMMs and packing.
4. Optimize indexed out-variants and top-k using the already
   established direct-output and specialization policy.
5. Repeat profile on actual RDNA3 ROCm once the AMD computer
   reconnects; do not reuse old revision-specific ratios as if new.
