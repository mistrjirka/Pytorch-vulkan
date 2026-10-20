# General Vulkan multi-tensor chunk scheduling — 20 October 2026

Baseline: commit 9d5c413. This is an optional-optimizer performance change,
not a replacement for standard PyTorch AdamW.

## Architecture

The existing native Vulkan fused AdamW/AMSGrad already combines weight
decay, first/second moment and parameter updates within each tensor.
However, it still emitted one compute dispatch *per parameter tensor*,
despite batching those dispatches in a single command sequence.

This change implements horizontal fusion: a single compute shader dispatch
updates chunks belonging to multiple independent parameter tensors.

- A reusable, **operator-independent** planner in
  csrc/ops/multitensor_plan.h partitions any list of workgroup counts into
  groups according to descriptor capacity, maximum grid size and a
  caller-supplied read/write buffer-conflict predicate.
- Each GPU workgroup uses cumulative chunk boundaries and maps to exactly
  one parameter tensor; groups contain arbitrary heterogeneous lengths.
  No padding to the largest tensor size, no concatenation buffer and no
  pack/unpack GPU kernels.
- Each tensor still uses its own parameter, gradient and moment buffers.
  Per-tensor bias corrections/step counters remain independent.
- Static descriptor shader variants for widths 4 and 8 require Vulkan 1.1,
  but no buffer-device-address or descriptor-indexing extensions.
  Device property checks select width 8 where supported, width 4 when
  necessary, and retain the old single-tensor path otherwise.
  Short tail groups use the narrower compatible variant.
- The planner refuses to execute aliased writes concurrently: if one
  candidate's output is another's input, those operations execute
  sequentially with existing deferred-command dependency handling.
- One generator, scripts/generate_adamw_multitensor.py, emits the
  ordinary AdamW and AMSGrad variants of each shader. Optimizer math
  remains identical, including AMSGrad NaN propagation.
- All existing Python optimizer defaults and PyTorch's standard AdamW
  behavior remain unchanged.

Process-start A/B controls:
- TORCH_VULKAN_ADAMW_DISABLE_MULTITENSOR=1 uses the original per-tensor
  dispatcher.
- TORCH_VULKAN_ADAMW_MULTITENSOR_WIDTH=4 restricts groups to four rather
  than the preferred eight (subject to physical device limits).

### Optimizer-only A/B measurements

Fifty heterogeneous FP32 parameter tensors, 2,420,547 elements,
AMSGrad enabled, static GPU-resident gradients.
15 warmups and 85 synchronized timed updates per fresh process;
three processes per device/configuration. Lower is better.

| GPU | Single-tensor dispatches | 4-wide | 8-wide |
|---|---:|---:|---:|
| RTX 2080 Ti | 0.985 ms | 0.756 ms | **0.741 ms** |
| Tesla V100 | 0.977 ms | 0.742 ms | **0.705 ms** |

The final 8-wide approach is ~1.33x faster on RTX and ~1.39x
on V100. Width-4 results come from an earlier but compatible
A/B build; the final 8-wide version additionally selects width 4
for short tail groups.
It reduces approximately 50 optimizer GPU dispatches to seven.
This is an improvement on top of the earlier fused AdamW AMSGrad
optimization, not a 1.31x speedup over standard PyTorch AdamW.

Prior native CUDA fused AMSGrad benchmark on the same RTX 2080 Ti
was ~0.427 ms. Vulkan 8-wide remains roughly 1.74x slower than
that historical CUDA measurement.

### Complete model-level A/B: mixed results

Same FP32 CPU-reference-validated model and optimizer semantics,
two separate processes per setting, 8 warmups and 20 timed steps.
Full forward + loss + backward + optimizer (training step):

| Network | Prior per-tensor path | Horizontal fusion |
|---|---:|---:|
| MLP | 3.414 ms | 3.724 ms |
| U-Net | 17.508 ms | 16.964 ms |
| Transformer | 49.120 ms | 49.933 ms |

These results were collected before the final narrow-tail-group
refinement, so model-level impact of that refinement was not independently
measured. The differences are relatively small, with some favoring the old
path, and each network has material run-to-run variability. The
reliable gain is in the isolated optimizer. **We cannot claim a
general full-network speedup from this two-process sample.**
The Transformer is still overwhelmingly limited by attention.

### Correctness

Final complete suites: 3904 tests passed on RTX 2080 Ti and V100;
3903 passed and one skipped on Lavapipe. All four generated SPIR-V
shaders passed spirv-val. New CPU reference tests cover ordinary AdamW and AMSGrad, 2 through 53
heterogeneously sized tensors, zero-length tensors, missing gradients
and varying bias-correction steps, state tensor parity, GPU dispatch
counts and deliberately aliased input/output states. The latter must
run sequentially rather than causing concurrent GPU write hazards.
Original fused optimizer checkpoint/NaN tests remain in the suite.

The C++ planner is general and can be reused for future SGD and
foreach kernels. Those operator shaders are **not** implemented in
this change.

Benchmark and validation artifacts:
- benchmarks/results/multitensor-adamw-2026-10-20/summary.json
- benchmarks/results/multitensor-adamw-2026-10-20/raw-timed-runs.jsonl.gz

The RTX 3090 was not used. AMD ROCm or Radeon Vulkan speedup has not
been measured for this new revision.

## Remaining work

Before using wider horizontal fusion in other algorithms, separately
profile GPU shader time versus host descriptor/command recording.
For complete-network performance, tiled attention backward and
convolution algorithm selection remain higher priority.
