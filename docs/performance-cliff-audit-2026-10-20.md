# Vulkan performance cliffs and execution-model audit

2026-10-20. Source revision: \`2cdd8f4\` before this documentation-only report.
Physical benchmark GPU: RTX 2080 Ti. Other device parity from previous tests:
V100 and Lavapipe; RX 7800 XT ROCm/Vulkan network timing from the
earlier Lab02 profile.

## Summary

There are **multiple independent performance cliff mechanisms**:
1. Conv weight-gradient algorithm **ineligibility from fixed scratch limit**;
2. BMM **CPU fallback above 3D Vulkan dispatch grid device limits**;
3. MM **suboptimal kernel selected by a fixed shape threshold**;
4. GroupNorm **one-workgroup-per-group scaling**;
5. Attention **redundant forward score computation and split backward passes**.

Separate systemic issues include descriptor binding churn, conservative
whole-op barriers, optimizer microkernel fan-out, eager host staging
zero-initialization, and selected host-read validation boundaries. Their
performance contribution must be measured independently: pipeline recompilation
is *not* the same thing as descriptor-set creation, and synchronized wall
time is *not* pure CPU overhead.

## Evidence and exact source

### 1. Convolution: confirmed high-severity cost cliff

- \`csrc/ops/convolution.cpp\`:
  \`workspace_bytes()\`, \`workspace_limit()\`, \`applicable()\`,
  \`heuristic()\`, \`execute_candidate()\`.
- Tiled weight-gradient scratch:
  \`out_weight_elements * ceil(N*Hout*Wout/256) * sizeof(float)\`.
- A particular decoder conv needs precisely 72 MiB:
  \`(128*256*3*3)*64*4 = 75,497,472\` bytes.
- Matched batch-1 MyNet medians on RTX 2080 Ti:

  | Workspace MB | 30-step median |
  |---|---:|
  | 71 | 102.274 ms |
  | 72 | 63.001 ms |
  | 73 | 62.664 ms |

- Earlier **100-step** runs: 64 MB 104.71 ms; 96 MB
  63.18 / 62.57 ms; reference forward/backward tested against CPU FP32,
  50 parameter-gradient tensors matched tolerances, no CPU fallback.
- Instrumented GPU timestamps at default vs 96 MB:
  weight-gradient generic reduction 40.38 -> 1.08 ms/step,
  tiled weight-gradient 6.24 -> 8.88 ms/step,
  **total shader time 74.02 -> 35.13 ms/step**.

**Root cause:** when its scratch exceeds a fixed global cap, a fast tiled
algorithm becomes ineligible. The fallback re-reduces activations/gradients
for many separate weights and reuses little data. The binary choice is driven
by the workspace bound, not an estimated relative runtime.

**Better design:** streaming/chunked Split-K tiled weight gradients with
bounded scratch (partial tile reduction as it proceeds), or an internal
grouped/compact accumulation algorithm between tiled and scalar-like
reduction. Tune per device/shape with workspace budget as a parameter.
Do not allocate scratch unconditionally based on a global 96 MB default;
honor RAM/device limits. Preserve generic correctness fallback.

### 2. BMM: confirmed capability-to-CPU cliff

- \`csrc/ops/bmm.cpp\`: \`batch <=
  maxComputeWorkGroupCount[2]\` part of \`can_dispatch\`; otherwise calls
  \`native::cpu_fallback_tensor\` on both inputs.
- Demonstration with 1x1 float32 matrices:
  - batch 65,534: GPU success;
  - batch 65,535: GPU success;
  - batch 65,536: attempts CPU fallback (explicit error with
    \`TORCH_VULKAN_FORBID_CPU_FALLBACK=1\`).
- This is a correctness-compatible but severe acceleration cliff, with no
  need for more than a few hundred KiB of tensor data. We did **not**
  measure how long the CPU fallback takes.
- Remedy: linear/2D packing of batch-grid indices where supported by
  grid limits, or multiple GPU dispatches over independent batch chunks,
  without synchronizing or readback between chunks.
  Maintain strided/broadcasted batch semantics and exact output ownership.

### 3. Matrix multiply: modest confirmed dispatch mismatch

- \`csrc/ops/mm.cpp\` selects the register kernel when
  \`M>=256 && N>=256 && K>=64\`, otherwise tiled.
- On 256x256 times 256x256 FP32, two runs of 100 synchronized calls,
  identical operands and CPU numerical parity:
  - forced register tile: 0.10437 and 0.10370 ms medians;
  - forced basic tile: 0.09087 and 0.09838 ms medians.
- Register is **6–15% slower** here on this RTX 2080 Ti.
  This is not evidence it is universally slower on other shapes/devices.
- Remedy: device-aware and shape-keyed benchmark/cost model, with a
  conservative fallback and no first-call autotuning surprises.

### 4. GroupNorm: measured scalability issue, not identical semantics

- \`csrc/ops/group_norm.cpp\` + \`csrc/shaders/group_norm_forward.comp\`.
  One workgroup per (batch,group), independently of elements per group.
- FP32 \`N=1,C=64,H=W=64\`, 40 warmed synchronized calls per choice,
  each validated against the corresponding CPU GroupNorm reference:

  | Groups | Values per group | Median |
  |---|---:|---:|
  | 1 | 262,144 | 0.8981 ms |
  | 2 | 131,072 | 0.5255 ms |
  | 4 | 65,536 | 0.3402 ms |
  | 8 | 32,768 | 0.2092 ms |
  | 16 | 16,384 | 0.1445 ms |
  | 32 | 8,192 | 0.1093 ms |
  | 64 | 4,096 | 0.0795 ms |

- Changing groups also changes the numerical operation. This is **not**
  a same-result speedup measurement. It is evidence that low-group-count
  normalization currently underutilizes the GPU and forces large serial
  loops per workgroup.
- Remedy: hierarchical/multi-workgroup partial mean and variance
  reductions for large groups, followed by normalization. Test Welford or
  stable two-pass variance; careful cross-workgroup synchronization, extra
  scratch, affine parameter gradients and numeric tolerances.

### 5. Reduction threshold: justified, not a pathological fallback

- \`csrc/ops/reduce.cpp\` changes from per-element serial reduction to
  per-output-row workgroup parallel reduction when \`reduced>=64\`.
- A 512-row sum, float32, 40 warmed synchronized calls, CPU parity:
  63 values per row: 0.0921 ms; 64: 0.0649 ms; 65: 0.0651 ms.
- On the tested shape it improves at the threshold. No CPU fallback.
  Keep it as a regression benchmark; test other devices/axes before changing.

### 6. Attention: architectural candidate, not newly benchmarked

- \`csrc/shaders/sdpa_training.comp\` forward traverses the score blocks
  once to produce logsumexp and a second time to weight values; backward
  runs separate query-gradient and key/value-gradient loops and recomputes
  probabilities. No SxL probability matrix is saved, a deliberate storage
  tradeoff.
- Hypothesis: redundant FLOPs/synchronization lead to poor training
  throughput on long sequences, but any blockwise fused algorithm must
  preserve causal and arbitrary mask semantics, dropout RNG, stable softmax,
  ragged dimensions, gradient correctness and tensor-lifetime safety.
- Implement FlashAttention-like tiled recomputation/data reuse as a new
  candidate, not as a special case tied to one transformer model.

## Shared backend execution concerns (source-verified; impacts not isolated)

- \`csrc/vulkan_context.cpp\`: bound \`Algorithm\` cache keyed by shader
  and concrete VkBuffer handles/ranges. \`kMaxBoundAlgorithms=512\`
  (in \`csrc/vulkan_context.h\`). Training output allocations churn keys.
  Previous profile: 493 bound algorithm misses and evictions over three
  U-Net steps, 493 **pipeline cache hits**, zero new pipelines.
  **Do not confuse descriptor binding churn with shader compilation.**
- Vendored \`Kompute/src/Algorithm.cpp\`:
  descriptors acquired from a reusable arena; individual
  \`vkUpdateDescriptorSets\` per binding. Safe reuse must account for
  in-flight command buffers. It is **not** evidence of a descriptor leak.
- \`VulkanContext::enqueue_resident_compute\` records dependency barriers
  on dirty inputs and after each output. This is conservative and likely
  limits overlap of independent ops; not yet measured separately.
  Never remove barriers without tracking read/write hazards and validating
  Vulkan memory model correctness.
- \`VulkanAllocator::allocate\` initializes a host
  \`std::vector<float>(num_floats,0)\` for each *new* buffer even when
  the caller immediately overwrites the GPU contents. The extent of
  additional upload overhead depends on later residency/sync behavior.
  Exact-size reuse and deferring reuse during in-flight work may worsen
  fragmentation/peak allocation. **Not yet measured in isolation.**
- Standard PyTorch Adam produces around 150 \`optimizer_pointwise\`
  dispatches per MyNet step; shader work is only about 0.84 ms in a
  separate eager timestamp profile versus ~15 ms end-to-end optimizer
  phase. These timings come from different instrumentation modes and
  **cannot be subtracted to yield exact CPU overhead**.
  \`torch_vulkan/optim.py\` contains *opt-in* native fused AdamW, which is
  not wired into ordinary \`torch.optim.Adam\`. Measure fused vs standard
  step with identical optimizer semantics before recommending defaults.
- A limited set of dynamic validation/metadata operations use
  \`.cpu().item()\`, including random, sparse and histogram operations.
  Those synchronize the GPU graph. A device-only validation/error-flag
  approach is preferable where exact validation semantics can be retained.

## OpenCode independent review

Used installed OpenCode 1.19.3 with
\`opencode/deepseek-v4.1-flash-free\` (verified usable).
Two broad read-only autonomous repository scans timed out while
repeatedly reading large source files; neither produced a completed
opinion. Two **bounded prompts containing inspected code facts and
observed measurements** completed successfully. These provide a
*second interpretation of source evidence*, not an independent
reproduction of code execution.

The two successful reviews concurred with:
- BMM and conv eligibility-to-fallback are confirmed cliffs;
- MM threshold may be suboptimal; GroupNorm and attention are scaling
  candidates; 64-element reduction threshold is beneficial in tested case;
- descriptor/allocator lifecycle and dispatch/barrier fan-out deserve
  architecture work with correctness safeguards;
- optimizer fusion and reducing host synchronization should be tested
  with exact numerical/semantic parity.
They proposed capacity-aware Split-K, grid chunking, hierarchical
normalization, per-device matmul choices, descriptor-lifetime-safe
binding management, and fused/foreach optimizers. Their unmeasured
speedup suggestions are **hypotheses**, not benchmark results.

## Next experiments, in order

1. **BMM general grid flattening/chunking**, add correctness tests around
   batch limit, offsets/transposes and CPU fallback forbidden.
2. **Convolution bounded tiled wgrad** below 72 MiB for the same shape:
   compare 64, 71, 72, 96 MB on RTX and actual AMD, timing partial kernel,
   reduction, allocation and total step. Preserve FP32 oracle.
3. **GroupNorm hierarchical reduction** for G=1/2 large HxW with
   batch/group diversity. Verify forward, dInput, dWeight, dBias.
4. **Optimizer/descriptor allocation profiling** with CPU instrumentation
   and GPU timestamps, then benchmark a semantics-equivalent fused optimizer.
5. **Attention recomputation and MM heuristic** per shape/device,
   recording dispatch choices and peak scratch alongside speed.

Goal remains a *universal PyTorch Vulkan backend*, not hardcoded networks
or silent CPU fallbacks.

## Useful upstream references

- Khronos descriptor/buffer management:
  https://docs.vulkan.org/samples/latest/samples/performance/descriptor_management/README.html
- AMD RDNA command buffer and dispatch advice:
  https://gpuopen.com/learn/rdna-performance-guide/
- MIOpen per-convolution workspace query and slow-algorithm warning:
  https://rocmdocs.amd.com/projects/MIOpen/en/latest/doxygen/html/group__convolutions.html
- ncnn pipeline/object cache:
  https://github.com/Tencent/ncnn/wiki/vulkan-pipeline-cache
- ncnn separate blob/workspace allocators:
  https://github.com/Tencent/ncnn/wiki/custom-allocator
