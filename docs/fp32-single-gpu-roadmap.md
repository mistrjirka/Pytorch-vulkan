# FP32 single-GPU correctness and coverage

The current target is general FP32 compute on one Vulkan device. FP16/BF16
compute and multi-GPU execution are deferred by project direction. Integer
indices and boolean masks remain necessary parts of FP32 networks.

This follows the [CUDA universality audit](cuda-universality-audit-b80a1f6.md),
which describes the earlier backend revision. The changes below address four
confirmed defects from that audit; device RNG independence is still unresolved.

## Fixed in this batch

- Matrix multiplication rejects mismatched dtypes and mixed devices before
  interpreting buffers. Empty output dimensions and zero-length reductions work.
  View addressing, storage ranges and dispatch limits are checked.
- Embedding uses exact int32/int64 indices, including row 16,777,217. GPU
  validation checks sign, high words and table bounds before a gather can read
  memory. Strided/offset indices and weight views use native packing; padding
  and frequency-scaled dense gradients retain ATen's semantics.
- `empty` uses ATen's allocation helper, honoring channels-last and
  channels-last-3d and validating dimensions, storage-size overflow, layout and
  pinned-memory requests. `empty`, `empty_strided`, transfers and checked device
  guards reject unavailable logical devices instead of silently using device 0.
  Selecting a physical device before initialization still works: each process
  exposes its selected GPU as logical device 0.
- FP32 `zero_` uses the existing view-aware GPU fill implementation, preserving
  aliases and tensor versions. It can remain queued in resident/deferred mode.
- Shared packing now supports exact int32 words. Device copies select the bool
  byte-packing shader for boolean tensors, fixing the previous FP32-shader
  selection; boolean copies assign one writer per output word.

These are operator, storage and dispatch changes. There are no network-name,
UNet or model-shape rules. All new shader code is original and uses Vulkan 1.0
features; int64 indices are read as pairs of uint32 words without shaderInt64.
The existing capability guards and CPU fallback handle embedding inputs outside
native rank/address/storage limits. Native embedding supports packable views up
to rank 8 and uint32 address bounds, subject to device storage-buffer limits.

Bounds validation reduces at most 256 workgroup flags on GPU and reads a single
FP32 status value on the host. **Embedding still synchronizes for this check.**
Dense embedding backward still uses CPU fallback. This is a correctness fix,
not a claim of fully asynchronous embedding or CUDA performance parity.

## Validation

Results and the ordinary-AdamW diagnostic are recorded under
[`benchmarks/results/fp32-contracts`](../benchmarks/results/fp32-contracts/).
The test matrix uses PyTorch 2.14.1+cpu, with CPU reference outputs and gradients.
SPIR-V is rebuilt with `scripts/compile_embedding.py` and checked by `spirv-val`.

| Execution mode | Full suite result |
|---|---|
| RTX 2080 Ti, resident/deferred | 1,307 passed |
| V100, resident/deferred | 1,307 passed |
| Lavapipe, resident/deferred | 1,306 passed, 1 hardware-test skip |
| RTX 2080 Ti, eager | 1,304 passed, 3 deferred-mode skips |

There are 78 new regression cases. Two standard AdamW steps match CPU parameters
and both moment tensors. Their trace contains zero `zero_` fallbacks, while
`lerp.Scalar_out`, `addcmul.out` and `addcdiv.out` each still fall back twice.
The JSON metadata records the tested extension and source/shader hashes.

The regressions cover exact large indices, int32/int64 views, invalid high words,
negative and out-of-range indices, empty tables and zero-width embeddings,
dense gradients with padding/frequency flags, channels-last convolution forward
and backward, allocator errors, empty matrix multiplication, mutation aliases,
deferred zeroing and integer/boolean device copies.

No new matched CUDA timing profile was performed in this batch, and AMD/Intel
execution remains unverified. The audit's matched CUDA reference observations
remain historical evidence, not a newly measured performance result. Three
existing MSE output-resize warnings remain in the full suite.

## Remaining work

| Area | General implementation goal |
|---|---|
| Device RNG and dropout | Implement an independent Vulkan generator/state, reproducible device-only seeding, random kernels and dropout forward/backward. Do not disguise the CPU generator as device RNG. |
| Optimizers and attention masks | Native `lerp`, `addcmul`, `addcdiv` and triangular masks; ordinary optimizer and causal-attention paths should use reusable primitives. |
| Indexing and reductions | Native embedding backward, gather/scatter/index-select, embedding-bag, sort/top-k, cumulative scans and variance, including valid strides and integer/bool semantics. |
| Spatial operators | General Conv3D and transposed convolution forward/backward, 3D pooling, grid sampling, padding and remaining resize modes. |
| Recurrent networks | General recurrent-cell primitives and gradients; fix the missing fused LSTM cell path. |
| Runtime integration | Single-GPU streams/events and default `torch.compile`/Inductor integration, with correct ordering and lifetime rules. |
| Wider API coverage | Sparse storage/dispatch and remaining FFT/linalg paths; track unsupported cases separately from CPU-backed execution. |
| Universality and speed | Extend feature-network coverage; profile matched Vulkan/CUDA workloads before choosing optimizations. Validate multiple shapes/layouts and GPU vendors. |

FP16/BF16 and multi-GPU are deferred, not requirements for completing these
single-GPU FP32 milestones. CPU-backed cases must continue to be reported as
such. This batch does not resolve the audit's broader operator/runtime gaps.
