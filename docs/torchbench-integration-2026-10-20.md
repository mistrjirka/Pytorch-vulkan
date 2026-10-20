# Official TorchBench integration and general integer scatter optimization (2026-10-20)

## Status and setup

TorchBench upstream checkout: `/workspace/pytorch-vulkan-official-torchbench`
at `9dd1f261115e4db7a999744d92931167e01c9be7`. The upstream
`torchbenchmark.models.*.Model` implementation is executed unchanged. We
provide the device-specific runner externally rather than patch TorchBench.

Dependencies installed in the two isolated Python 3.12 environments:
`PyYAML`, `psutil`, and TorchBench's other pre-existing import dependencies.
CUDA PyTorch `2.14.1+cu130`; Vulkan PyTorch `2.14.1+cpu` and native Vulkan backend.

Run on the **same RTX 2080 Ti**, explicitly pinned by physical GPU UUID
for CUDA and Vulkan physical index for Vulkan, using:

```bash
.venv/bin/python benchmarks/run_official_torchbench_matrix.py \
  --cases dcgan:eval:2,pytorch_unet:eval:1 \
  --warmup 3 --repeats 10 --rounds 3 \
  --out benchmarks/results/torchbench-regression
```

The matrix runner alternates backend order across rounds; it records every
unsupported model, error, and timeout, not just successes. The per-model
runner saves the upstream revision, backend revision, PyTorch version, GPU,
sample timing distribution, and numerical comparison to a CPU copy of the
**same upstream model** for evaluation. For Vulkan the runner checks
`TORCH_VULKAN_FORBID_CPU_FALLBACK=1` and enforces zero recorded CPU
fallback calls. It uses production-default GPU residency and deferred
submission modes. All timings are synchronous and include model call
overhead, not initialization. Upstream TorchBench eval() may contain
postprocessing; we time it as-is. Upstream training() remains the whole
original training method; training numerical parity is **not yet checked**
by this evaluation-only parity procedure.

## Initial measured results: original TorchBench models

Same RTX 2080 Ti, FP32, TF32 off. Both backends were tested directly, not
through inferred benchmark results.

| Case | CUDA median | Vulkan before | Vulkan after | Numerical parity | CPU fallback |
|---|---:|---:|---:|---|---:|
| DCGAN discriminator eval batch 2 | 1.130 ms | 2.368 ms | unchanged | pass (max abs 4.17e-7) | 0 |
| U-Net segmentation eval batch 1, 640x959 | 39.609 ms | 1261.671 ms | 355.924 ms | pass (max abs 6.71e-8) | 0 |

DCGAN: 2 warmup and 5 timed steps each. A separate single-step matrix
smoke also successfully ran the unmodified DCGAN **training** workload,
which includes forward, backward, and both Adam optimizers: CUDA 319.99 ms,
Vulkan 771.72 ms (2.41x), no Vulkan CPU fallback. That single cold
iteration is not a stable training performance estimate, and training-step
numerical parity has not yet been checked.
U-Net initial: 1 warmup and 2 timed steps each.
U-Net after optimization: 2 warmup and 5 timed steps. Additional paired,
multi-round regression runs recommended before treating changes as stable
speedups.

The U-Net latency was unexpectedly dominated by an `advanced_index`
GPU kernel. With diagnostic Vulkan timestamps enabled and deferred submission
disabled, the baseline ~1291 ms step contained:

- `advanced_index`: 905.37 ms from a single dispatch
- `conv2d_tiled_register3` + `conv2d_tiled`: 249.26 ms
- Remaining kernels: ~58 ms total

The original `advanced_index` write shader serialized the entire
`index_put` operation into one GPU invocation to avoid write races. Upstream
`F.one_hot` postprocessing of segmentation logits eventually used integer
scatter and therefore triggered that costly general indexing path.

## General fix

`csrc/ops/indexing.cpp` now routes **int32/int64 scatter and
scatter_add** through the existing output-centric
`indexed_integer` shader rather than the serialized `advanced_index`
write path. The shader scans the input indices for each distinct output
element, so output writes are parallel with no cross-thread write races.
For repeated indices, regular scatter deterministically takes the last
matching element, and scatter_add accumulates all matches starting with
the original destination value. Native int64 arithmetic uses two 32-bit
words. Generic `index_put_` remains on its separate serial
correctness path; it is not implicitly changed. This optimization is
automatic, with no performance-specific environment switch.

Direct regression tests: 28 matching tests pass; broad
`test_complete_indexing.py`, `test_opinfo_regressions.py`, and new
`test_parallel_integer_scatter.py` suite: **1740 passed**.
Tests include duplicate indices, positive/negative int32/int64 data,
integer scatter_add, scalar scatter, and spatial one-hot.

**Important limits:** This is not a CUDA-equivalent Vulkan
convolution: U-Net still takes ~9.0x CUDA time. The benchmark is a
small initial TorchBench subset; whole-suite train/eval coverage,
precision policy, GPU-independent deterministic training parity,
and operator_benchmark integration are future milestones.
