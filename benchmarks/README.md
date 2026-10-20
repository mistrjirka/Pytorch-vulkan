# Benchmarking and model-zoo smoke tests

[Native pooling, layouts and fence reuse](../docs/pooling-layouts-and-fences.md)
contains the pooling U-Net A/B, matched CUDA operator profiles, and submission
API measurements. The [allocation and scalar-update follow-up](../docs/runtime-allocation-and-scalar-updates.md)
adds interleaved A/B runs, one-change ablations, and synchronized stage comparisons.

**[Latest matched CUDA/Vulkan optimization analysis](../docs/cuda-vulkan-bottleneck-analysis.md)**
documents kernel timestamps, profiler evidence, GPU-resident training and
direct before/after comparisons on the same RTX 2080 Ti.

**[GPU comparison report: Vulkan vs CUDA vs CPU](GPU_COMPARISON.md)** includes 8 architectures, measured FP32 inference and full AdamW training, medium-resolution image inputs, and optimizer-stage profiling.


This project intentionally **imports the actual model definitions** from
[mistrjirka/pytorch-XDNA2](https://github.com/mistrjirka/pytorch-XDNA2);
no copies of its network implementations are vendored.

The upstream benchmark compares **BF16 CPU vs BF16 XDNA2**, with larger
batch/spatial sizes. Vulkan doesn't support BF16 yet, so this is a
**reduced-size FP32 correctness and speed probe, not an equivalent benchmark**.
It compares forward output, gradients, one AdamW step, and forward+backward
CPU versus real Vulkan. Errors are absolute maxima.

```sh
git clone https://github.com/mistrjirka/pytorch-XDNA2.git ../pytorch-XDNA2
python -m pip install --index-url https://download.pytorch.org/whl/cpu \
    'torchvision==0.29.1+cpu'
VK_ICD_FILENAMES=/etc/vulkan/icd.d/nvidia_icd.json \
TORCH_VULKAN_PHYSICAL_DEVICE_INDEX=0 \
python benchmarks/model_zoo_smoke.py --zoo ../pytorch-XDNA2 \
    --models mlp,unet,embedding,style_net --repeats 3 \
    --json benchmarks/results/rtx2080ti_model_zoo_fp32.json
```

The sample result in `results/` uses XDNA2 commit
`c7c063101ec5209caf8022f5b4ea6948d1a8d6ed` from 2026-10-15.
The scripts require PyTorch 2.14.1, torchvision for StyleNet, and a
functional Vulkan driver.

Known unsupported models: LSTM (missing fused LSTM cell),
convolutional autoencoder (missing transposed convolution). The reduced
ResNet18 probe can show substantial gradient sensitivity at tiny spatial
sizes; it is excluded from the validated four-model summary.

## Reproducible cross-operator and full-network performance suites (2026-10-20)

See [the latest broad CUDA/Vulkan performance coverage report](../docs/performance-suite-coverage-2026-10-20.md).
The suite measures two distinct levels: 18 complete training/inference networks and
39 operator cases (33 currently have paired CUDA/Vulkan correctness and timings).
Full network timing includes zero-grad, forward, loss, backward, AdamW, and
synchronization; operator timing includes backward when that case requests it.
These are **performance measurements**, not replacements for the existing correctness tests.

Run each backend **on the same physical RTX 2080 Ti**, with matching models,
FP32/TF32 settings, independent warmup, numerical CPU-reference parity and
explicit GPU synchronization. Vulkan's PrivateUse1 device should not be timed
using an unsynchronized CPU timer. The harness isolates API backends in fresh
processes and alternates CUDA/Vulkan order. Never let CUDA select a different
GPU by accident: the runner defaults to the RTX 2080 Ti UUID, not the RTX 3090.

```bash
.venv/bin/python benchmarks/run_network_matrix.py --rounds 3 \
  --warmup 8 --repeats 20 --scale medium --prefix my-baseline \
  --out benchmarks/results/my-baseline
.venv/bin/python benchmarks/summarize_network_matrix.py \
  benchmarks/results/my-baseline --prefix my-baseline --expected-rounds 3 --pack

.venv/bin/python benchmarks/run_operator_performance.py --rounds 3 \
  --warmup 5 --repeats 25 --out benchmarks/results/my-operator-baseline
.venv/bin/python benchmarks/summarize_operator_performance.py \
  benchmarks/results/my-operator-baseline --expected-rounds 3 --pack
```

The operator suite preserves unsupported/error cases, and the network suite
requires CPU-reference output/gradient parity **before** accepting any latency.
No single model score establishes universal speed: consult its input shape,
full latency distribution, numerical execution audit, and CUDA/Vulkan ratio.
Official [TorchBench](https://github.com/pytorch/benchmark) offers broader
third-party models; its unmodified model drivers have **not** been verified
against this Vulkan PrivateUse1 backend.
