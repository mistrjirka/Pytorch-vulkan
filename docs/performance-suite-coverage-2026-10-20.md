# Expanded Vulkan backend performance coverage, 20 October 2026

Baseline/verified GitHub commit: \`1c2d9360376619d90740a1ff533eec98f07d4fcd\`.
This is **post-optimization performance characterization**, not a new shader
optimization or a formal CUDA replacement certification.

## Framework: distinguish three test levels

1. **Correctness:** the complete repository unit/integration suite, previously
   3876 passing tests on RTX 2080 Ti and Tesla V100 (3875/1 skipped on Lavapipe).
   These are not latency measurements.
2. **End-to-end model performance:** the established
   \`benchmark_network_matrix.py\` and \`run_network_matrix.py\` matched
   CUDA/Vulkan suite. It times *entire forward/loss/backward/AdamW steps* and
   eval-mode forward, and includes seeded CPU output, loss, input-gradient,
   and every parameter-gradient parity before recording speed.
3. **Operation performance:** the existing 39-case \`probe_complete_audit.py\`
   correctness probe gained opt-in \`--perf-warmup\` and \`--perf-repeats\`
   timing controls. Its timed region synchronizes the backend explicitly and
   measures forward+backward where the case requests gradients. Unsupported
   dtypes/cases remain visible, not discarded.

External alternatives:
- Official [TorchBench](https://github.com/pytorch/benchmark) provides many
  real models with a standardized API. Full unmodified TorchBench is **not**
  validated against this PrivateUse1 Vulkan backend, and importing its entire
  dependency ecosystem is not a substitute for checking model semantics.
- [torch.utils.benchmark](https://docs.pytorch.org/docs/stable/benchmark_utils.html)
  provides microbenchmark utilities and replicate controls. Our runner uses
  explicitly synchronized wall timing to avoid relying on CUDA-only automatic
  device synchronization for PrivateUse1.

## Measurement contract

Hardware: **the same RTX 2080 Ti for both Vulkan and CUDA**, physical GPU
UUID \`GPU-30acbf2b-a41a-8b88-e298-fe66c4138b29\`. The RTX 3090
was **not used**. No AMD/ROCm measurements were performed in this run.
PyTorch revision \`5c4886908584029761b579af026dcfb627c84070\`
for both APIs; TF32 disabled; FP32 main workloads; four CPU threads.

Vulkan GPU residency and deferred submissions explicitly enabled.
Numerical CPU fallback forbidden. Warmed execution timings exclude
model construction, shader compilation, initial data uploads and CPU
reference checks. CPU wall time includes PyTorch dispatcher, host graph
recording, submission and GPU completion, which can matter greatly for
small workloads.

The runner records model/input weight hashes, current source revision,
GPU name, optimizer, original per-step samples and execution audit counters.
All per-model comparisons require CPU output **and gradients** parity
(\`rtol=2e-3, atol=8e-5\`); CUDA and Vulkan API order alternates across
rounds. Models are matched across GPUs, dtype, shape, and optimizer:
\`torch.optim.AdamW(lr=1e-4, foreach=False, amsgrad=True)\`.

### Three-round, 18-network medium suite

- Six zoo architectures: \`mlp\`, \`unet\`, \`resnet18\`,
  \`mobilenet_v3\`, \`embedding\`, \`transformer\`.
- Twelve feature architectures: pooling, bilinear/bicubic upsampling,
  reductions, standard and causal attention, 3D convolutions and transpose,
  grid sampling, bidirectional LSTM, embedding/index/sort/cumsum,
  FFT and differentiable linear solve.
- Each backend/model: **3 fresh processes × 8 warmups × 20 synchronized
  timed iterations**, both full training and eval inference.
- **108/108 processes passed** parity; zero registered CPU numerical
  fallback in all measured Vulkan workloads. Logical host reads were
  at most 4 bytes/read (validation/status); zero large tensor readback
  is needed in the timed computation.

| Network | CUDA train ms | Vulkan train ms | V/C train | V/C infer |
|---|---:|---:|---:|---:|
| embedding | 2.417 | 5.391 | 2.23x | 4.98x |
| mlp | 2.368 | 6.170 | 2.61x | 3.45x |
| mobilenet_v3 | 34.254 | 125.555 | 3.67x | 5.47x |
| resnet18 | 12.523 | 58.629 | 4.68x | 4.76x |
| transformer | 16.142 | 64.850 | 4.02x | 8.90x |
| unet | 10.071 | 23.528 | 2.34x | 2.30x |

Selected smaller-feature training gaps: pool classifier 1.31x,
bilinear autoencoder 1.38x, bicubic autoencoder 1.37x,
reduction regressor 1.43x, spatial transformer 1.38x,
LSTM recurrent regressor 4.17x. **Recurrent inference is 19.81x
CUDA** at the tiny sequence shape and may be dominated by excessive
dispatch/host overhead; this needs a stage profile to establish the cause.

Relative to the historical October 19 network suite (25 timed steps
and the same architecture/scale, also 3 fresh processes), Vulkan
full-training medians changed as follows:
MLP 9.609→6.170 ms (-35.8%); U-Net 31.923→23.528 (-26.3%);
embedding 8.230→5.391 (-34.5%); Transformer 77.220→64.850
(-16.0%); ResNet18 72.676→58.629 (-19.3%);
MobileNetV3 145.608→125.555 (-13.8%).

**Attribution warning:** multiple runtime, convolution, optimizer and
direct-output commits accumulated between those suites, so these
before/after figures are not proof that the immediately preceding
\`out=\` commit alone accelerated functional model training. Some
new model medians have material run-to-run variance. In particular
the medium U-Net Vulkan training medians were ~23.53/35.55/23.12 ms.
The reported *median of process medians* is 23.53 ms.

Several feature-network inference medians were slower in the new suite
(e.g. FFT and reduction classifiers, attention). CUDA inference
remained broadly similar while Vulkan runs varied. These are
**regression candidates**, not verified causal regressions.
Re-run matched commit-level A/B under controlled clocks before fixing.

### Larger-shape scaling probe

Two fresh processes per backend/model; 8 warmup and 20 timed steps.

| Model/shape | CUDA training ms | Vulkan training ms | V/C train | V/C infer |
|---|---:|---:|---:|---:|
| U-Net / 8×3×96×96 | 10.387 | 53.342 | 5.14x | 4.10x |
| Transformer / 16×128 tokens | 16.307 | 150.695 | 9.24x | 8.09x |

Both networks passed parity and strict no-fallback checks. Historical
Vulkan U-Net at this shape was 105.170 ms training; the current
53.342 ms is ~49% lower. The Transformer was 164.888 ms then and
150.695 ms now (~9% lower). These are multi-commit observations,
not single-change A/B results. The two-process scaling probe has
lower statistical confidence than the three-process medium suite.

### 39-case operator sweep

\`probe_complete_audit.py --perf-warmup 5 --perf-repeats 25\` runs
three independent processes per backend, retaining all unsupported
cases. 33 out of 39 operations have paired passing CUDA/Vulkan
forward and requested backward checks. Six Vulkan cases
(\`add_fp16, add_bf16, add_fp64, mm_fp16, mm_bf16, mm_fp64\`) are
**not implemented** and are explicitly excluded from speed ratios.

Selected whole-operation forward+backward or forward-only (when
applicable) Vulkan/CUDA ratios on the RTX 2080 Ti:

| Case | V/C latency |
|---|---:|
| add_int64 | 5.96x |
| sum_int64 | 4.18x |
| FFT (forward-only) | 3.70x |
| embedding | 3.66x |
| cross_entropy | 3.58x |
| complex64 mm | 3.57x |
| topk | 3.51x |
| gather | 3.47x |
| scatter_add | 3.01x |
| SDPA, noncausal | 2.03x |
| SDPA, causal | 2.04x |
| FP32 mm | 1.26x |

There were **zero numerical CPU fallbacks** across accepted Vulkan
operator timing loops. These audit shapes are deliberately small
(e.g. FFT length 8); this is a latency/coverage measurement, not
peak throughput. The six unpaired dtype cases are **feature gaps**,
not performance regressions. Avoid ranking a 0.02-ms integer
operation above multi-ms Transformer workload based only on slowdown
ratios: absolute impact and frequency matter.

## Repeatability / commands

From the project root, with the physical RTX 2080 Ti specified by the
default CUDA UUID and no additional GPU workloads:

\`\`\`sh
.venv/bin/python benchmarks/run_network_matrix.py \
  --rounds 3 --warmup 8 --repeats 20 --scale medium \
  --prefix post-1c2d936 \
  --out benchmarks/results/network-matrix-post-1c2d936-2026-10-20
.venv/bin/python benchmarks/summarize_network_matrix.py \
  benchmarks/results/network-matrix-post-1c2d936-2026-10-20 \
  --prefix post-1c2d936 --expected-rounds 3 --pack
.venv/bin/python benchmarks/run_operator_performance.py \
  --rounds 3 --warmup 5 --repeats 25 \
  --out benchmarks/results/operator-performance-post-1c2d936-2026-10-20
.venv/bin/python benchmarks/summarize_operator_performance.py \
  benchmarks/results/operator-performance-post-1c2d936-2026-10-20 \
  --expected-rounds 3 --pack
\`\`\`

The runner also accepts \`--start-round 2\` for recovery from an
interrupted two-round network benchmark without duplicating rounds.

Published compact artifacts, with original individual timings in gzip
JSONL and per-case validation/queue counters in JSON summaries:

- \`benchmarks/results/network-matrix-post-1c2d936-2026-10-20/summary.json\`
- \`benchmarks/results/network-matrix-post-1c2d936-2026-10-20/post-1c2d936-raw.jsonl.gz\`
- \`benchmarks/results/network-matrix-post-1c2d936-reference-2026-10-20/summary.json\`
- \`benchmarks/results/network-matrix-post-1c2d936-reference-2026-10-20/post-1c2d936-raw.jsonl.gz\`
- \`benchmarks/results/operator-performance-post-1c2d936-2026-10-20/summary.json\`
- \`benchmarks/results/operator-performance-post-1c2d936-2026-10-20/raw-operator-runs.jsonl.gz\`

## Next work: prioritize by actual bottleneck

1. **Transformer (reference batch16)**: training gap 9.24x and
   inference gap 8.09x. Profile attention forward/backward and
   per-call synchronization again on the latest backend;
   compare shader-only GPU time and CPU record/submit costs.
2. **U-Net / ResNet / MobileNet larger spatial and grouped convolutions**:
   compare forward, dInput, dWeight and normalization across
   batch/stride/groups/dilation, selecting algorithms by capability and
   workload rather than model name. The U-Net improvement shows the
   backend is responsive to general optimizations but still 5.14x
   slower at the reference batch.
3. **Bidirectional LSTM / recurrent inference**: 19.81x at tiny shape
   despite numerical correctness. Determine whether the repeated
   gate operations, dispatch overhead or GPU serialization dominate.
4. **Non-FP32 and data-movement operations**: BF16/FP16, gather/topk,
   complex matmul, integer reductions, embedding and FFT performance.
   Add *larger shapes and stride/dtype variations* before interpreting
   microbench ratios as throughput limits.
5. Curate a **TorchBench model subset** with device-agnostic inputs
   and licenses/dependencies vetted. Do not claim complete TorchBench
   compatibility until its standardized train/eval modes actually
   execute under PrivateUse1 with numerical parity and zero fallback.

No performance threshold is enforced as CI pass/fail based on a single
noisy GPU run. A future regression gate should compare at least three
repeated process medians, pin a trusted baseline, allow noise margins
and require correctness independently.
