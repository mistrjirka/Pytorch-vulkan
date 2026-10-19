# Matched Vulkan and CUDA network performance — 19 October 2026

CUDA is faster on every accepted whole-network comparison in this FP32 eager suite. The six medium zoo cases are 3.18–4.80× slower on Vulkan; the larger U-Net and transformer are 10.51× and 9.41× slower. The closest focused result is pooling-classifier inference at 1.23×. All 63 Vulkan timing processes pass the CPU output and gradient gate, and 3,150 measured Vulkan training/inference executions record zero CPU fallback.

The gaps have concrete, general causes: optimizer primitives compute temporaries then copy into destinations, validation causes extra submission/readback waits, GPU barriers are conservative, and the larger convolution/attention kernels lag CUDA. The larger CUDA ResNet is excluded from speed ratios after failing the fixed gradient gate; its numerical-sensitivity diagnostic is retained.

## Measurement conditions

- Both APIs use the same physical NVIDIA GeForce RTX 2080 Ti, UUID `GPU-30acbf2b-a41a-8b88-e298-fe66c4138b29`, driver 580.178.04. Jobs hold an exclusive `gpu:all` benchmark lock and run the APIs sequentially.
- PyTorch 2.14.1 CPU/Vulkan and 2.14.1+cu130 have the same Torch revision, `5c4886908584029761b579af026dcfb627c84070`. FP32, TF32 disabled, four CPU threads, eager execution.
- Both use ordinary `torch.optim.AdamW(lr=1e-4, foreach=False, amsgrad=True)`. This comparison uses the same optimizer API and options on both backends.
- Vulkan uses resident storage, deferred submissions and strict CPU-fallback rejection. The backend source SHA256 remains `718f6b58fba20e0659af08feaca01dbe4b29c83a77b139e95507422709771b30`; extension SHA256 is `0a6cca6ad0a8aa8cb5b8e41f7980e8900e108a0961277cf69ffff3c15927fa95`.
- Initial parameters, inputs, targets, architecture, shape and Torch revision are hash-checked across APIs and independent processes. Model seed 42; input seed 123. The clean model-zoo reference is `c7c063101ec5209caf8022f5b4ea6948d1a8d6ed` (`model_zoo.py` SHA256 `03562cba36b9b3a8c804d0e2e56b6f633706d520bf8fa0296b0194969d7bd6e6`).
- Three independent processes per backend/case, eight warmup training steps, 25 timed training steps, then eval-mode warmup and 25 timed inference passes. API order alternates across cases and rounds. Reported latency is the median of the three process medians; raw samples and run-to-run ranges are retained.
- Training includes zeroing gradients, forward, loss, backward, input gradients for floating-point inputs, AdamW and the completion synchronization. Inference uses `eval()` and `no_grad()` after the training section. Data creation, initial model/input placement, parity checks, cold pipeline construction and data-loader activity are outside timed samples. GPU clocks are not forced.
- Before timing, outputs, loss, input gradients and all parameter gradients are compared with CPU (`rtol=2e-3`, `atol=8e-5`). MobileNet dropout is temporarily disabled for exact parity only; its original probability is restored for timed training. The original smoke run exposed the expected cross-device mask mismatch on both CUDA and Vulkan; those failed smoke records are retained and excluded from performance results.

## Model coverage

The six zoo architectures exercise dense GEMM and LayerNorm (MLP), convolutions/BatchNorm/pooling/resize (U-Net and ResNet), depthwise/grouped convolution and squeeze/excitation (MobileNet), embedding gather/backward and large parameter updates, and multi-head attention/softmax/GEMM (transformer). Three larger input variants measure scaling without changing network layers.

Twelve focused stock-PyTorch networks add average/adaptive pooling, bilinear/bicubic interpolation, reductions, noncausal/causal attention, Conv3D and transposed Conv3D, grid sampling, bidirectional LSTM, EmbeddingBag/index selection/sort/cumulative sum, FFT and a differentiable linear solve. Their small fixed inputs test coverage and dispatch latency. The FFT length is 16, the solve is 5×5 and the recurrent input is 2×4×5; these are not large FFT, dense linear-algebra or recurrent throughput tests.

### Zoo architectures and larger shapes

| Network / scale | Input shape | CUDA train ms | Vulkan train ms | V/C train | CUDA infer ms | Vulkan infer ms | V/C infer |
|---|---|---:|---:|---:|---:|---:|---:|
| embedding / medium | 128×64 | 2.428 | 8.230 | 3.39× | 0.126 | 0.634 | 5.04× |
| mlp / medium | 256×784 | 2.357 | 9.609 | 4.08× | 0.213 | 0.738 | 3.46× |
| mobilenet_v3 / medium | 2×3×96×96 | 34.728 | 145.608 | 4.19× | 3.881 | 24.371 | 6.28× |
| resnet18 / medium | 4×3×64×64 | 15.290 | 72.676 | 4.75× | 2.051 | 9.889 | 4.82× |
| transformer / medium | 4×128 | 16.091 | 77.220 | 4.80× | 1.257 | 11.106 | 8.84× |
| unet / medium | 1×3×64×64 | 10.051 | 31.923 | 3.18× | 1.091 | 2.918 | 2.68× |
| resnet18 / reference | 32×3×64×64 | Rejected | 118.065 | — | Rejected | 19.428 | — |
| transformer / reference | 16×128 | 17.516 | 164.888 | 9.41× | 2.968 | 23.211 | 7.82× |
| unet / reference | 8×3×96×96 | 10.006 | 105.170 | 10.51× | 2.287 | 21.676 | 9.48× |

### Focused feature networks

| Network / scale | Input shape | CUDA train ms | Vulkan train ms | V/C train | CUDA infer ms | Vulkan infer ms | V/C infer |
|---|---|---:|---:|---:|---:|---:|---:|
| attention_block / feature | 2×37×16 | 1.776 | 3.219 | 1.81× | 0.155 | 0.316 | 2.04× |
| bicubic_autoencoder / feature | 2×3×23×29 | 1.405 | 2.207 | 1.57× | 0.123 | 0.183 | 1.49× |
| bilinear_autoencoder / feature | 2×3×23×29 | 1.379 | 2.204 | 1.60× | 0.122 | 0.184 | 1.50× |
| causal_attention_block / feature | 2×17×16 | 1.792 | 3.176 | 1.77× | 0.157 | 0.311 | 1.98× |
| pool_classifier / feature | 2×3×23×29 | 1.968 | 3.059 | 1.55× | 0.199 | 0.245 | 1.23× |
| recurrent_regressor / feature | 2×4×5 | 6.896 | 34.234 | 4.96× | 0.406 | 5.212 | 12.85× |
| reduction_regressor / feature | 2×3×23×29 | 1.846 | 2.896 | 1.57× | 0.184 | 0.265 | 1.44× |
| retrieval_regressor / feature | 2×7 | 1.828 | 4.502 | 2.46× | 0.204 | 0.731 | 3.58× |
| solve_regressor / feature | 3×5 | 1.576 | 2.799 | 1.78× | 0.222 | 0.412 | 1.85× |
| spatial_transformer / feature | 2×2×9×11 | 1.259 | 2.031 | 1.61× | 0.111 | 0.183 | 1.65× |
| spectral_filter / feature | 3×16 | 1.449 | 2.473 | 1.71× | 0.117 | 0.231 | 1.98× |
| volume_autoencoder / feature | 2×2×6×8×10 | 1.569 | 3.652 | 2.33× | 0.166 | 0.472 | 2.84× |

MobileNet Vulkan training run medians range from 143.235 to 208.647 ms; the reported median is 145.608 ms. Its variability is material, and no cause is assigned from timing alone. All process medians, minimum/maximum samples and pooled p95 values are retained in the summary. No accepted whole-network timing in this suite favors Vulkan.

The larger ResNet (`32×3×64×64`) is rejected on CUDA by the unchanged gradient parity gate in all three processes. Vulkan passes and its standalone training median is retained, but no CUDA/Vulkan speed ratio is calculated. Forward maximum error versus CPU is about 2.1e-5 on CUDA; the first convolution weight-gradient maximum error is about 2.84e-3. The separate diagnostic finds six CUDA-versus-CPU-FP32 ReLU sign disagreements; even CPU FP32 versus CPU FP64 has one. The first-convolution gradient relative L2 error is 0.329% for CUDA versus CPU FP32; CPU FP32 itself differs from CPU FP64 by 0.131%. This supports sensitivity at activation boundaries as an explanation, rather than establishing a CUDA kernel defect. The fixed gate is retained and no speed ratio is published. The diagnostic intentionally downloads activations and is excluded from timing and copy traces. CPU FP64 is only a reference here; this does not add Vulkan FP64 support.

## Phase and shader attribution

The phase diagnostic gives the following CUDA / Vulkan medians in milliseconds. Each phase is synchronized for attribution, so these columns do not sum to the primary deferred latency.

| Network / shape scale | Forward CUDA / Vulkan | Backward CUDA / Vulkan | AdamW CUDA / Vulkan |
|---|---:|---:|---:|
| mlp / medium | 0.251 / 0.857 | 0.958 / 2.308 | 1.113 / 4.869 |
| unet / medium | 1.576 / 4.157 | 4.000 / 7.192 | 4.194 / 19.813 |
| resnet18 / medium | 2.920 / 12.004 | 3.177 / 20.771 | 6.434 / 41.315 |
| mobilenet_v3 / medium | 5.671 / 23.629 | 13.711 / 50.536 | 13.320 / 87.433 |
| transformer / medium | 2.300 / 10.472 | 8.157 / 33.468 | 5.121 / 38.644 |
| unet / reference | 2.343 / 21.645 | 6.747 / 49.891 | 4.124 / 25.321 |
| transformer / reference | 3.361 / 24.373 | 8.636 / 96.710 | 5.280 / 40.935 |

The optimizer deficit is broad. In `csrc/ops/optimizer_primitives.cpp`, `lerp_`, `addcmul_`, `addcdiv_` and `maximum.out` build temporary results and use `out.copy_(result)`. With this AdamW configuration that means four destination-copy operations per parameter tensor, in addition to the arithmetic dispatches. CUDA's corresponding optimizer primitives write their destinations directly. Some other copies perform layout packing and remain useful.

| Workload | Vulkan `device_copy` dispatches / step | Instrumented copy shader ms / step |
|---|---:|---:|
| mlp / medium | 40 | 0.423 |
| unet / medium | 178 | 1.146 |
| resnet18 / medium | 285 | 3.654 |
| mobilenet_v3 / medium | 645 | 8.368 |
| transformer / medium | 252 | 1.937 |
| unet / reference | 178 | 0.889 |
| transformer / reference | 252 | 2.125 |

These are **GPU-to-GPU shader copies**, not CPU fallback. The extra arithmetic, allocation, binding and barrier work around them is also relevant; copy shader time alone understates their total effect.

The large networks also have substantial arithmetic deficits:

- **U-Net, 8×3×96×96:** the top three Vulkan convolution families take 54.883 ms per instrumented step: weight-gradient split/reduction base 20.835 ms, input-gradient tiled convolution 18.774 ms and forward tiled convolution 15.274 ms. Total instrumented Vulkan shader time is 65.521 ms. CUDA uses cuDNN Winograd/SGEMM families; its separately profiled GPU-event sum is 10.351 ms.
- **Transformer, 16×128 tokens:** Vulkan `sdpa_training` accounts for 71.116 ms of 94.981 ms instrumented shader time. The synchronized backward phase is 96.710 ms versus 8.636 ms on CUDA. CUDA selects an FP32 CUTLASS memory-efficient attention backward kernel. The Vulkan shader recomputes attention terms and reduces over keys/queries with repeated workgroup barriers; tiled reuse and parallel reductions are the main algorithmic opportunity.
- **ResNet, 4×3×64×64:** the direct input-gradient convolution path takes 9.197 ms across three calls, and the tiled input-gradient path another 7.136 ms. This identifies the input-gradient algorithm family as a specific kernel target.
- **GEMM:** the 1024×1024 by 1024×1024 diagnostic measures 0.508 ms Vulkan shader time versus a 0.237 ms CUDA event interval. Both use FP32. The full seven-shape matrix is published. CUDA events and Vulkan dispatch timestamps use different instrumentation; small event intervals can include launch gaps and should not be treated as exact kernel-only ratios.

### Copies and synchronization in normal batching

The final Nsight capture uses `trace_network_step.py`: only full-step boundaries synchronize. Its phase markers describe **CPU enqueue ranges**, and impose no extra phase fences. Inputs, initial weights, shape, optimizer and Torch revision match the corresponding wall benchmark and are checked across APIs.

| Workload | Vulkan recorded compute ops | Vulkan queue submits / fence waits | Vulkan pipeline barriers | Descriptor-set allocations | CUDA device completion syncs |
|---|---:|---:|---:|---:|---:|
| mlp / medium | 201 | 5 / 5 | 384 | 161.7 | 1 |
| unet / medium | 853 | 1 / 1 | 1650 | 852.0 | 1 |
| resnet18 / medium | 1368 | 5 / 5 | 2544 | 1368.0 | 1 |
| mobilenet_v3 / medium | 3329 | 5 / 5 | 6256 | 3329.0 | 1 |
| transformer / medium | 1134 | 9 / 9 | 2149 | 1134.0 | 1 |
| unet / reference | 857 | 1 / 1 | 1658 | 856.0 | 1 |
| transformer / reference | 1134 | 9 / 9 | 2149 | 1134.0 | 1 |

Counts are means per measured step. Descriptor allocations are binding objects, and differ from compiled compute pipelines: the primary runs record **zero warmed pipeline-cache misses** on every Vulkan case. Bound-algorithm cache misses and evictions remain frequent.

The MLP's five waits occur as two in loss validation, two in backward validation and one at final completion. `VulkanContext::ensure_host_read` first synchronizes pending compute, then submits and waits for a separate readback sequence. Combining a pending computation and its status readback before waiting could remove one submission/fence per status boundary while preserving immediate validation. Transformer validation produces nine waits. **U-Net already uses one submission/wait**, including at the larger shape; its 10.5× deficit cannot be explained by per-operator CPU fence waits.

The enqueue path also records a barrier after output writes and another before reading dirty inputs. The 384 barriers for 201 MLP operations, and 6,256 for 3,329 MobileNet operations, support reviewing redundant dependencies. A hazard tracker must preserve read-after-write, write-after-read, write-after-write and aliasing semantics.

CUDA has no host/device memcpy activity inside these measured step windows. It does retain device-to-device layout copies: MobileNet records 19 copies totaling 1,180,928 bytes per step; the transformer records 12, totaling 6,291,456 bytes at batch 4 and 25,165,824 bytes at batch 16. Thus some copying is part of both implementations. Vulkan copy-shader counts include the additional optimizer destination copies identified above.

**Trace limit:** this Nsight version omits some Vulkan command-recording calls, including dispatch/copy calls, even though queue counters establish that work was recorded. An absent `vkCmdCopyBuffer` event does not establish zero DMA copies. Vulkan host-read auditing, shader profiles and observed API counts must be considered together. Fence duration includes required GPU completion time and is not a pure CPU overhead measurement.

### Next changes, kept general

1. Implement destination-writing FP32 `lerp`, `addcmul`, `addcdiv` and extrema out/in-place kernels, preserving broadcasting, strides, overlap, aliases and special-value behavior. Retain the composed reference path for unsupported geometry. This improves ordinary PyTorch AdamW across models.
2. Reduce binding-object churn, reuse command resources, merge status readback with pending work where safe, and remove proven redundant barriers through dependency tracking.
3. Tile FP32 attention forward/backward and optimize convolution input/weight-gradient candidates selected by geometry, groups, strides, dtype and device capability.
4. Continue matched-shape GEMM tuning, especially small-batch and narrow-output cases, using kernel timings as well as whole-network latency.

These findings do not require network-name dispatch or model-specific kernels. This change publishes measurements and analysis; the audited backend implementation remains unchanged.


Primary latency and diagnostic profiling are separate. The phase diagnostic synchronizes after each phase on both APIs, changing batching. Vulkan shader timing enables per-dispatch GPU timestamps and disables deferral; CUDA shader/kernel attribution uses the PyTorch profiler. These diagnostic wall times must not replace normal deferred latency. Vulkan timestamp intervals bracket dispatches and can include associated memory ordering costs. Their sums cannot be subtracted from normal wall time to derive CPU overhead.

Normal-batching Nsight host-API counts are restricted to measured NVTX ranges after warmup; only full-step boundaries synchronize. Their phase markers describe host enqueue, rather than synchronized GPU phases. Vulkan GPU workload capture can omit submitted work; observed intervals are lower bounds. Missing Vulkan GPU intervals do not establish idle time or GPU utilization. CUDA kernel names, Vulkan shader timing, queue/cache counters and host-API calls provide complementary evidence.

## CPU residency and correctness

All 63 Vulkan processes pass the CPU output/loss/input-gradient/parameter-gradient parity check. Across 1575 measured training steps and 1575 measured inference passes, registered CPU fallback calls total **zero**. No recorded host read is larger than **four bytes**.

The small readbacks are explicit GPU-computed validation/status values: NLL target validation (`csrc/ops/nll_loss.cpp`), index/embedding validation (`csrc/ops/indexing.cpp`, `embedding_bag.cpp`) and linear-solver status/pivot validation (`csrc/ops/linalg.cpp`). They impose host synchronization even though the numerical tensor operations run on the GPU. The per-case summary records their frequency and bytes per step. The U-Net, image-resize, pooling, reduction, noncausal/causal attention, recurrent, FFT and 3D-convolution cases record zero host reads in the measured numerical paths.

Zero CPU fallback does not mean zero CPU work: Python/autograd, optimizer orchestration, allocation/binding setup, command recording and error/status decisions still execute on the host. Registered CPU fallback and host-read counters are sampled before explicit result downloads. Host-read counters measure logical reads; they do not measure PCIe traffic or GPU-to-GPU copy bytes. The numerical model workloads are assessed separately from four-byte validation/error status reads.

These measurements cover the listed FP32 eager workloads on one GPU. They do not establish performance for every ATen operator/shape/dtype, FP16/BF16/AMP, compiled graphs, real-data training convergence, other GPU vendors or multi-GPU execution, and they do not measure hardware occupancy, bandwidth saturation or every GPU functional unit.

## Reproduction and evidence

From the repository root, with the CPU/Vulkan and CUDA environments installed:

```bash
.venv/bin/python benchmarks/run_network_matrix.py --rounds 3 --warmup 8 --repeats 25 --prefix wall
.venv/bin/python benchmarks/run_network_matrix.py --rounds 3 --warmup 8 --repeats 25 --prefix reference --scale reference --cases resnet18 unet transformer
.venv/bin/python benchmarks/run_network_profiles.py
.venv/bin/python benchmarks/summarize_network_matrix.py benchmarks/results/network-matrix-2026-10-19 --pack --allow-rejected
```

Use `--cuda-device` and `--vulkan-device` to select the same physical device on another machine. Performance jobs require exclusive GPU access. The GEMM diagnostic uses `benchmarks/probe_matmul_gpu_time.py` with 40 repeats per shape; Vulkan additionally sets `TORCH_VULKAN_GPU_RESIDENT=1 TORCH_VULKAN_DEFER_SUBMISSIONS=0 TORCH_VULKAN_PROFILE_TIMESTAMPS=1 TORCH_VULKAN_FORBID_CPU_FALLBACK=1`.

Evidence: [summary](../benchmarks/results/network-matrix-2026-10-19/summary.json), [environment and source manifest](../benchmarks/results/network-matrix-2026-10-19/environment.json), [medium/focused raw runs](../benchmarks/results/network-matrix-2026-10-19/wall-raw.jsonl.gz), [larger-shape raw runs](../benchmarks/results/network-matrix-2026-10-19/reference-raw.jsonl.gz), [diagnostic raw records](../benchmarks/results/network-matrix-2026-10-19/diagnostics-raw.jsonl.gz), and [original smoke records](../benchmarks/results/network-matrix-2026-10-19/smoke-raw.jsonl.gz). Each compressed file is JSON Lines containing the original filename and complete JSON record. Full Nsight captures and SQLite exports remain in the development sandbox under the corresponding `diagnostics` directory; measured-range summaries are published.

Partial GPU clock/power telemetry is retained in [gpu-telemetry.csv.gz](../benchmarks/results/network-matrix-2026-10-19/gpu-telemetry.csv.gz). Collection starts during the final medium timing round and covers the later profiling work; it does not cover every baseline sample. Its coarse GPU-utilization readings are not occupancy measurements.
