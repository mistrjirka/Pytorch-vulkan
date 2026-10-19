# Convolution 3×3 heuristic — 19 October 2026

## Change

Keep the universal FP32 convolution implementation and all four existing tiles. When automatic dispatch encounters sufficiently large dense 3×3 convolutions (groups=1, unit stride/dilation, Cin/Cout≥32 and batch×output-spatial≥4096), prefer the already validated 32×32, four-output/thread register tile if its device/workspace capability checks permit it. Otherwise keep the existing base tile or eligible generic fallback. Explicit tile/algorithm overrides and optional autotuning are unchanged. **No Winograd or attention kernels are added.**

## Numerical gates

- Full repository suite: RTX 2080 Ti **3679 passed**, V100 **3679 passed**, Lavapipe **3678 passed, one skipped**. All seven pre-existing warnings remain.
- Strict focused convolution suite: **133 passed on each of RTX, V100 and Lavapipe**, in **both eager and deferred** modes. Zero numerical CPU fallback permitted.
- New focused tests exercise the default selected implementation on dense large/small 3×3, rectangular 3×5 and grouped 3×3; compare forward and input/weight/bias gradients to a CPU float64 reference using unchanged helper tolerances.
- Original pinned PyTorch OpInfo `nn.functional.conv2d` selection: on both RTX and V100 **11 passed, one pre-existing FP64 random-tensor construction error**, exactly the same original IDs/outcomes as the earlier `be9c29d` audit. No fallback calls. Upstream tests and expected-failure metadata were not edited.
- All 27 new matched benchmarking processes passed unchanged CPU parity on output, loss, input gradient and every parameter gradient (`rtol=2e-3`, `atol=8e-5`). Vulkan timed runs have zero recorded CPU fallback.

## Matched FP32 complete-training results

Same RTX 2080 Ti, PyTorch git revision, inputs/weights, AdamW options, TF32 off, 4 CPU threads, exclusive GPU access, 8 warmup and 25 timed samples per fresh process, three independently measured processes per configuration. Reported numbers are **median of process medians**; details and distributions are in `raw.jsonl.gz`.

| Network | Base tile | New automatic tile | CUDA | Reduction vs base | New / CUDA |
|---|---:|---:|---:|---:|---:|
| U-Net 8×3×96×96 | 92.269 ms | 55.200 ms | 10.102 ms | 40.2% | 5.46× |
| U-Net 1×3×64×64 | 24.577 ms | 22.529 ms | 9.950 ms | 8.3% | 2.26× |
| ResNet18 4×3×64×64 | 70.596 ms | 66.773 ms | 15.116 ms | 5.4% | 4.42× |

Changes for smaller U-Net and ResNet are modest relative to run-to-run variation; repeat on more hardware before generalizing. For large U-Net, the register tile was previously confirmed independently against optional autotuning.

### GPU shader attribution (large U-Net)

Eager, non-deferred timestamp diagnostics, 25 steps each. These **cannot be substituted for or subtracted from deferred wall times**. Tiled main passes and split-weight kernels:

| GPU shader family | Previous base | New heuristic |
|---|---:|---:|
| Forward tiled | 15.158 ms | 6.345 ms |
| Input-gradient tiled | 18.667 ms | 7.701 ms |
| Weight-gradient split | 20.676 ms | 8.004 ms |

`all_conv2d` also includes direct, bias and reduction dispatches; this report does not measure Vulkan occupancy or PCIe DMA totals. CUDA still uses more efficient Winograd/SGEMM plans for the large 3×3 case. Forward, input-gradient and weight-gradient arithmetic remain the primary next optimization targets; split-weight scratch-buffer pressure remains.

## Remaining and next steps

1. Validate the heuristic's performance across more shapes and devices; do not force register for small, grouped, 10×10/11×11 or other unmeasured shapes. Current tests show correctness on RTX, V100 and Lavapipe, but AMD/Intel performance is unmeasured.
2. Pursue **general Winograd 3×3 or stronger convolution tiling/data reuse** with input and weight gradients, under feature/geometry guards and generic fallback.
3. Preserve strict CPU fallback rejection, unchanged original upstream cases, CPU float64 convolution oracle and matched CUDA wall/GPU profiles after each change.
4. Only after convolution optimization, begin SDPA backward work; this change does not touch attention.

Raw paired processes and upstream outcomes: [`raw.jsonl.gz`](../benchmarks/results/convolution-default-tile-2026-10-19/raw.jsonl.gz). Structured summary: [`summary.json`](../benchmarks/results/convolution-default-tile-2026-10-19/summary.json).
