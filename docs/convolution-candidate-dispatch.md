# General convolution candidate dispatch

This dispatch layer selects reusable operator implementations by convolution geometry and device capabilities. It contains no network-name checks. U-Net is one benchmark, not a supported-shape definition.

## What is implemented

Each convolution phase has its own compatible candidate set and selection:

| Phase | Candidates |
|---|---|
| Forward | Direct, depthwise, tiled implicit GEMM |
| Input gradient | Direct, tiled implicit GEMM |
| Weight gradient | Direct, parallel reduction, tiled split reduction |
| Bias gradient | Direct, parallel reduction |

A candidate describes an algorithm and, where applicable, a tile. Eligibility checks are separate from ranking: workgroup dimensions/counts, shared memory, workspace budget, phase and grouping must be compatible before selection. Tensor storage ranges and indexing limits are checked as well. Immutable physical-device properties are queried once per context, avoiding repeated driver queries during selection. If a preferred heuristic candidate cannot run, automatic selection tries a compatible fallback. An explicitly forced incompatible candidate raises an error.

The finite tile menu is shared by forward, input-gradient and weight-gradient shaders:

| Tile | Output channels × output positions × reduction | Local size | Threads | Shared memory |
|---|---:|---:|---:|---:|
| `base` | 8 × 16 × 16 | 16 × 8 × 1 | 128 | 1,536 bytes |
| `wide` | 8 × 32 × 16 | 32 × 8 × 1 | 256 | 2,560 bytes |
| `tall` | 16 × 16 × 16 | 16 × 16 × 1 | 256 | 2,048 bytes |
| `register` | 32 × 32 × 16 | 16 × 16 × 1 | 256 | 4,096 bytes |

For input gradients, the channel axis is input channels per group. For weight gradients, the position axis is input channels per group × filter height × filter width. Weight gradients split batch/spatial reduction into 256-element chunks; a separate deterministic reduction combines partials when more than one chunk is needed. There are no floating-point atomics in these convolution implementations.

Each phase has one shared GLSL body and four small wrappers selecting tile dimensions and register blocks. The register tile computes four outputs per invocation; its output tile is larger without increasing workgroup size. Shared accumulation/store helpers preserve masked tails for every phase. `scripts/compile_conv_tiles.py` rebuilds and validates those twelve SPIR-V files plus the five direct/depthwise shaders. Direct/depthwise shaders now use 128-thread groups and flatten a two-dimensional dispatch, instead of requiring 256-thread groups and a single dispatch axis.

The default heuristic still uses the base tile. The register tile is available through explicit selection or optional tuning; it does not win every geometry. See [register-tile measurements](convolution-register-tiles.md) for the follow-up comparison.

## A 10×10 filter and an 11×11 filter

Both retain their exact mathematical dimensions. A tile partitions work; it does not round a filter to a different convolution.

Filter dimensions, stride, padding, dilation, groups, batch and tensor dimensions remain runtime push constants. If both configurations select `tiled_wide`, they reuse the same compiled shader pipeline. The reduction lengths differ: `(Cin / groups) × 100` versus `(Cin / groups) × 121`. Partial tiles are masked and loaded with zero where appropriate.

This differs from the optional selection cache: two exact geometries get separate cached *candidate choices*, because their fastest candidates may differ. Creating another choice entry does not require compiling another shader. Compiled pipelines, candidate choices and descriptors bound to current tensor buffers are separate resources.

## Default selection and optional measurement

By default, the previous shape heuristics remain in use, with the base tile. There is no tuning delay and no new automatic preference for wide/tall tiles. The larger tiles are available for explicit comparison or opt-in tuning. A finite menu is not a promise that a larger tile is faster.

With `TORCH_VULKAN_CONV_AUTOTUNE=1`, an unseen configuration is measured separately for each requested phase:

1. Filter candidates for correctness and device/workspace compatibility.
2. Shortlist compatible candidates. Expensive scalar-direct candidates are excluded above eight million estimated forward multiply-accumulate terms when they are not the heuristic choice. This is bounded search, not exhaustive optimization.
3. Complete outstanding work, warm each candidate, then take five synchronized wall-time samples per candidate, rotating candidate order between rounds.
4. Compare medians. Change the current choice only when a candidate measures more than 5% faster; keep the heuristic for smaller differences.
5. Cache the selected implementation ID and execute it again to produce the observable output.

Trials include candidate-specific scratch allocation/release, execution, required transfers and synchronization. Common input/layout packing and output allocation occur before measurement and are excluded. These are **wall-time measurements, not shader-only GPU timestamps**. Measuring one phase in isolation can rank candidates differently from a complete deferred network execution; end-to-end verification is still required. Tuning adds first-use latency and offers no guaranteed speedup.

The process-local cache contains at most 256 choices, with FIFO eviction. Keys include exact convolution parameters, phase, physical-device vendor/device IDs and driver version, workspace budget, residency/deferred/timing modes, and input/output memory classes. Inputs at this point are normalized FP32 NCHW with zero offset. Keys contain no buffer addresses or model identities. The cache does not persist across processes; its lifetime is naturally restricted to the loaded shader revision. Cached choices are checked against the current compatible set before reuse.

## Controls and diagnostics

| Environment variable | Values and meaning |
|---|---|
| `TORCH_VULKAN_CONV_ALGO` | `auto` (default), `direct`, `depthwise`, `tiled`; forward selection |
| `TORCH_VULKAN_CONV_BACKWARD_ALGO` | `auto` (default), `direct`, `optimized`; backward selection |
| `TORCH_VULKAN_CONV_TILE` | `auto` (default), `base`, `wide`, `tall`, `register`; tile preference when a tiled candidate is selected |
| `TORCH_VULKAN_CONV_AUTOTUNE` | `0` (default) or `1`; tuning applies only with automatic algorithm and tile selection |
| `TORCH_VULKAN_CONV_WORKSPACE_MB` | Integer 0–4096, default 64; scratch budget in units of 1,048,576 bytes |

A tile preference alone does not force a direct/depthwise/reduction workload to become tiled. To explicitly compare a tile for forward, use `CONV_ALGO=tiled` together with `CONV_TILE`; for input/weight gradients, use `CONV_BACKWARD_ALGO=optimized` together with `CONV_TILE`. Use the full variable names above. Bias gradients have no tiled implementation. A zero workspace budget excludes tiled weight-gradient candidates requiring multiple partial buffers; candidates with no scratch remain eligible.

Python diagnostics:

```python
import torch_vulkan

print(torch_vulkan.conv_dispatch_stats())
torch_vulkan.clear_conv_dispatch_cache()
```

Counters include `entries`, `hits`, `misses`, `tunes`, `evictions`, `candidates_timed`, `candidates_timed_<phase>_<candidate>`, and `selected_<phase>_<candidate>`, such as `selected_dinput_tiled_wide`. Hits/misses concern the optional choice cache; selected counters also work with heuristic/forced selection. Clearing resets both choices and these counters, not the compiled-pipeline cache. The reported selection count does not count each internal tuning trial as an application dispatch. Candidate timing counters increment once per candidate considered in a tuning event, rather than once per timing sample.

## Supported semantics and remaining limits

- FP32 two-dimensional convolution with arbitrary valid positive rectangular filters, stride, padding, dilation, groups and depthwise channel multipliers; forward and requested input/weight/bias gradients.
- Channels-last, transposed, sliced, offset and broadcast views are normalized on the GPU to the internal contiguous NCHW layout. This is layout support through packing, not a native channels-last convolution kernel.
- ConvTranspose, FP16/BF16 convolution, empty batches and symbolic-dimension execution are not supported by this path. Invalid or unsupported configurations raise explicit errors; there is no claim of an automatic CPU fallback for every unsupported convolution.
- Tensor/view indexing is limited to unsigned 32-bit ranges; signed spatial-coordinate arithmetic is checked. Physical storage-buffer and workgroup limits still apply. The two-axis direct fallback removes a single-axis dispatch restriction, not all size limits.
- Workspace-dependent candidates respect both the configured budget and physical buffer/indexing limits. Workspace-free alternatives remain available when compatible.
- The native convolution kernels do not yet use tensor/cooperative-matrix instructions, a dedicated pointwise-GEMM route, or lower-precision acceleration. General shape support does not imply cuDNN/MIOpen performance parity.
- Vulkan feature checks do not substitute for numerical and performance testing on AMD/Intel hardware.

## Comparison with established backends

The common design is **operator semantics → compatible implementations → shape/device selection → reusable compiled code → current tensor bindings**. Reusing source code does not require every shape to use the same compiled kernel.

| Backend/source | Selection and reuse | Relation to this implementation |
|---|---|---|
| PyTorch CUDA/cuDNN | ATen selects a convolution backend. cuDNN execution plans are selected through heuristics or benchmark search, with configuration-based caching and workspace/precision/determinism constraints. Forward, backward-data and backward-filter can choose different plans. | Separate phases, applicability checks and cached choices follow that separation. Our candidate inventory and tuning policy are much smaller. |
| ROCm MIOpen | Solvers expose applicability constraints; implementation families include runtime-dimension kernels and compile-time-specialized kernels. FindDb records solution selection information; PerfDb stores solver tuning configurations. | Exact-shape choice caching and a finite tile family are compatible with general operators. Our process-local cache is not a replacement for MIOpen's persistent databases. |
| llama.cpp Vulkan | Convolution selects among output tile shapes by channels/spatial work and hardware. Its reviewed shader specializes exact filter width/height, stride, padding, dilation and alignment, so 10×10 and 11×11 can share source/tile but use distinct compiled pipelines. Matmul has device-aware tile selectors and aligned/unaligned variants. | Our filter dimensions stay runtime parameters, so selected tile pipelines reuse more broadly. We now have a tile menu and selection cache, but not the same hardware-specific implementation breadth. |

Primary references:

- PyTorch [`Convolution.cpp`](https://github.com/pytorch/pytorch/blob/9664c41ccfc87a9661cb2b177985131948306876/aten/src/ATen/native/Convolution.cpp#L1297-L1410) and [`cudnn/Conv_v8.cpp`](https://github.com/pytorch/pytorch/blob/9664c41ccfc87a9661cb2b177985131948306876/aten/src/ATen/native/cudnn/Conv_v8.cpp): reviewed snapshot `9664c41ccfc87a9661cb2b177985131948306876`.
- MIOpen [Find and Immediate modes](https://rocm.docs.amd.com/projects/MIOpen/en/latest/how-to/find-and-immediate.html), [performance database](https://rocm.docs.amd.com/projects/MIOpen/en/latest/conceptual/perfdb.html), [`conv_asm_implicit_gemm_v4r1_dynamic.cpp`](https://github.com/ROCm/rocm-libraries/blob/6bb613ae7408a7b90c6db1f5dc7d2f7c41cb227a/projects/miopen/src/solver/conv/conv_asm_implicit_gemm_v4r1_dynamic.cpp), and [`conv_hip_implicit_gemm_fwd_v4r4_xdlops_padded_gemm.cpp`](https://github.com/ROCm/rocm-libraries/blob/6bb613ae7408a7b90c6db1f5dc7d2f7c41cb227a/projects/miopen/src/solver/conv/conv_hip_implicit_gemm_fwd_v4r4_xdlops_padded_gemm.cpp).
- llama.cpp [`ggml_vk_conv_select_shape`](https://github.com/ggml-org/llama.cpp/blob/5aeac37a35b44fde4bdffaec9159f2fcb253b622/ggml/src/ggml-vulkan/ggml-vulkan.cpp#L8656-L8688), [exact convolution pipeline-state key](https://github.com/ggml-org/llama.cpp/blob/5aeac37a35b44fde4bdffaec9159f2fcb253b622/ggml/src/ggml-vulkan/ggml-vulkan.cpp#L9277-L9335), [`conv2d_mm.comp`](https://github.com/ggml-org/llama.cpp/blob/5aeac37a35b44fde4bdffaec9159f2fcb253b622/ggml/src/ggml-vulkan/vulkan-shaders/conv2d_mm.comp#L75-L104), and [matmul tile selectors](https://github.com/ggml-org/llama.cpp/blob/5aeac37a35b44fde4bdffaec9159f2fcb253b622/ggml/src/ggml-vulkan/ggml-vulkan.cpp#L2813-L2859): reviewed snapshot `5aeac37a35b44fde4bdffaec9159f2fcb253b622`.

llama.cpp is an inference-oriented reference. Its scheduler assumptions must not be copied over PyTorch saved tensors, mutation/version tracking or autograd lifetimes.

## Original dispatch measurements

The tables and validation counts below describe the original three-tile dispatch change. For the subsequent four-output register tile, 476-test validation and new matched CUDA measurements, see [the register-tile report](convolution-register-tiles.md).

RTX 2080 Ti, FP32, TF32 disabled. Eight warmups; 15 samples per run. Vulkan base/tuned/tuned/base order with CUDA between the two tuned runs; Vulkan medians pool 30 samples, CUDA uses 15. Residency and deferred submission enabled. Base means the current code with tuning disabled and the original tile preference, not an older checkout. First-use compilation/tuning is excluded.

| Case | Phase | Base ms | Tuned ms | CUDA ms | Base/tuned |
|---|---|---:|---:|---:|---:|
| pointwise | forward | 0.0823 | 0.0845 | 0.0408 | 0.97x |
| pointwise | dinput | 0.1082 | 0.1043 | 0.0452 | 1.04x |
| pointwise | dweight | 0.1346 | 0.1212 | 0.0762 | 1.11x |
| pointwise | dbias | 0.0840 | 0.0876 | 0.0430 | 0.96x |
| rect3x5 | forward | 0.0976 | 0.0957 | 0.0430 | 1.02x |
| rect3x5 | dinput | 0.1605 | 0.1329 | 0.0503 | 1.21x |
| rect3x5 | dweight | 0.1271 | 0.1230 | 0.0678 | 1.03x |
| rect3x5 | dbias | 0.0813 | 0.0846 | 0.0388 | 0.96x |
| filter10x10 | forward | 0.3583 | 0.2860 | 0.1213 | 1.25x |
| filter10x10 | dinput | 0.7622 | 0.5446 | 0.1243 | 1.40x |
| filter10x10 | dweight | 0.3889 | 0.3418 | 0.0940 | 1.14x |
| filter10x10 | dbias | 0.0832 | 0.1150 | 0.0394 | 0.72x |
| filter11x11 | forward | 0.4231 | 0.3526 | 0.1222 | 1.20x |
| filter11x11 | dinput | 0.9366 | 0.6757 | 0.1543 | 1.39x |
| filter11x11 | dweight | 0.4675 | 0.4055 | 0.1013 | 1.15x |
| filter11x11 | dbias | 0.1116 | 0.1472 | 0.0387 | 0.76x |
| grouped | forward | 0.1183 | 0.1292 | 0.0668 | 0.92x |
| grouped | dinput | 0.1621 | 0.1814 | 0.0880 | 0.89x |
| grouped | dweight | 0.1880 | 0.1770 | 0.0632 | 1.06x |
| grouped | dbias | 0.1115 | 0.1369 | 0.0403 | 0.81x |
| depthwise | forward | 0.0700 | 0.0961 | 0.0234 | 0.73x |
| depthwise | dinput | 0.0942 | 0.1291 | 0.0353 | 0.73x |
| depthwise | dweight | 0.0919 | 0.1256 | 0.0351 | 0.73x |
| depthwise | dbias | 0.0834 | 0.1181 | 0.0407 | 0.71x |
| conv3x3_large | forward | 0.4543 | 0.3083 | 0.0659 | 1.47x |
| conv3x3_large | dinput | 0.8407 | 0.4836 | 0.0642 | 1.74x |
| conv3x3_large | dweight | 0.6165 | 0.4378 | 0.1326 | 1.41x |
| conv3x3_large | dbias | 0.1267 | 0.1336 | 0.0387 | 0.95x |
| conv3x3_small | forward | 0.1545 | 0.1584 | 0.0520 | 0.98x |
| conv3x3_small | dinput | 0.2677 | 0.2174 | 0.0562 | 1.23x |
| conv3x3_small | dweight | 0.2073 | 0.1844 | 0.0678 | 1.12x |
| conv3x3_small | dbias | 0.1073 | 0.1151 | 0.0392 | 0.93x |

These are isolated operator phases. Small/depthwise cases sometimes regress despite unchanged selected kernels; the cache lookup and runtime/measurement state affect total latency. A separate same-process depthwise ABBA probe measured roughly 3 microseconds additional cache-enabled overhead, which does not explain every cross-process difference. Tuning stays opt-in; these data do not justify a universal default switch.

### Complete models

Same source architectures from model-zoo commit `c7c063101ec5209caf8022f5b4ea6948d1a8d6ed`. U-Net input [1,3,64,64]; StyleNet [1,3,36,36]. Same interleaved sampling. Vulkan uses its opt-in fused AdamW for both configurations; CUDA uses standard AdamW with foreach=False, so training compares these specific optimizer implementations. This is not a comparison against CUDA fused AdamW. Training includes forward, loss, backward and optimizer. Run-to-run training variation remains visible in raw samples.

| Model | Mode | Base ms | Tuned ms | CUDA ms | Base/tuned |
|---|---|---:|---:|---:|---:|
| unet | inference | 2.560 | 2.173 | 1.088 | 1.18x |
| unet | training | 16.059 | 13.398 | 7.631 | 1.20x |
| style_net | inference | 40.843 | 39.983 | 1.539 | 1.02x |
| style_net | training | 189.955 | 145.284 | 12.214 | 1.31x |

The separate `standard_optimizer/` diagnostic used standard AdamW on Vulkan too; its CPU fallbacks dominate training and it is not the configuration in this table. These timings and matching initial losses are not a claim of long-run training equivalence.

### Where CUDA is still faster

Vulkan GPU timestamps use eager execution with transfers measured separately; the tall tile matches the selected heavy-workload implementation. CUDA values sum actual GPU kernel durations from five profiler iterations, excluding launch gaps. These separate instrumented runs explain computation cost; they are not end-to-end timings. The split-weight Vulkan interval includes the inter-kernel barrier and finalizer.

| Case | Phase | Vulkan GPU ms | CUDA GPU ms | Vulkan upload/download ms |
|---|---|---:|---:|---:|
| filter10x10 | forward | 0.2188 | 0.1171 | 0.0336/0.0729 |
| filter10x10 | dinput | 0.4349 | 0.1117 | 0.0378/0.0262 |
| filter10x10 | dweight | 0.2132 | 0.0564 | 0.0348/0.0630 |
| filter10x10 | dbias | 0.0083 | 0.0069 | 0.0222/0.0035 |
| filter11x11 | forward | 0.2592 | 0.1170 | 0.0366/0.0565 |
| filter11x11 | dinput | 0.5239 | 0.1223 | 0.0403/0.0247 |
| filter11x11 | dweight | 0.2414 | 0.0586 | 0.0338/0.0714 |
| filter11x11 | dbias | 0.0077 | 0.0062 | 0.0215/0.0036 |
| conv3x3_large | forward | 0.2808 | 0.0408 | 0.0911/0.6472 |
| conv3x3_large | dinput | 0.4629 | 0.0288 | 0.1709/0.3328 |
| conv3x3_large | dweight | 0.3498 | 0.0931 | 0.2442/0.0154 |
| conv3x3_large | dbias | 0.0230 | 0.0065 | 0.1635/0.0032 |

The recorded CUDA kernels show FFT-based forward for both 10x10 and 11x11. Their input-gradient choices differ: FFT for 10x10, `dgrad_engine` for 11x11. The tested large 3x3 uses Winograd forward/input gradients and Winograd plus SGEMM for weight gradients. Our corresponding paths remain tiled implicit GEMM. Thus algorithm-family breadth, not only tile choice or submission overhead, is a concrete remaining gap. These choices are specific to the measured device/software/configuration.

### Validation

- rtx: 422 passed, 3 warnings in 12.34s
- v100: 422 passed, 3 warnings in 11.34s
- lavapipe: 421 passed, 1 skipped, 3 warnings in 10.64s
- resident: 422 passed, 3 warnings in 12.63s

Includes 61 new dispatch tests: forced tiles, ragged boundaries, filters9/10/11/12, groups/stride/dilation/layouts, gradients and masks, pipeline reuse, tuning-cache reuse, workspace changes, explicit-control bypass, signed-coordinate rejection and two-axis direct dispatch. All14 shader binaries pass spirv-val. Three pre-existing MSE output-resize warnings remain; Lavapipe skips the hardware-acceleration assertion. AMD/Intel hardware remains untested. The previously observed reduced-size ResNet18 end-to-end numerical mismatch was not resolved by this convolution work. A subsequent [BatchNorm precision investigation](batch-norm-precision.md) separates local operator defects from FP32 amplification using matched CPU FP64 and CUDA references; it does not establish training convergence.


Raw measurements: [`benchmarks/results/conv-dispatch`](../benchmarks/results/conv-dispatch/).
Reproduce the summary with `python3 benchmarks/summarize_conv_dispatch.py`.
The JSON records the parent Git revision and dirty-worktree status; the convolution source/SPIR-V SHA256 identifies the exact measured implementation.
