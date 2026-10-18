# Native spatial operators, reductions and network coverage

Measured October 18, 2026. This extends the general operator backend with FP32
average pooling, bilinear/bicubic resize, arbitrary-axis reductions and the
boolean/selection operations used by composed attention and clamp gradients.
Dispatch depends on operator geometry, dtype and layout. There are no model-name
checks or fixed image-size paths.

## Source research and reuse

| Source | License and inspected scope | Decision |
|---|---|---|
| [Tencent ncnn](https://github.com/Tencent/ncnn/tree/44b02dabdcf97499281a4692ebd65bad73dcf8b6/src/layer/vulkan/shader), commit `44b02dabdcf97499281a4692ebd65bad73dcf8b6` | BSD-3-Clause; interpolation coefficients, interpolation and pooling shaders | Adapted only the cubic coefficient calculation from `interp_bicubic_coeffs.comp`. Attribution and the upstream license are retained in [third_party](../third_party/README.md). |
| [Burn](https://github.com/tracel-ai/burn), [MIT](https://github.com/tracel-ai/burn/blob/main/LICENSE-MIT) / [Apache-2.0](https://github.com/tracel-ai/burn/blob/main/LICENSE-APACHE) | Portable tensor kernels through CubeCL | Useful alternative architecture, but not a drop-in implementation for this C++/ATen/SPIR-V runtime. No code imported. |
| [PyTorch interpolation definitions](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/UpSample.h) | Coordinate and boundary semantics | Consulted for behavior; validated against installed ATen. No source copied. |

The retained ncnn component keeps its BSD conditions alongside the project's MIT
code. Changes include ordinary FP32 buffers, runtime geometry, Horner evaluation,
a directly evaluated fourth coefficient and clamped boundary taps. The resize
backward search/gather implementation, pooling, reductions, predicates and boolean
packing are original implementations. No external runtime or prebuilt third-party
SPIR-V is added. The inspected inference shaders did not supply the required
ATen training/backward integration.

## Implemented scope

| Family | Native behavior | Bounds / remaining work |
|---|---|---|
| Average and adaptive average pool2d | FP32 CHW/NCHW forward/backward; stride, padding, ceil mode, divisor override, count-include-pad; strided view packing | Adaptive zero-sized output retains CPU behavior; 3D pooling not added |
| Bilinear and bicubic resize2d | FP32 forward/backward; align-corners, supplied fractional scales, singleton dimensions, up/downsampling and strided views | Antialias variants and other dtypes not added |
| Sum and selected-axis mean | FP32 full/multiple/negative axes, keepdim, empty/scalar and strided inputs | Native packing supports up to eight dimensions; other dtype/layout cases retain fallback |
| Predicates and masks | FP32 scalar comparisons and isneginf; bool all, bitwise-and/not and logical-and; FP32 where with tensor/scalar branches and out overloads | Selected unsupported layouts/dtypes retain fallback; partial strided out copies can use host paths |
| Power and clamp | FP32 scalar exponent and scalar bounds, including gradient-supporting comparisons/selection | Other overloads/dtypes retain fallback |

Large pooling windows and reductions use a cooperative 128-lane workgroup;
small windows use one output owner. Resize and pooling backward gather into each
input-gradient element, avoiding floating-point atomics. Byte booleans are packed
through uint words with one writer per word, without requiring 8-bit storage.
Special-value tests cover NaN, infinity and negative bases/zero in scalar powers.

The attention case uses PyTorch's composed math path. This does not add a fused
training SDPA kernel or dropout support. `TORCH_VULKAN_NATIVE_SUM=0` retains the
sum fallback. The diagnostic registration switches
`TORCH_VULKAN_DISABLE_NATIVE_SPATIAL` and
`TORCH_VULKAN_DISABLE_NATIVE_PREDICATES` disable their registrations when present.
The benchmarks below use a separately rebuilt baseline, not these switches.

## Correctness

| Execution mode | Complete pytest suite |
|---|---:|
| RTX 2080 Ti, resident/deferred | 1,229 passed |
| Tesla V100, resident/deferred | 1,229 passed |
| Lavapipe | 1,228 passed, 1 hardware-only skip |
| RTX 2080 Ti, eager/host-backed | 1,227 passed, 2 deferred-only skips |

There are 248 added tests. The three pre-existing MSE resize warnings remain.
All nine new shader binaries passed SPIR-V validation. Tests cover operator
outputs/gradients, odd geometry, views/offsets/broadcasts, empty cases, aliases,
partial destination preservation and five complete feature-network gradients.
AMD and Intel hardware have not been tested.

The five feature-network tests compare output, input gradient and every parameter
gradient with CPU and reject generic fallback during forward/backward. All 30
before/after/CUDA timing captures also passed CPU parity (`rtol=8e-4`,
`atol=8e-5`); the largest recorded absolute difference was below `3e-7`.

Fresh captures of the existing medium U-Net, StyleNet and MLP passed output,
loss and all 42/50/10 parameter-gradient comparisons against both the saved
prior Vulkan captures and saved matched CUDA captures (`rtol=5e-4`,
`atol=2e-5`). Input and initial-weight hashes match those references. These are
single-step numerical checks, not training-convergence claims.

## Matched training measurements

RTX 2080 Ti, NVIDIA 580.178.04, PyTorch 2.14.1 (`+cpu` for Vulkan,
`+cu130` for CUDA), FP32, TF32 disabled. Vulkan residency, deferred submissions
and convolution autotuning are enabled. Each process runs 12 warmup steps and
50 measured steps. Two runs reverse backend order; the table pools their 100
samples and reports the median, not the mean of the two run medians.

Each step includes forward, squared-mean loss, backward, optimizer update and
synchronization. Inputs and initial weights are identical. Vulkan uses the
backend's fused AdamW; CUDA uses standard AdamW with `foreach=False`. These are
small complete-workload timings with different optimizer implementations, not
isolated kernel comparisons or evidence that Vulkan generally exceeds CUDA.

| Feature network | Before ms | After ms | CUDA ms | Before / after |
|---|---:|---:|---:|---:|
| Pool classifier | 3.632 | 1.318 | 1.910 | 2.76x |
| Bilinear autoencoder | 2.684 | 0.994 | 1.339 | 2.70x |
| Bicubic autoencoder | 3.026 | 1.008 | 1.336 | 3.00x |
| Reduction regressor | 6.002 | 1.490 | 1.794 | 4.03x |
| Attention block | 2.852 | 1.903 | 1.644 | 1.50x |

Attention remains about 16% slower than CUDA by pooled median. Its individual
run medians vary: Vulkan 1.833–1.999 ms, CUDA 1.457–1.734 ms. The bicubic baseline
also varies substantially (2.493–3.453 ms), while the new path is stable at
1.007–1.008 ms. Both run medians and all raw samples are retained. This measurement
does not establish the GPU-kernel cause of the remaining attention gap; that
requires a matched per-kernel profile.

The before package is rebuilt from `c6da814bd6c8eb3a3a64f603c67749807f8416d3`;
the implementation under test is `a2cc925cae8a526181f268f762c3b52f8c530a5f`.
Subsequent changes add trace phase markers and this evidence, without changing
kernels. The [manifest](../benchmarks/results/spatial-reductions-c6da814/manifest.json)
records package paths, extension and shader hashes, and zoo revision. Hashes
were unchanged throughout the full validation/timing run.

### What still falls back in these captures?

Cold traces include parity evaluation and the first optimizer step. After the
change they contain only `aten::zero_`: two AdamW state initializations per
parameter (12 calls for the pool classifier, eight for each other network).
These are startup fallbacks and remain unresolved.

Separate traces mark the first step and the following measured step. All five
**steady-state** steps have zero generic `PV_FALLBACK` calls. Before the change,
they had 6/4/4/9/5 calls respectively for pooling, bilinear, bicubic, reduction
and attention networks. A zero generic-fallback count does not prove that all
host work is absent: native helpers can perform conditional CPU copies, and
existing label validation remains host-side in applicable loss paths.

## Are the features represented in the zoo?

`audit_network_features.py` executes CPU forward/backward under TorchDispatchMode
for all 12 original zoo cases and five added cases. All 17 audits succeeded.
This establishes actual feature use, **not** that all 17 networks train on Vulkan.
The zoo is pinned at `c7c063101ec5209caf8022f5b4ea6948d1a8d6ed` and original
architectures use reduced input sizes for the audit.

| Existing cases | Observed relevant features | Coverage implication |
|---|---|---|
| ResNet18, ResNet18_224 | Axis mean, sum | Existing reduction users; earlier BatchNorm/convergence caveats still apply |
| VGG11_128 | Adaptive average pooling forward/backward, sum, dropout RNG | Exercises pooling; dropout remains a gap |
| MobileNet V3 | Axis mean, sum, dropout RNG | Exercises reductions; dropout remains a gap |
| Transformer | Attention, sum | Existing attention user; audit alone does not validate Vulkan execution |
| MLP, LSTM | Sum | LSTM still lacks its fused cell |
| Conv1D audio, embedding | Axis mean, sum | Feature presence is not whole-model GPU support |
| Conv autoencoder | Transposed convolution | Still unsupported |
| U-Net, StyleNet | None of the newly targeted feature families | Existing nearest/max-pool paths do not test bilinear/bicubic/average pooling |

The new pool classifier explicitly exercises both pooling families and backward.
Two autoencoders separately exercise bilinear and bicubic backward. The reduction
regressor combines multi-axis mean/sum with clamp/power and their gradients. The
attention block exercises dropout-free self-attention and the safe-softmax
predicates/selection on Vulkan. All five have forward/backward CPU-parity tests
and matched CUDA timing runs. Their definitions live in
[feature_networks.py](../benchmarks/feature_networks.py).

Remaining work includes dropout/RNG, transposed convolution, the fused LSTM cell,
fused training SDPA, antialiased resize, 3D pooling, wider dtype/layout coverage
and general FP16/BF16 compute. This is not a complete elimination of CPU fallback.

## Evidence and reproduction

Raw samples, parity metrics, test logs, cold/warmed traces, operator audit and
[summary](../benchmarks/results/spatial-reductions-c6da814/summary.json) are in
[`benchmarks/results/spatial-reductions-c6da814`](../benchmarks/results/spatial-reductions-c6da814).

```sh
python scripts/compile_spatial_reductions.py
cmake --build .cmake-build --parallel 4
PYTHONPATH=. TORCH_VULKAN_GPU_RESIDENT=1 TORCH_VULKAN_DEFER_SUBMISSIONS=1 \
  python -m pytest -q tests
PYTHONPATH=. TORCH_VULKAN_GPU_RESIDENT=1 TORCH_VULKAN_DEFER_SUBMISSIONS=1 \
  TORCH_VULKAN_CONV_AUTOTUNE=1 python benchmarks/probe_feature_networks.py \
  --backend vulkan --model pool_classifier --warmup 12 --repeats 50 --out /tmp/pool.json
python benchmarks/probe_feature_networks.py --backend cuda \
  --model pool_classifier --warmup 12 --repeats 50 --out /tmp/pool-cuda.json
python benchmarks/audit_network_features.py --help
```

Use the same physical GPU for both backends and the matching Python environment.
For a trace, set `TORCH_VULKAN_TRACE_FALLBACK=1`, use one warmup and one measured
step, and count calls between `PV_PHASE step=1 measured=True` and
`PV_PHASE finished`. Do not compare traced latency with untraced latency.
