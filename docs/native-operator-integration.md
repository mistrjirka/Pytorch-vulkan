# Combined native-operator and runtime validation

18 October 2026. Integration of the independently developed operator commit
`960e8f44a5d66fbdb4215db3c1a311b239e450ed` with published runtime/convolution commit
`f23ad1e9c9b19da6e9db4330d6813e63cd392305`. Both already contain constant-buffer
commit `ef6aaf3`.

The local integration commit is `25f664ee899e4c3db42d6085048377773fb89607`, with tree
`78432a84fbb976f62c1d0bd880ffb8a9e67d0cb8`. Validation used that source tree;
subsequent changes only add this report, evidence and README test counts.

Only CMakeLists.txt overlapped between the two branches. The combined tree keeps
all four new operator source entries and the barrier-batching patch invocation.
Every other changed source/shader blob from each branch is unchanged. The build
used the isolated worktree and its own Kompute dependency sources. The primary
checkout was left intact. No operator dispatch rule or model-specific kernel
was introduced by integration.

## Correctness

| Configuration | Result |
|---|---:|
| RTX 2080 Ti, GPU-resident / deferred | 981 passed |
| Tesla V100, GPU-resident / deferred | 981 passed |
| Lavapipe, GPU-resident / deferred | 980 passed, 1 hardware-only skip |
| RTX 2080 Ti, eager / host-backed | 979 passed, 2 deferred-only skips |

All eight integrated SPIR-V modules pass spirv-val. Three existing PyTorch MSE
resizing warnings remain; the logs retain them. AMD/Intel hardware and Khronos
synchronization validation remain untested.

Existing U-Net, StyleNet and MLP outputs, losses and all 102 parameter gradients
match the pre-integration Vulkan captures exactly. They also pass comparisons
against saved, identical-input CUDA captures at rtol=5e-4, atol=2e-5. Fresh basic
and attention diffusion probes compare output, input gradients and parameter
gradients against CPU; each maximum absolute error is below 2e-5. The operator
suite separately covers layouts, offsets, groups, axes and backward masks.

## Fresh matched latency

RTX 2080 Ti, FP32, TF32 disabled. Two processes per backend and workload, each
with 12 warmups and 50 synchronized samples. Backend order is Vulkan/CUDA then
CUDA/Vulkan. Values below are medians pooled over 100 samples. Training includes
forward, MSE or model-zoo L1 loss, backward and optimizer: Vulkan fused AdamW,
CUDA standard AdamW with foreach=False.

| Workload | Vulkan inference (ms) | CUDA inference (ms) | Vulkan training (ms) | CUDA training (ms) |
|---|---:|---:|---:|---:|
| Basic diffusion U-Net | 1.512 | 1.191 | 10.898 | 9.536 |
| Diffusion U-Net with attention | 2.486 | 1.436 | 15.576 | 11.555 |
| StyleNet | 4.912 | 1.567 | 16.335 | 12.120 |

The diffusion probes use batch 1, 16x16 inputs; StyleNet uses batch 1, 36x36.
Diffusion training is about 14% and 35% slower than CUDA in these captures.
These are small synthetic diffusion networks, not full Stable Diffusion.

StyleNet Vulkan training run medians differ: 14.000 and 16.499 ms. Basic diffusion
inference also varies (1.263 and 1.515 ms). Raw sample ranges are preserved. This
check establishes combined correctness and current latency, not an attributable
integration speedup or a new GPU bottleneck profile. The earlier convolution
input-gradient profile remains a separate measurement.

General GroupNorm, unary math, BMM and axis-softmax support now coexist with
bounded immutable constants, batched barriers and unit-stride convolution input
gradients. Fused attention training, broader dtype support and remaining CPU
fallbacks are still separate work; see the
[operator coverage and license report](diffusion-native-operators.md).

## Evidence and reproduction

[Raw results and manifest](../benchmarks/results/native-integration-f23ad1e/)
include build/test logs, per-run samples, parity comparisons, the loaded
extension hash and all shader hashes. Large parity tensors remain in the ignored
`benchmarks/results/nsys/native-integration-f23ad1e/` directory. The model-zoo
revision is `c7c063101ec5209caf8022f5b4ea6948d1a8d6ed`.

```bash
TORCH_VULKAN_GPU_RESIDENT=1 TORCH_VULKAN_DEFER_SUBMISSIONS=1 \
TORCH_VULKAN_CONV_AUTOTUNE=1 python -m pytest -q tests

# Repeat twice, alternating backend order; add --attention for the second model.
NVIDIA_TF32_OVERRIDE=0 TORCH_VULKAN_GPU_RESIDENT=1 \
TORCH_VULKAN_DEFER_SUBMISSIONS=1 TORCH_VULKAN_CONV_AUTOTUNE=1 \
python benchmarks/probe_diffusion_train.py --backend vulkan \
  --warmup 12 --repeats 50 --out diffusion.json
```

For CUDA, select the CUDA-enabled Python environment and `--backend cuda` on
the same GPU. For StyleNet use `benchmarks/compare_cuda_vulkan.py --model
style_net --scale medium --warmup 12 --repeats 50`, the model-zoo path, and
`--optimizer vulkan_fused` or `--optimizer standard` respectively. V100 uses
`TORCH_VULKAN_PHYSICAL_DEVICE_INDEX=1`; Lavapipe selects its ICD; the eager check
sets both residency/deferred flags to zero.
