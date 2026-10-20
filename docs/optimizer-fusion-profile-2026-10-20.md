# Optimizer fusion and generality: Vulkan vs CUDA profiling

20 October 2026. Source baseline: 938d13d. The intent is to preserve standard
PyTorch semantics and broad Vulkan tensor support, not optimize one neural
network or substitute one optimizer algorithm for another.

## Workload and method

Same deterministic FP32 Lab02 MyNet, 50 parameter tensors containing 2,420,547
elements, batch 1, 128x128 patches. Same initialized weights and input SHA256
for all whole-network RTX runs. GPU resident inputs, gradients and weights;
zero recorded CPU numerical fallback or host reads in Vulkan timing runs.

Normal full-network timings: 12 warmups, 60-80 synchronized timed steps
per process. Optimizer-only: 12 warmups, 100 synchronized static-gradient
GPU-resident updates, no forward/backward operations. Stage wall times use
separate synchronization, outside the primary benchmark. GPU shader timestamps
use separately instrumented, eager-mode V100 runs: those times are NOT
directly subtractable from normal deferred-mode wall timings.

Original video script and full video encoding are NOT executed.

## RTX 2080 Ti full-network training

| Optimizer | Vulkan median | CUDA median |
|---|---:|---:|
| Adam, standard foreach=False | 70.53 ms (two processes) | 14.22 ms |
| Adam, foreach=True | 61.74 ms | 12.41 ms |
| Adam, fused=True | Not implemented on Vulkan | 9.57 ms |
| AdamW, standard foreach=False | 71.02 ms (two processes) | 13.96 ms |
| AdamW, foreach=True | 63.24 ms | Not measured |
| AdamW, fused=True on CUDA / native opt-in Vulkan AdamW | 49.58 ms (Vulkan native, two processes) | 8.71 ms (CUDA fused) |

The final line compares AdamW-algorithm variants, not interchangeable
optimizer APIs. The opt-in native Vulkan AdamW lacks some standard options.
Do not substitute AdamW for Adam: Adam has coupled weight decay, while AdamW
uses decoupled decay.

V100 full-network Vulkan: standard Adam 67.07 ms, foreach Adam 48.93 ms;
standard AdamW 67.23 ms, opt-in native AdamW 45.71 ms.

## RTX 2080 Ti optimizer-only latency, 100 steps

| Backend and optimizer | Median |
|---|---:|
| Vulkan Adam, standard | 15.82 ms |
| Vulkan Adam, foreach=True | 7.66 ms |
| CUDA Adam, standard | 4.57 ms |
| CUDA Adam, fused=True | 0.353 ms |
| Vulkan AdamW, standard | 14.64 ms |
| Vulkan AdamW, foreach=True | 5.16 ms |
| Vulkan opt-in native AdamW | 0.852 ms |
| CUDA AdamW, fused=True | 0.353 ms |

V100 optimizer-only: standard Adam 11.81 ms, foreach Adam 8.31 ms;
standard AdamW 11.50 ms, foreach AdamW 5.34 ms; opt-in native AdamW 0.832 ms.

## Root-cause GPU/host instrumentation

V100 GPU timestamps in eager instrumented mode:

| Optimizer implementation | Shader compute per step | GPU shader dispatches |
|---|---:|---:|
| Standard Adam | 3.18 ms | 400 |
| Standard Adam foreach=True | 3.25 ms | 400 |
| Standard AdamW | 3.33 ms | 400 |
| Standard AdamW foreach=True | 3.37 ms | 400 |

Standard Adam executes 150 optimizer_pointwise kernels per optimizer step,
plus another 250 kernel dispatches for elementary state/parameter operations.
Foreach reduces Python/ATen overhead but still issues the same number of
Vulkan compute kernels. Eager mode is slower due to per-operation submission
and timestamp profiling, so the GPU shader totals do not precisely partition
the ordinary deferred wall times.

The native AdamW implementation emits one Vulkan shader dispatch per parameter
tensor, about 50 total. Its batch path does not expose per-op GPU timestamps
in our existing profiler. Its 0.83-0.85 ms optimizer-only wall latency is a
separate uninstrumented measurement, NOT a measured 0.83 ms GPU shader total.

Algorithm cache counters over the RTX isolated optimizer process:
- Standard Adam: 41,287 bound-algorithm cache misses, 40,775 evictions,
  but only 6 pipeline misses.
- Native AdamW: 150 bound-algorithm misses, zero evictions,
  only 2 pipeline misses.

This confirms repeated buffer-specific descriptor/algorithm construction,
not shader recompilation. Increasing the 512-entry cache would not fix
ephemeral intermediate tensor identities in the general case.

## What a genuinely general optimizer-fusion implementation should do

1. Implement the standard PyTorch ATen fused operator interfaces for Vulkan:
   aten::_fused_adam_ and aten::_fused_adamw_. PyTorch's existing
   torch.optim.Adam / AdamW with fused=True should call them without a
   custom optimizer class or monkey patching.
2. Preserve the exact algorithm and options: coupled vs decoupled weight decay,
   parameter groups, per-parameter missing gradients, AMSGrad, maximize,
   varying beta/epsilon/learning rates, device-resident step counters and
   gradient-scaler/found-inf handling. Preserve PyTorch-required autograd
   limitations and dtype contracts; never silently substitute AdamW for Adam.
3. First implement portable vertical fusion: one complete update per tensor,
   with heterogeneously sized tensors handled by the standard tensor list API.
   Dispatch batches may share Vulkan command buffers while preserving read/
   write barriers and in-flight resource lifetime. Bound descriptor reuse
   must honor GPU completion.
4. Only optionally use horizontal multi-tensor fusion when Vulkan capability
   queries permit descriptor arrays/descriptor indexing and the device can
   bind the necessary resources. Preserve a portable per-tensor GPU path.
5. When fused=True explicitly requests an unsupported configuration, report
   the specific unsupported feature rather than silently changing
   numerical behavior. When fusion is not requested, leave ordinary PyTorch
   Adam/AdamW, foreach True/False and other standard optimizers operational.
6. No model names, fixed layer count, or hardcoded shapes should enter backend
   dispatch logic; support heterogeneous shapes, empty/missing gradients,
   non-contiguous cases or safe GPU alternatives, and large tensors.

PyTorch on Vulkan currently does not implement these fused ATen operators:
strict CPU-fallback mode makes fused=True fail. The existing custom Vulkan
AdamW path is a useful speed reference, not the right general API.

As an immediate, semantics-preserving option, torch.optim.Adam(...,
foreach=True) and torch.optim.AdamW(..., foreach=True) already work on this
backend, but may use more peak temporary memory than single-tensor PyTorch
optimizers, as documented by PyTorch.

## Regression tests added

tests/test_optimizer_foreach_generality.py validates 16 combinations of
ordinary Adam/AdamW, foreach False/True, AMSGrad False/True, maximize
False/True; heterogeneous per-group hyperparameters, missing gradients and
seven updates. GPU final parameter values, step counters and moments match
CPU PyTorch, with zero CPU numerical fallback.

The combined optimizer suite including existing fused AdamW and primitive
checks passed 219 tests on EACH of RTX 2080 Ti, V100 and Lavapipe.

## Reproduction and privacy

Sources:
- benchmarks/profile_optimizer_only.py
- benchmarks/profile_nis_lab02.py
- benchmarks/results/optimizer-fusion-profile-2026-10-20/summary.json
- Per-run timing JSON in the rtx and v100 subfolders.

Set NIS_LAB02_SOURCE to a local directory containing the original Lab02
models.py. That private source is intentionally NOT included in this
repository. Only measured architecture metadata and SHA256 hashes are saved.

The RTX 3090 was not touched; the RX 7800 XT was unavailable in this round.
No underlying Vulkan optimizer implementation was modified in this audit.

PyTorch docs:
https://docs.pytorch.org/docs/stable/generated/torch.optim.Adam.html
https://docs.pytorch.org/docs/stable/generated/torch.optim.AdamW

Khronos descriptor lifetime/feature guidance:
https://docs.vulkan.org/samples/latest/samples/extensions/descriptor_indexing/README.html
