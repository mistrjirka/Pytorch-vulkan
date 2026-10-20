# Generic Vulkan fused AdamW with AMSGrad — 20 October 2026

## Implementation

Our existing fused Vulkan AdamW was restricted to ordinary AdamW.
It therefore could not handle the AMSGrad optimizer in the broad network
benchmarks. This change extends the opt-in Vulkan optimizer to AMSGrad
without changing the default PyTorch AdamW behavior.

- New general shader csrc/shaders/adamw_fused_amsgrad.comp with a fifth
  storage buffer for max_exp_avg_sq.
- Shared C++ batch dispatcher and new native primitive
  torch_vulkan::adamw_amsgrad_step_batch_.
- Python torch_vulkan.optim.AdamW now accepts amsgrad=True.
- Computes m, v, running maximum of v, decoupled weight decay,
  bias-corrected AdamW weights in one kernel per parameter tensor.
- Explicit torch.maximum-style NaN propagation to ensure the AMSGrad
  maximum state remains correct across Vulkan drivers.
- Existing non-AMSGrad shader and original operator ABI unchanged.
- Batched/unbatched dispatch, multi-parameter groups, empty tensors,
  missing gradients and state checkpoint import/export are tested.

This remains an explicit opt-in FP32, contiguous zero-offset GPU optimizer.
Standard torch.optim.AdamW is never silently monkey-patched.
Sparse gradients, maximize, mixed precision, capture and other optional
PyTorch optimizer behaviors are not claimed as supported.

## Numerical and regression verification

Targeted tests compare multiple shapes, parameters and all three
AMSGrad moment buffers against standard PyTorch AdamW over 11 steps.
Additional tests cover no-gradient parameters, NaNs, chained autograd
updates, grouped parameter hyperparameters, checkpoint save/reload and
resuming a standard PyTorch checkpoint whose step is a scalar Tensor.

Full repository suite after the new shader and NaN tests:
- RTX 2080 Ti: 3,889 passed, 14 warnings
- Tesla V100: 3,889 passed, 14 warnings
- Lavapipe: 3,888 passed, one skipped, 14 warnings

Warnings are deprecation and sparse-tensor invariant advisories;
there were no numerical CPU fallback calls in the GPU test cases.

## Matched optimizer-only performance

50 heterogeneous tensors, 2,420,547 FP32 elements and precomputed
gradients. Each independent process executes 12 warmups followed by
75 synchronized optimizer updates. The result below is the median
of process medians (3 RTX runs and 2 V100 runs per setting).

| Device | Standard Vulkan AMSGrad | Fused Vulkan AMSGrad | Speedup |
|---|---:|---:|---:|
| RTX 2080 Ti | 9.089 ms | 0.980 ms | 9.27x |
| Tesla V100 | 9.021 ms | 0.966 ms | 9.34x |

On the same RTX 2080 Ti, the earlier CUDA fused AMSGrad reference
measured around 0.427 ms. The Vulkan fused optimizer is still about
2.3x slower in the standalone case. No RTX 3090 was used.

## Complete FP32 forward/backward/optimizer training steps

Three matched medium network cases on the same RTX 2080 Ti.
Two independent processes per configuration; eight warmup iterations
and 20 synchronized timed iterations per process. Includes loss,
backward, dynamic gradients and optimizer state updates. Both CUDA
and Vulkan numerical forward/gradient checks pass before timing;
Vulkan CPU fallback counters stay at zero.

| Network | Standard Vulkan AMSGrad | Fused Vulkan AMSGrad | Vulkan speedup | CUDA fused AMSGrad |
|---|---:|---:|---:|---:|
| MLP | 5.359 ms | 3.954 ms | 1.36x | 1.537 ms |
| U-Net | 23.398 ms | 18.124 ms | 1.29x | 6.173 ms |
| Transformer | 65.442 ms | 49.959 ms | 1.31x | 11.435 ms |

Measured full-step performance therefore improves about 26%, 23%,
and 24%, respectively, while maintaining AMSGrad semantics.
Vulkan is still 2.57x, 2.94x, and 4.37x slower than fused CUDA
in those same complete workloads. Optimizer fusion does not fix
the Transformer attention-backward algorithm.

## Reproduction

Select the new optimizer explicitly in application code:

    from torch_vulkan.optim import AdamW
    optimizer = AdamW(model.parameters(), lr=1e-4,
                      amsgrad=True, batched=True)

The opt-in selection in benchmarks/benchmark_network_matrix.py is
--optimizer-implementation vulkan_fused_amsgrad. The unchanged
default is standard AdamW. On CUDA the explicit comparison is
--optimizer-implementation cuda_fused_amsgrad.

To run the complete comparative set:

    .venv/bin/python benchmarks/benchmark_amsgrad_models.py \
      --rounds 2 --warmup 8 --repeats 20 \
      --cases mlp unet transformer --scale medium \
      --out /tmp/amsgrad-model-benchmark

The benchmark assumes RTX 2080 Ti UUID
GPU-30acbf2b-a41a-8b88-e298-fe66c4138b29 and Vulkan index 0.
Change those settings on other systems. The optimizer-only harness
also needs the local MyNet source configured; no private model
files are included in this commit.

Saved machine-readable results:
- benchmarks/results/amsgrad-fused-2026-10-20/summary.json
- benchmarks/results/amsgrad-fused-2026-10-20/raw-timed-runs.jsonl.gz

## Remaining work

The new fused optimizer addresses the largest reusable known
optimizer gap in our prior audit, but is not the default AdamW
or a universal replacement for all PyTorch optimizer options.
Extend dtype/layout/capability handling only with separate numerical
validation. Next larger algorithmic target is attention backward,
where the previous profiler attributed about 77% of Vulkan
Transformer reference-shape shader time to sdpa_training.

ROCm/RDNA3 speedup is not claimed: the AMD machine was unavailable
for a new matched test.
