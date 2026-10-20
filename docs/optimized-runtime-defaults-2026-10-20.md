# Vulkan optimized execution is now the default

20 October 2026. Previous optimized Vulkan execution required users to
opt into both resident GPU buffers and deferred command submission with
process-level environment variables. Those settings gave substantial
performance gains but ordinary application code silently used the slow
eager host-mirroring path unless the user knew the environment names.

## New normal behavior

GPU residence and deferred command submission are now enabled in C++
without any special environment configuration. Native operator fast paths,
bounded descriptor/pipeline caching and direct-output kernels were already
default-on and remain so. For users selecting the separate native fused
Vulkan AdamW optimizer, horizontal multi-tensor scheduling is also on,
with capability-checked 8-wide/4-wide/single-device-tensor handling.

Application code can inspect the actual execution mode:

    import torch_vulkan
    print(torch_vulkan.performance_options())

Expected on normal supported hardware:

    {'gpu_resident': True, 'deferred_submissions': True,
     'adamw_multitensor_requested': True}

The request flag for the native optimizer does not imply that every
physical device supports eight-wide grouping; the native dispatcher
checks actual Vulkan capabilities and uses a safe fallback where needed.

## Intentional API boundary

The backend does not implicitly replace torch.optim.AdamW. The faster
fused implementation is selected using the ordinary Python class:

    from torch_vulkan.optim import AdamW
    optimizer = AdamW(model.parameters(), lr=1e-4, amsgrad=True)

Automatically replacing the standard class could change supported
optimizer options and state behavior. This is a semantic API selection,
not a hidden performance environment flag.

Convolution optimized heuristics and verified kernels are enabled by
default. Timing-based per-shape convolution autotuning remains off,
because its startup/kernel-compile overhead is substantial and not
a universally valid throughput win.

## Compatibility and diagnostics

Existing environment controls remain available but are documented
as diagnostic opt-outs, rather than prerequisites:

- TORCH_VULKAN_GPU_RESIDENT=0: legacy eager nonresident mode, including
  automatically disabling deferred submissions.
- TORCH_VULKAN_DEFER_SUBMISSIONS=0: eager submission with GPU residency.
- TORCH_VULKAN_ADAMW_DISABLE_MULTITENSOR=1: original one-update-per-tensor
  native AdamW path.
- TORCH_VULKAN_ADAMW_MULTITENSOR_WIDTH=4: restrict horizontal grouping to four.

Explicitly requesting deferred submissions while disabling residency
produces an explanatory error. Values other than 0 or 1 for residency
and deferred submission are rejected.

The README provides a complete default-policy and troubleshooting table.
No per-model hardcoded optimization setting is required.

## Verification

Full repository correctness suite, invoked with both residency and
deferred-submission environment variables ABSENT, and the optimized
execution path thus active by default:

| Device | Result |
|---|---|
| RTX 2080 Ti | 3,911 passed |
| Tesla V100 | 3,911 passed |
| Lavapipe | 3,910 passed, 1 skipped |

The test suite previously skipped several deferred-only tests unless
the environment variables were manually set. Those tests now inspect
the actual runtime mode instead, and run automatically in the default
configuration. Additional subprocess tests verify default, residency-off,
deferred-off, explicit-on, invalid, and conflicting configurations.

Benchmark with the same RTX 2080 Ti and identical fused AMSGrad optimizer:
50 tensors, 2,420,547 FP32 parameters, 12 warmups and 65
synchronized timed updates per fresh process, 3 processes per setting.

| Execution setup | Median of process medians |
|---|---:|
| No performance environment variables | 0.743 ms |
| Explicit resident/deferred flags set to 1 | 0.736 ms |

The approximately 0.007 ms difference is within run-to-run variability;
the default runs the same optimized implementation, without any required
user configuration. RTX 3090 and AMD GPU were not used.

Machine-readable results and raw timed samples:
- benchmarks/results/default-performance-2026-10-20/summary.json
- benchmarks/results/default-performance-2026-10-20/raw-timings.jsonl.gz

No claims are made that timing-based convolution autotuning, host-visible
buffer storage, or profiling instrumentation are universally faster;
these remain opt-in diagnostics.
