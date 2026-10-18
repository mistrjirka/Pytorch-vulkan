# CUDA coverage and Vulkan universality audit

Audited October 18, 2026, against backend commit
`b80a1f6dcb17287b14404ec802e1b725878ef1c6`.

**The backend is not as universal as PyTorch CUDA.** Its operator dispatch is
reusable across models and shapes within a supported domain. Its dtype,
operator, integration and hardware coverage is much narrower. CPU fallback
extends what can execute, but does not make it GPU-native or guarantee that
every unsupported configuration works.

This is an investigation. The backend implementation was not changed. The
correctness defects below remain present at the audited revision.

## Evidence and scope

Three independent OpenCode v2.0.32 sessions used
`opencode/deepseek-v4.1-flash-free` to review public-code snapshots: operator
coverage, dispatch architecture and framework integration. Read/glob/grep were
allowed; edits and shell execution were denied. Their reports were treated as
hypotheses and checked against source and runtime behavior. Several claims were
rejected; see the review corrections below.

Matched probes used the same RTX 2080 Ti and PyTorch 2.14.1, source revision
`5c4886908584029761b579af026dcfb627c84070`: `+cpu` for Vulkan, `+cu130` for CUDA.
Vulkan used resident/deferred mode and physical device 0. CUDA selected the same
physical GPU by UUID. The loaded Vulkan extension hash, environment, direct
registrations and agent session IDs are in the
[summary](../benchmarks/results/universality-b80a1f6/summary.json).

There were **39 operator cases**, generally testing forward and every input
gradient against CPU; integer, FFT and sparse cases were forward-only. CUDA
passed 39; Vulkan passed 27 and raised errors in 12. These cases were deliberately
chosen to inspect gaps, not sampled randomly. This is **not a coverage percentage
of PyTorch**. The separate integration probes also include expected-error cases,
so process exit success or a JSON `status: ok` does not mean the API is correct.
For example, returning the wrong embedding row does not raise an exception.

The tests are small, use one seed and check tolerance-based numerical parity.
They do not establish convergence, all shape/dtype combinations or cross-vendor
portability. Earlier tests covered RTX, V100 and Lavapipe; this new diagnostic
matrix ran on RTX only. AMD/Intel execution remains unverified. No new latency
benchmark was performed in this audit.

## What CUDA covers that this backend does not

The CUDA column below reflects the tested configurations, not a promise that
CUDA supports every dtype, sparse layout or hardware generation.

| Area | CUDA result / capability | Vulkan at the audited revision |
|---|---|---|
| FP32 Conv1D and Conv2D, including 10x10 and 11x11 filters | Forward and gradients passed | Passed without generic CPU fallback. Conv1D is normalized to 2D by ATen. |
| Conv3D and ConvTranspose2D | Forward and gradients passed | Explicit errors from the convolution handler |
| FP16, BF16, FP64 and complex64 matrix multiply and add | All tested cases passed | All eight cases fail dtype checks; int64 tensor add also fails |
| Mixed precision training | Autocast matrix multiply returns FP16 | Autocast errors because the device module lacks `get_amp_supported_dtype`; no native mixed precision compute layer |
| 3D average/max pooling, antialiased bilinear resize, trilinear resize, grid sampling, reflection padding | Forward and gradients passed | Correct outputs/gradients through generic CPU fallback |
| Index-select, gather, scatter-add, embedding-bag | Forward and gradients passed | Generic CPU fallback. Embedding forward has a shader, but its backward falls back and index conversion uses the host. |
| Sort, top-k, cumulative sum, variance | Forward and gradients passed | CPU fallback for the listed operations or parts of backward |
| FFT and dense linear solve | Tested forward / solve gradients passed | CPU fallback; complex FFT output can be copied even though general complex add/matmul fails |
| Sparse COO round trip | Passed | Fails at `Sparsevulkan` tensor construction |
| Dropout and random generation | Work with CUDA's device RNG | Dropout forward/backward and random generation run on CPU; no Vulkan generator implementation |
| LSTM | Forward and parameter-gradient parity passed | Fails when the fused cell routes to a CPU implementation that does not exist |
| Scaled dot-product attention, dropout=0 | Unmasked and causal cases passed | Unmasked composed path has no generic fallback in this test; causal path calls CPU `tril`. No fused training SDPA kernel. |
| Ordinary `torch.optim.AdamW(foreach=False)` | Two steps completed | Two steps completed but update primitives (`lerp`, `addcmul`, `addcdiv`) fall back. The separate Vulkan fused AdamW is a narrower opt-in optimizer. |
| `torch.compile(..., fullgraph=True)` using default Inductor | `sin(x)+x` compiled and matched CPU | Fails with `KeyError: 'vulkan'`. This does not claim that graph capture with every custom/eager backend is impossible. |
| Streams and events | New stream and event record/synchronize work | New stream fails; event object construction works but recording fails |
| Multiple devices and distributed GPU execution | CUDA provides per-device contexts and NCCL collectives | One logical Vulkan device per process; no native distributed collective integration. This last point is source inspection, not a distributed runtime test. |

CUDA's AMP, streams, compilation and distributed capabilities are documented in
[AMP](https://docs.pytorch.org/docs/2.14/amp.html),
[CUDA semantics](https://docs.pytorch.org/docs/2.14/notes/cuda.html),
[torch.compile](https://docs.pytorch.org/docs/2.14/generated/torch.compile.html)
and [distributed](https://docs.pytorch.org/docs/2.14/distributed.html).
Dtype availability does not imply fast native tensor-core execution: the RTX
2080 Ti is not a BF16 tensor-core GPU, even though these CUDA BF16 probes pass.

## Confirmed semantic defects

These deserve priority over additional performance tuning.

| Defect | Concrete observation | Cause / consequence |
|---|---|---|
| Embedding indices lose integer precision | A valid index `16,777,217` returns row `16,777,216`: actual value 456, expected 123. CUDA returns 123. The table is only about 64 MiB with one column. | `mm.cpp` converts indices to FP32 before dispatch. Integers above 2^24 are not all exactly representable. |
| Matrix multiply misses a second-input dtype check | FP32 left input multiplied by FP64 right input returns numbers on Vulkan. CUDA correctly rejects the dtype mismatch. | `mm()` checks the left dtype but uses the right buffer as FP32. This is incorrect interpretation of bytes, not legitimate dtype promotion. |
| Requested allocation layout is ignored | `torch.empty((2,3,5,7), memory_format=channels_last, device='vulkan')` returns strides `(105,35,7,1)`; CUDA returns `(105,1,21,3)`. | `empty_memory_format()` always creates contiguous storage. This does not negate support for already-existing channels-last tensors in convolution. |
| Invalid device index is ignored | Requesting `vulkan:17` silently returns `vulkan:0`; CUDA rejects index 17. | Allocation/guard code hardcodes logical device 0 without consistent argument validation. |
| Device RNG seeding is a no-op | Repeating `torch.vulkan.manual_seed_all(99)` does not repeat device random draws. The CUDA equivalent does. | The Vulkan function is empty. Global `torch.manual_seed` does repeat the tested draws because generation consumes CPU RNG state. A Vulkan draw changes CPU RNG state; a CUDA draw does not. |

Locations: [`csrc/ops/mm.cpp`](../csrc/ops/mm.cpp), `mm` around lines 230–291,
`empty_memory_format` around 492–510, `vulkan_embedding` around 644–680;
[`torch_vulkan/__init__.py`](../torch_vulkan/__init__.py), lines 117–132;
[`csrc/torch_vulkan.cpp`](../csrc/torch_vulkan.cpp), device guard/hooks.
The integration probe is saved with raw results and can reproduce each case.

## Does the architecture still depend on specific networks?

The convolution dispatch is general **within its FP32 supported domain**.
It separates candidate eligibility from ranking; has forward/input-gradient/
weight-gradient/bias phases; and checks geometry, groups, stride, dilation,
workspace and relevant Vulkan device limits. Shared tile configurations divide
work while the actual filter dimensions remain runtime parameters. Changing
10x10 to 11x11 does not require another model-specific implementation.
Both configurations passed forward/backward here.

CUDA uses the same broad separation of operator semantics and implementation
selection, with much broader implementations and tooling. Current upstream
[Convolution.cpp](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/native/Convolution.cpp)
normalizes 1D cases to 2D and selects cuDNN, depthwise, transposed, slow and other
backends. [Conv_v8.cpp](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/native/cudnn/Conv_v8.cpp)
selects/caches engine plans with heuristics or timed search, workspace and
precision/determinism constraints. CUDA embedding backward preserves integer
indices through [typed index dispatch](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/aten/src/ATen/native/cuda/Embedding.cu).
These files were fetched at the exact installed PyTorch revision and inspected.
The agents originally used the older pinned source snapshot from the previous
audit; runtime conclusions and these current-source checks supersede it.

Remaining design work is broader than adding individual shaders:

- The capability/limit checking used by convolution and `bmm` is not consistently
  applied to older `mm`/elementwise/embedding paths. For example, `mm` computes a
  dispatch grid and casts strides to uint32 without the same comprehensive
  device-limit checks. Oversized-device failure was not triggered in this audit;
  this is a source-grounded portability risk, not an observed GPU failure.
- Some registered handlers reject unsupported dtypes, while other handlers use
  explicit CPU copies. The catch-all fallback cannot rescue an operation that
  is already registered and throws inside its handler.
- Hand-written shape thresholds and a small convolution candidate set do not
  provide CUDA-like performance coverage across devices. This is separate from
  accepting arbitrary model graphs. AMD/Intel testing and device-specific
  measurements are still required.
- There is no shared implementation of CUDA-like RNG, autocast, stream/event,
  compiler and distributed contracts. More operator shaders alone will not
  supply those contracts.

## CPU fallback tracing undercounts host execution

`PV_FALLBACK` records only calls to the generic boxed CPU fallback in `mm.cpp`.
An explicitly registered implementation can still copy input to CPU, calculate
there, and copy the result back without producing that log line.

A concrete example from this audit is **int64 sum**: it passes and emits no
`PV_FALLBACK`, but `reduce.cpp:25` explicitly runs `at::sum(x.cpu(), ...)`.
Embedding index conversion and NLL target validation also have host work.
Consequently, the earlier five-network statement means **zero generic fallback
calls in those tested steady-state steps**, not zero CPU work throughout the
backend. Copy instrumentation or a unified fallback-reason counter is needed
for an honest residency coverage metric.

## DeepSeek findings accepted, corrected or rejected

The raw reports are preserved as review inputs. They are **not authoritative
coverage reports**; this document and the measurements supersede their errors.

| Agent claim | Verification |
|---|---|
| FP32-only basic compute, Conv3D/transpose gaps, RNG/AMP/compile gaps | Confirmed by source and runtime |
| Conv1D must fail because our convolution handler expects rank 4 | Rejected: ATen normalizes it to 2D; forward/backward passed |
| Views may not share version counters | Not reproduced: base/view counters both increment, and autograd rejects modification of a saved view, matching CUDA |
| `as_strided` does not check storage bounds | Rejected for the tested path: `at::native::setStrided` rejects an out-of-bounds view on both backends |
| Ignored embedding flags imply incorrect padding/frequency gradients | Rejected for tested dense gradients: autograd supplies those semantics; padding/frequency outputs and gradients match CPU. The separate float-index precision bug is confirmed. |
| All device-to-device copies use the host | Too broad: a whole contiguous supported copy has a GPU path; other copies/casts can use the host |
| Updating cached push constants corrupts earlier deferred commands | The proposed sequential explanation is unsupported: `Sequence::record` immediately calls `OpAlgoDispatch::record`, which records push constants and dispatch dimensions into the command buffer. This is not proof of unrestricted multithread safety. |
| A generic event object cannot exist | Corrected: construction succeeds; actual record/synchronize fails |
| All RNN variants fail | Not established. LSTM was tested; no blanket GRU/RNN conclusion is made. |

## Recommended order

1. **Correctness and truthful capability checks:** preserve integer embedding
   indices, reject mismatched GEMM dtypes, honor/reject allocation options and
   invalid device indices, and add the missing limit checks. Unify logging of
   all CPU fallback/copy paths. Add focused regression cases from this audit.
2. **General primitives used by many graphs:** device RNG/generator state,
   dropout, zeroing, gather/scatter/index-add, embedding backward, triangular
   masks, remaining common reductions and optimizer update primitives. These
   cover reusable operation families, without identifying model names.
3. **Dtypes and missing operator families:** shared FP16/BF16 load/store and
   accumulation rules plus autocast, then transposed/3D convolution and recurrent
   cell support. Define supported dtype/layout/gradient combinations explicitly.
4. **Framework integration:** streams/events, compiler integration, multiple
   logical devices and native distributed collectives. Until implemented, expose
   clear capability failures instead of silently substituting another device.
5. **Performance breadth:** benchmark these common primitives and full workloads
   against CUDA, then tune by operator configuration and device. Keep a generic
   correct implementation available for every supported configuration.

## Reproduction and raw evidence

[`benchmarks/results/universality-b80a1f6`](../benchmarks/results/universality-b80a1f6)
contains the three agent prompts/reports, 39-case probe, integration probe,
event probe, raw JSON/logs, summary and upstream-source hashes. Agent JSONL
transcripts remain in the persistent audit workspace; their hashes and session
IDs are recorded. No credentials or private repository contents were provided.

```sh
# Use the project's CPU torch environment for Vulkan and its CUDA environment
# for CUDA. Select the same physical GPU in both environments.
PYTHONPATH=. TORCH_VULKAN_GPU_RESIDENT=1 TORCH_VULKAN_DEFER_SUBMISSIONS=1 \
  TORCH_VULKAN_TRACE_FALLBACK=1 TORCH_VULKAN_PHYSICAL_DEVICE_INDEX=0 \
  python benchmarks/results/universality-b80a1f6/probe_coverage.py \
  --backend vulkan --out /tmp/vulkan-coverage.json
python benchmarks/results/universality-b80a1f6/probe_coverage.py \
  --backend cuda --out /tmp/cuda-coverage.json
python benchmarks/results/universality-b80a1f6/probe_integration.py \
  --backend vulkan --out /tmp/vulkan-integration.json
```

A source registration count is included for diagnosis (126 direct PrivateUse1
registrations, 1,340 CUDA registrations in these builds). It is not a native
coverage ratio: composites, aliases, overloads and internal CPU branches prevent
that interpretation.
