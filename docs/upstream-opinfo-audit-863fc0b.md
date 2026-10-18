# Upstream OpInfo correctness audit at 863fc0b

**The backend does not pass broad upstream FP32 correctness testing.** The previous
430-test feature-completion milestone covered selected contracts; it was not an
upstream conformance result. This audit finds missing basic dispatches, unsupported
edge cases, runtime contracts, and a confirmed numerical bug in an implemented
operator. No backend code was changed during this evaluation.

## Scope and reproducibility

The runner loads the unmodified upstream
[`test/test_ops.py`](https://github.com/pytorch/pytorch/blob/5c4886908584029761b579af026dcfb627c84070/test/test_ops.py)
matching installed PyTorch 2.14.1's source commit
`5c4886908584029761b579af026dcfb627c84070`. Its SHA-256 is
`17ffae61a1654f23d7c2c8f73879bee99c56a80482d061c96975c0fc4720020c`.
Installed test-helper hashes and dependency versions are recorded in the results.
The upstream registry has **703 OpInfo entries**.

We use PyTorch's `PrivateUse1TestBase` and its documented-in-source
`bypass_device_restrictions=True` hook to open built-in-device gates. We do not
change test bodies, OpInfo samples, tolerances or expected failures, and do not
allowlist operators based on what this backend implements.

The selected scope contains **8,696 generated cases** from `TestCommon`,
`TestCompositeCompliance` and `TestForwardADWithScalars`: CPU comparison,
noncontiguous forward/backward consistency, eager variants, output buffers and
warnings, error contracts, unusual bool representations, reductions, composite
forward/backward/forward AD, view replay, and copy-on-write behavior. Explicit
dtypes are FP32, int64 and bool (the selector also accepts int32, but upstream
produced no selected int32 case). Tests without a dtype parameter are retained.
An FP32-tagged test may explicitly request a different output/temporary dtype;
those failures are kept visible rather than silently removed.

**24,078 other generated Vulkan cases were outside this first-pass scope.** They
include other test families and precision variants, FakeTensor and math-bit tests.
This is not all of PyTorch's tests, every OpInfo's every possible contract, or a
complete numerical finite-difference gradcheck suite. Selected and excluded case
IDs and reasons are preserved in `rtx/inventory.json.gz`.

The full sweep runs on **RTX 2080 Ti**, with GPU residency, deferred submissions,
and `TORCH_VULKAN_FORBID_CPU_FALLBACK=1`. Each case records execution counters.
Host-read counts include intentional CPU comparisons and sample generation, so
this suite is not a GPU-residency certification. Independent reproductions also
run on V100 and Lavapipe; the full 8,696-case sweep has not been run on those two.

## Results

| Outcome | Initial sweep | After isolated timeout recheck |
|---|---:|---:|
| Passed | 1,432 | **1,433** |
| Errors / failed checks | **6,721** | **6,721** |
| Upstream skips | 436 | 436 |
| Upstream expected failures | 102 | 102 |
| Unexpected successes under upstream expected-failure marks | 4 | 4 |
| Timeout | 1 | 0 |
| Total | 8,696 | 8,696 |

The max-pool backward eager-variant case exceeded the initial 30-second watchdog.
In isolation with a 180-second budget, it **passed in 35.7 seconds**. Both records
are retained. A worker was restarted after that timeout; all selected cases have
an outcome and no case is left unexecuted. A runner exit status of zero means the
audit completed, not that the audited tests passed.

These are **test-case counts, not percentages of operators implemented**. One
case can contain many samples and stops on its first failure. Error-expectation
and output-warning tests can pass without establishing full numeric coverage.
Upstream wrappers often turn assertion failures into exceptions, so the raw
`error` label includes assertion failures as well as missing implementations.

Of the 6,721 errors, **5,999 include a forbidden CPU fallback** and **469 reject
an RNG output dtype**. The traces contain 279 distinct first-failing fallback
dispatches. These are not 279 proven independent FP32 operator bugs: failures can
occur in sample construction, comparison helpers, the target operator, or
composite infrastructure; some samples use excluded dtypes or CUDA-only features.

| Frequently encountered first missing dispatch | Cases |
|---|---:|
| `eq.Tensor_out` | 1,772 |
| `_linalg_svd.U` | 357 |
| `randperm.generator_out` | 188 |
| `_index_put_impl_` | 176 |
| `arange.start_out` | 96 |
| `norm.out` | 90 |
| `set_.source_Tensor` | 82 |
| `linalg_vector_norm.out` | 81 |

For example, the assertion helper calls `isclose`, which needs tensor equality;
missing comparison dispatches block many otherwise unrelated tests. Masked-op
sample construction needs boolean RNG. Composite compliance needs storage/view
operations such as `set_`. **None of those blocked cases is counted as passed.**
The summary and per-case index retain the full family/dtype/operator breakdown.

## Independently checked findings

All observations below reproduced on **RTX, V100 and Lavapipe**, with CPU controls.

| Finding | Independent result |
|---|---|
| `addmm` with matrix bias, `alpha=0.6`, `beta=0.2` | **Wrong numbers**, with zero fallback and zero host reads before explicit comparison |
| Same `addmm` with `beta=1` | Passes; explains why ordinary linear-layer cases missed it |
| `t()` on a one-dimensional tensor | Rejects valid input with `t: only 2D tensors` |
| Boolean `randint` and integer-valued `random_` on FP32 output | Rejected by the RNG dtype guard |
| Advanced tensor indexing | Missing `index.Tensor_out`; fallback forbidden |
| Tensor equality | Missing `eq.Tensor_out`; fallback forbidden |
| Direct offset partial-view `as_strided` | Passes |
| Empty inner-dimension matrix multiply and direct empty uniform fill | Pass these small reproductions; broader failing cases need separate reduction |

The generic `addmm` path currently ends with `add_tensor(bias, product, beta)`,
which scales the product by beta instead of scaling the bias. With alpha already
applied, it computes `bias + beta * alpha * (A @ B)` rather than
`beta * bias + alpha * (A @ B)`. The independent reproduction has an absolute
error of approximately **1.1853**. Fixing this dispatch path is a concrete priority.

Two apparent numerical issues require care:

- Upstream's partial-view CPU comparison converts the sliced GPU view to a new
  CPU allocation, changing its storage origin before calling `as_strided` with an
  explicit offset. Direct matching CPU/GPU views agree; the copied-reference
  construction does not. Upstream already marks these cases as expected failures.
- The raw `alpha_dropout` CPU comparison reports different stochastic outputs.
  This audit does not establish that it is an arithmetic bug: Vulkan does not
  promise the same random sequence as CPU. It needs distribution/semantic testing,
  not an assumption that independent RNG streams produce identical samples.

The 66 copy-on-write materialization checks and the remaining error-contract,
composite, dtype and sample-generation failures remain open for investigation.
This evaluation does not weaken their assertions or relabel them as successes.

## Reproduction and next work

```sh
.venv/bin/python -m pip install expecttest hypothesis scipy
curl -fL https://raw.githubusercontent.com/pytorch/pytorch/5c4886908584029761b579af026dcfb627c84070/test/test_ops.py -o /tmp/test_ops.py
.venv/bin/python benchmarks/run_upstream_opinfo.py \
  --source /tmp/test_ops.py --out /tmp/opinfo-audit --timeout 180
python3 benchmarks/summarize_upstream_opinfo.py benchmarks/results/upstream-opinfo-863fc0b
```

`--match` accepts a substring of a generated case ID for focused reproductions.
Use a new output directory for an independent rerun; an existing directory resumes
completed cases. The test source hash and PyTorch source version are checked.

Next priorities are the confirmed `addmm`/transpose semantics, shared comparison
and factory/RNG dispatches, then indexed writes and storage/composite contracts.
After those blockers are removed, rerun this same suite to expose failures that
are currently hidden behind them. Broader linalg and other API families remain
separate implementation work. The earlier feature-suite pass counts must not be
used as evidence of general FP32 completeness.

Raw outcomes with traceback excerpts (up to 14,000 characters per error), the
selected/excluded inventory, concise case index, dependency provenance, timeout
recheck and three-device reproductions are under
[`benchmarks/results/upstream-opinfo-863fc0b`](../benchmarks/results/upstream-opinfo-863fc0b/).
