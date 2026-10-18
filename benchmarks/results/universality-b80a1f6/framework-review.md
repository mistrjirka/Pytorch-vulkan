> Unverified agent review input. See [the verified audit](../../../docs/cuda-universality-audit-b80a1f6.md) for corrections and measured conclusions.

I have enough to report. Findings below cite `source/`-relative paths; `docs/` and `README.md` are treated as claims, with the commit `b80a1f6` source taking precedence.

# Vulkan backend audit vs PyTorch CUDA — integration beyond shader coverage

Scope: integration semantics, not kernel math. Categories: **[GPU]** native, **[HIDDEN-CPU]** CPU work inside registered native handlers (invisible to `PV_FALLBACK`), **[BOXED]** boxed `cpu_fallback`, **[FAIL]** explicit failure, **[UNVERIFIED]**.

## Ranked gaps

### P0-1. RNG / generator semantics — [BOXED] + [FAIL]
- `torch_vulkan/__init__.py:129-132`: `manual_seed_all` is a deliberate no-op; `_is_in_bad_fork` hard-returns `False` (125-126). No `_C` generator registration, no `Generator`/Philox, no `get_rng_state`/`set_rng_state`.
- No `manual_seed` on the device module (only `manual_seed_all`), so `torch.vulkan.manual_seed(...)` fails while `torch.manual_seed` works via CPU.
- Random factories (`randn`, `rand`, `dropout`) have no native impl, so they hit the boxed fallback (`mm.cpp:950-952`) and consume the **CPU** generator. Deterministic but never GPU-resident; `torch.randn(..., device="vulkan")` goes through `_copy_from_and_resize` (`mm.cpp:462-480`). README:129/219 concedes "complete dropout/RNG" missing.
- CUDA has per-device generators, `torch.cuda.manual_seed`, fork/torch.initial_seed semantics. Probe: `torch.randn(4,device="vulkan")` vs `torch.randn(4)` after seeding; check `torch.vulkan.manual_seed` existence.

### P0-2. Autocast / AMP / dtype — [FAIL]/[BOXED], gap is default-level
- No `AutocastPrivateUse1` and no `TORCH_LIBRARY_IMPL(aten, AutocastPrivateUse1, …)` anywhere in `csrc/` (grep across source found none). `torch.amp.autocast("vulkan")` / `torch.autocast("vulkan")` is undefined; kernels are literally FP32-only (`TOCRH_CHECK` float32 in `mm.cpp:96,146,233`; README:214).
- `torch.amp.GradScaler("vulkan")` unverified (device module lacks the required hooks). This is not opt-in — plain CUDA-style AMP code cannot run.

### P0-3. Streams / events / device guard — [FAIL], partly misleading
- `csrc/torch_vulkan.cpp:30-65`: guard hardcodes device 0, `getStream`/`getDefaultStream`/`exchangeStream` all return `Stream::DEFAULT`; `deviceCount()==1`. `exchangeStream` drops the requested stream.
- Python device module (`__init__.py:86-135`) defines only `device/device_count/current_device/synchronize/manual_seed_all/_is_in_bad_fork`; it has **no** `Stream`, `Event`, `current_stream`, `stream`, `set_stream`, `set_device`. `torch.accelerator.current_stream("vulkan")`, `torch.Stream(device="vulkan")`, CUDA-graph-style capture are therefore unsupported (AttributeError, not a clean `TORCH_CHECK`).
- `non_blocking` is silently ignored: `vulkan_copy_` (mm.cpp:422-423) and `_to_copy` (388-395) never read it and the CPU-mirror path is synchronous (`host_mirror` calls `synchronize()`, mm.cpp:33). `record_stream` absent.

### P0-4. torch.compile / symbolic shapes — [UNVERIFIED], likely [FAIL]
- No Dynamo/Inductor/FakeTensor/meta integration in source (grep found only a doc mention). PrivateUse1 is not an Inductor backend; expect graph breaks or compile failure.
- Symbolic shapes would break at eager runtime even if traced: `native::u32` rejects anything outside `[0, UINT32_MAX]` (`csrc/ops/native_utils.h:17-19`) and callers use `SymInt::expect_int()`-style conversions (`at::infer_size_dv`, `mm.cpp:821`). `empty_strided` (514-523) and view/as_strided (804-830) do use SymInt, so metadata ops may survive. Probe: `torch.compile(fn)(x_vulkan)` and a dynamic-shape variant.

## P1

### P1-5. Multi-GPU / device indexing — [FAIL]/misleading
- Physical GPU is chosen by `TORCH_VULKAN_PHYSICAL_DEVICE_INDEX` before import (`vulkan_context.cpp:27`; README:48-63) but logical device is fixed at 0: `device_count()→1` (`__init__.py:117-118`), hook `deviceCount()=1` and `setCurrentDevice` `TORCH_CHECK(index==0)` (`torch_vulkan.cpp:94-98`). No `set_device` on the Python module.
- `empty.memory_format` ignores its `device` argument entirely (`mm.cpp:492-510`), so `torch.empty(…, device="vulkan:1")` silently yields device 0. CUDA raises for invalid index.

### P1-6. Distributed — [FAIL]
- No `c10d`/ProcessGroup backend registration, no collectives, no `init_process_group(backend="vulkan")`. Fully absent; DataParallel/DDP cannot work.

### P1-7. Optimizer generality — opt-in only
- `torch_vulkan/optim.py:1-56` is an opt-in fused AdamW; `torch.optim.AdamW` is intentionally not patched (line 3). Missing AMSGrad, `maximize`, `foreach`/`fused`/`differentiable`, `capturable`, per-parameter tensor LRs, master-weight AMP; sparse raises (`optim.py:32-33`); non-Vulkan params raise (34-35). Host-side `state['step'] += 1` and Python-int step list (41-46, 50-52). Generic `torch.optim.*` relies on boxed fallback for the update ops; README:220 notes `zero_` isn't native. CUDA's optimizers have all of the above shared.

### P1-8. Alias / version counter + async correctness — [UNVERIFIED]
- `alias_with_strides` (`mm.cpp:791-802`) builds a `TensorImpl::VIEW` from `tensor.storage()`/`key_set()`/`dtype()` without going through `at::alias` or sharing a `VariableVersion`. Sharing storage without a shared version counter can defeat autograd's in-place-modification detection across base↔view; `vulkan_as_strided` (804-810) also does no bounds/overlap validation. Probe: create a Vulkan view, in-place mutate base, check the version-counter error that CUDA raises.
- Default execution is synchronous, not asynchronous: every native path `synchronize()`s (`mm.cpp:33,651,680,846,896`; `torch_vulkan.cpp:168`). Deferred (`TORCH_VULKAN_DEFER_SUBMISSIONS`, `vulkan_context.cpp:301`) and GPU-resident (`TORCH_VULKAN_GPU_RESIDENT`, `vulkan_context.cpp:45`) are opt-in/experimental (README:154-169) — distinguish these from defaults.

## P2 — hidden CPU work and misleading APIs

### P2-9. CPU work invisible to `PV_FALLBACK` — [HIDDEN-CPU]
`cpu_fallback` only prints when the **boxed** `_` fallback is taken (`mm.cpp:904-917`). Registered handlers that internally do `x.cpu()` / `.to(device)` emit nothing:
- `ops/predicates_math.cpp:12,20,28,33,38,55,58,63,68,76,80,85`
- `ops/loss_primitives.cpp:58,61,64,68,90-98,112-114,142,212,215,240,252,274`
- `ops/nll_loss.cpp:57,75`; `ops/group_norm.cpp:50,63`; `ops/softmax_axis_backward.cpp:23`
- `ops/resize_linear_cubic.cpp:22,28`; `ops/average_pool.cpp:32,36,40,44`; `ops/cat.cpp:20`; `ops/bmm.cpp:34`; `ops/reduce.cpp:25`; `ops/scalar_add.cpp:48,53,127`; `ops/layer_norm_backward.cpp:17`
- `ops/mm.cpp:534` (non-packable softmax) and `contiguous_for_kernel`'s host round-trip `host_mirror(src).contiguous().to(device)` (`mm.cpp:43-59`), triggered for un-packed dtypes/ranks.
Probe: a Python counter wrapping `torch.Tensor.cpu`/H2D will reveal these; `PV_FALLBACK` alone undercounts.

### P2-10. Misleading hooks vs public API — [FAIL]
- `VulkanHooks::isAvailable()` returns `true` unconditionally and `isBuilt()` true (`torch_vulkan.cpp:88-90`), whereas public `is_available()` actually probes the driver (138-145). `getDeviceFromPtr(void*)` returns `vulkan:0` for **any** pointer (99-101) — misclassification risk in mixed CPU/Vulkan autograd. CUDA hooks reflect real availability.

### P2-11. `empty.memory_format` ignores options — [FAIL]
`mm.cpp:492-510` ignores `memory_format` (never channels-last/preserve), `pin_memory`, `layout`, and the device index. `torch.empty(..., memory_format=torch.channels_last, device="vulkan")` silently returns contiguous. CUDA honors these.

### P2-12. `embedding` flags silently ignored — [FAIL]
`mm.cpp:644-649` accepts `padding_idx`, `scale_grad_by_freq`, `sparse` but never uses them; indices are cast to float (654) — precision loss >2²⁴, no int64 path. Misleading relative to CUDA.

### P2-13. Minor
- `mul_scalar` (`mm.cpp:773-780`) casts `numel()` to `uint32` with no bound check, unlike `add_tensor` (`mm.cpp:348`).
- `torch.Tensor.vulkan` shim (`__init__.py:35-41`) rejects positional args; the legacy `is_vulkan` conflict is acknowledged at 28-34. `torch.Tensor.is_vulkan` still refers to legacy Vulkan (README:232-234).
- `conv_dispatch_stats`/`queue_stats`/`gpu_timing_stats` are diagnostics; timestamps are opt-in (`TORCH_VULKAN_PROFILE_TIMESTAMPS=1`, `__init__.py:71-73`).

## Practical low-cost probes
1. `torch.vulkan.manual_seed`, `torch.randn(device="vulkan")`, `torch.dropout` under a seeded generator — RNG.
2. `with torch.autocast("vulkan"): torch.mm(x,y)` and `GradScaler("vulkan")` — AMP.
3. `torch.accelerator.current_stream("vulkan")`, `torch.Stream("vulkan")`, `cuda.Event` equivalents, `torch.compile(fn)` + dynamic shapes — streams/compile.
4. `torch.empty(4, device="vulkan:1")` and `memory_format=torch.channels_last` — device/allocator honesty.
5. Wrap `Tensor.cpu` and compare counts with `TORCH_VULKAN_TRACE_FALLBACK=1` to expose P2-9 hidden host work.
6. Vulkan view → in-place mutate → expect version-counter error — alias/version.

**Bottom line:** native FP32 compute coverage is broad and growing, but for "general PyTorch like CUDA" the integration layer is the bottleneck: RNG, AMP/autocast, streams/events, `torch.compile`, logical multi-GPU indexing, distributed, and optimizer generality are absent or fixed-single-device; several device-module APIs are misleading (unconditional availability, ignored `device`/`memory_format` flags, ignored `embedding` flags), and substantial CPU work is performed inside native handlers where `PV_FALLBACK` cannot see it.
