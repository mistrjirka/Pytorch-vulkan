> Unverified agent review input. See [the verified audit](../../../docs/cuda-universality-audit-b80a1f6.md) for corrections and measured conclusions.

# PyTorch Vulkan backend audit vs CUDA — commit b80a1f6

## Scope & method

Read-only inspection of `source/` (csrc/ops registrations, dtype guards, docs, tests). CUDA contrast is drawn from the pinned refs in `source/cuda_refs/` and the characterization in `docs/vulkan-dispatch-universality-audit.md:19-24`. Distinctions used below: **[GPU]** native shader; **[HOST-IN]** host work hidden inside a registered handler; **[BOXED]** `TORCH_LIBRARY_IMPL(_, PrivateUse1)` → `cpu_fallback` (`mm.cpp:904-951`); **[FAIL]** explicit `TORCH_CHECK`; **[UNVERIFIED]** needs a runtime probe.

Critical global fact: the catch-all fallback is registered in `mm.cpp:950-951`, so *unregistered* ops usually still run correctly via **[BOXED]** CPU fallback. But *registered* ops that then type-check with `native::base()` (default `kFloat`, `native_utils.h:22-26`) throw **[FAIL]** instead of falling back — worse than an unregistered op. `can_pack_view`/`packed` only accept FP32/int64/bool (`native_utils.h:63,83`).

## Top 8 limitations (ranked)

**1. Convolution is 2D-only and rejects transposed convolution — hard failure, not fallback.**
`convolution.cpp:108` `TORCH_CHECK(!transposed,"Vulkan ConvTranspose is not implemented yet")`; `:109-110` requires 4-D NCHW; `:111-113` exactly two spatial dims; `:40` FP32-only. So `ConvTranspose2d`, `Conv1d` and `Conv3d` **fail at the overrideable kernel** rather than reaching `cpu_fallback` (README:225 confirms ConvTranspose2d unsupported; README:224 LSTM fails). CUDA dispatches these through `Convolution.cpp`/`Conv_v8.cpp` engines. Backward is otherwise well-covered for 2-D: dInput/dWeight/dBias with the autograd mask honored at `convolution.cpp:502-513`. **Probe:** `torch.nn.ConvTranspose2d(...).to("vulkan")(x)` and `Conv1d`/`Conv3d`.

**2. FP16/BF16/FP64/complex compute: no kernels; inconsistent handling (some FAIL, some BOXED).**
No `kHalf`/`kBFloat16`/`kDouble`/`kComplex*` appears in any `csrc/ops` compute guard; packing is FP32/int64/bool only (`native_utils.h:63,83`). Registered elementwise ops without a dtype guard (`add.Tensor`/`mul.Tensor` → `mm.cpp:344-363`; `relu` → `mm.cpp:366`) route into `native::base()` and **throw** for half/double/complex. Tests codify this: `test_pytorch_214.py:93-97` expects int32 `+` to raise `"float32 inputs required"`. By contrast `sum`/`cat`/`bmm`/pooling/resize *do* guard and fall back (`reduce.cpp:25`, `cat.cpp:16,20`, `bmm.cpp:25,34`, `average_pool.cpp:32`, `resize_linear_cubic.cpp:22`). FP64 survives only as a device copy (`mm.cpp:408-416`) and then compute throws. CUDA supports FP16/BF16/FP64/complex broadly. **Probe:** `(x.half()+x.half())`, `x.double()*2`, `torch.complex64` add on `vulkan`, comparing error-vs-fallback; also `_dispatch_has_kernel_for_dispatch_key("aten::add.Tensor","PrivateUse1")`.

**3. RNG/dropout absent; `randn(...,device="vulkan")` is host-generated.**
No `native_dropout`, `bernoulli`, `uniform`, `normal`, or Philox registration exists. Device construction depends on `_copy_from_and_resize`, which **requires identical dtype** (`mm.cpp:463`) and copies host bytes. `torch.randn(device="vulkan")` therefore computes on CPU then transfers **[HOST-IN]**. `F.dropout` falls to **[BOXED]** fallback (and its RNG is host-side). README:219 lists dropout/RNG as fallback. **Probe:** trace `TORCH_VULKAN_TRACE_FALLBACK=1` on `torch.randn`/`tensor.uniform_()`/`nn.Dropout`, and confirm no `aten::native_dropout` PrivateUse1 kernel.

**4. Recurrent layers (LSTM/GRU/RNN) unsupported.**
No `aten::lstm`, `gru`, `rnn`, or `*_cell` registration in any `csrc/ops` file. README:224 states LSTM fails because the fused cell implementation is missing. CUDA has cuDNN RNN. **Probe:** one-layer `nn.LSTM` forward and backward on Vulkan; expect failure or full boxed fallback.

**5. Indexing/scatter/gather/masked ops absent → boxed fallback with host staging.**
No registration for `index`, `index_select`, `index_put_`, `gather`, `scatter`/`scatter_add_`, `masked_select`/`masked_fill_`, `take`, `embedding_dense_backward`. Only boolean primitives are native: `predicates_math.cpp:89-94` (`logical_and`, `bitwise_and/not`, `where`, `all`, comparisons, `clamp`, `pow`, `isneginf`). `embedding` forward is native (`mm.cpp:935`) but its gradient is not, so `nn.Embedding` training pulls a **[BOXED]** boundary. These *work* via fallback (correctness), but each is CPU-staged. **Probe:** `x.index_select` / `scatter_add_` / `F.embedding` backward under `TRACE_FALLBACK`.

**6. Reductions, scans and sorts: only FP32 `sum`/`mean` native; the rest fall back.**
`reduce.cpp:30` registers `sum`/`sum.dim_IntList`; `loss_primitives.cpp:313-314` registers `mean`/`mean.dim`; both fall back off-FP32. No `prod`, `amax`/`amin`, `var`/`std`, `norm`, `cumsum`/`cumprod`, `topk`, `sort`, `argmax`/`argmin`, `unique`. Note an `argmax.comp`/`argmax.spv` shader exists but **no registration was found** — likely dead/unused (flag as **[UNVERIFIED]**). CUDA has native reductions. **Probe:** `x.max()`, `x.var()`, `torch.topk`, `x.argmax()`, `x.cumsum(0)` on Vulkan with fallback tracing.

**7. Attention has no fused kernel; SDPA is deliberately disabled.**
`mm.cpp:939-941` documents the custom SDPA shader is *not registered* (inaccurate forward, no autograd); `test_attention.py:25-27` asserts no PrivateUse1 kernel for `scaled_dot_product_attention`. It still works through composite/decomposed operators (matmul, softmax, `where`) and `test_attention.py:22-58` verifies forward+backward via the fallback/composite path. So attention **is supported but not GPU-fused**; flash/efficient backends cannot be selected. Mask primitives exist (`where`, `logical_*`, `all`) but no `_scaled_dot_product*` device kernels. **Probe:** `F.scaled_dot_product_attention(..., is_causal=True)` and `torch.nn.attention.sdpa_kernel([SDPBackend.FLASH_ATTENTION])`.

**8. FFT, linalg and sparse are entirely fallback or unsupported; complex unverified.**
No `fft`/`_fft_r2c`, `linalg.*`, `_sparse*`, or `sparse` registration exists anywhere in `csrc/ops`. Complex needs FP64/FP32 pair storage the backend does not model (`native_utils.h:22-26,63,83`). CPU fallback will cover many dense linalg calls but sparse tensor creation on PrivateUse1 is likely rejected because the allocator is strided-only (`empty_strided` checks `layout==kStrided`, `mm.cpp:518`). **Probe:** `torch.fft.fft(x_vk)`, `torch.linalg.svd(x_vk)`, `x_vk.to_sparse()`.

## Cross-cutting issues (not just absence)

- **Hidden host work inside native handlers.** `_log_softmax` falls back to CPU for small/irregular shapes (`log_softmax.cpp:38-39,61-66`); `_softmax_backward_data` for non-rowwise (`softmax_axis_backward.cpp:23`); GroupNorm (`group_norm.cpp:50,63`), NLL (`nll_loss.cpp:57,75`), `cat` (`cat.cpp:20`), bilinear/bicubic (`resize_linear_cubic.cpp:22,28`), `mean.dim` partial axes (`loss_primitives.cpp:274`), log-sigmoid (`:240,252`). These are **[HOST-IN]**: correct but silently device-boundary-crossing.
- **All device↔device copies are host-mediated.** `_to_copy` does `host_mirror` + `memcpy` (`mm.cpp:400-416`); `copy_` uses `host_mirror.copy_` for the general case (`mm.cpp:456`). Only whole-contiguous same-dtype distinct-storage copies use the pack shader (`:431-451`). This is the dominant residency cost.
- **Missing in-place/out variants.** Registered in-place/out is partial: `leaky_relu_`/`.out`/`_backward.grad_input` (`activation.cpp:111-118`), `add_*`, `sub_*`, `mul_*`, `div_*`, `fill_` (`loss_primitives.cpp:294-314`), `logical_and.out`, `where.*_out`. Absent: `relu_`, `gelu_`, `mm.out`, `addmm.out`, `softmax.out`, `sigmoid_`, `threshold.out`. Unregistered variants go **[BOXED]**.
- **Unsafe-looking assumptions** are guarded inconsistently: ops that omit a dtype check before `base()` produce **[FAIL]** (add/mul/relu), while others **[BOXED]**. That asymmetry, not raw absence, is the main correctness-adjacent hazard for mixed dtypes/int.

## Minimal runtime probes (source-confirmed vs. needs probe)

1. `_dispatch_has_kernel_for_dispatch_key` for `convolution_overrideable`, `scaled_dot_product_attention`, `native_dropout`, `index`, `fft.fft`, `linalg.svd` — source-confirmed missing for the last five.
2. `ConvTranspose2d`, `Conv1d`, `Conv3d` forward — source-confirmed **[FAIL]** (`convolution.cpp:108-113`).
3. `half`/`bfloat16`/`float64`/`complex64` `+`/`*`/`relu` — source-confirmed **[FAIL]** path; confirm driven by `base()`.
4. `int64`/`int32` `+` vs `sum` — source: `+` fails, `sum` falls back; probe to confirm.
5. `nn.LSTM`, `nn.Embedding` backward, `scatter_add_`, `topk`, `var` — **[BOXED]**, confirm via `TRACE_FALLBACK`.
6. `torch.randn(device="vulkan")` and `F.dropout` — confirm host generation vs fallback.
7. `x.to_sparse()` and `x.to(torch.complex64)` then compute — sparse/complex likely unsupported.
8. Autograd through the boxed path: verify `AutogradPrivateUse1` hooks (`torch_vulkan.cpp:84-111`) keep grads on device for fallback ops (README/tests claim it; not source-verified here).
