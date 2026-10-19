"""Independent semantic checks for failures exposed by the upstream audit."""
import pytest
import torch
import torch_vulkan as vk


def resident():
    stats = vk.execution_stats()
    assert stats['cpu_fallback_calls'] == 0, stats
    assert stats['host_read_max_bytes'] == 0, stats


@pytest.mark.parametrize('bias_shape', [(), (3,), (1, 3), (2, 1), (2, 3)])
@pytest.mark.parametrize('alpha,beta', [(0.6, 0.2), (1., 1.), (-2., -3.), (1., 0.)])
@pytest.mark.parametrize('k', [0, 4])
def test_addmm_coefficients(bias_shape, alpha, beta, k):
    torch.manual_seed(82)
    a, b, bias = torch.randn(2, k), torch.randn(k, 3), torch.randn(bias_shape)
    if beta == 0: bias.fill_(float('nan'))
    av, bv, cv = [t.to('vulkan') for t in (a, b, bias)]
    vk.reset_execution_stats()
    actual = torch.addmm(cv, av, bv, alpha=alpha, beta=beta)
    resident()
    torch.testing.assert_close(actual.cpu(), torch.addmm(bias, a, b, alpha=alpha, beta=beta))


@pytest.mark.parametrize('shape', [(0, 3, 7), (3, 7, 0), (0, 0, 0)])
def test_addmm_empty_dimensions(shape):
    m, k, n = shape
    a, b, c = torch.empty(m, k), torch.empty(k, n), torch.empty(n)
    actual = torch.addmm(c.to('vulkan'), a.to('vulkan'), b.to('vulkan'), beta=.2)
    torch.testing.assert_close(actual.cpu(), torch.addmm(c, a, b, beta=.2))


@pytest.mark.parametrize('shape', [(), (7,), (0,), (2, 3)])
def test_transpose_alias(shape):
    x = torch.empty(shape, device='vulkan')
    y = x.t()
    assert y.shape == torch.empty(shape).t().shape
    assert y.untyped_storage().data_ptr() == x.untyped_storage().data_ptr()
    y.fill_(19)
    torch.testing.assert_close(x.cpu(), torch.full(shape, 19.))


def test_transpose_rejects_rank_three():
    with pytest.raises(RuntimeError, match='<= 2 dimensions'):
        torch.empty(2, 3, 4, device='vulkan').t()


@pytest.mark.parametrize('op', ['eq', 'ne', 'lt', 'le', 'gt', 'ge'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.int32, torch.int64, torch.bool])
def test_comparison_broadcast_scalar_and_strided_out(op, dtype):
    vals = [0, 1, -1, 2, 7]
    if dtype == torch.int64: vals = [-2**63, 2**60, 2**60+1, 0, 2**63-1]
    if dtype == torch.float32: vals = [float('nan'), float('inf'), -float('inf'), -0., .25]
    a = torch.tensor(vals, dtype=dtype).reshape(5, 1)
    b = a.t()
    av, bv = a.to('vulkan'), b.to('vulkan')
    backing = torch.ones(7, 11, dtype=torch.bool).to('vulkan')
    out = backing[1:6, 1::2]
    vk.reset_execution_stats()
    actual = getattr(torch, op)(av, bv, out=out)
    scalar = getattr(torch, op)(av, 1)
    resident()
    expected = torch.ones(7, 11, dtype=torch.bool)
    expected[1:6, 1::2] = getattr(torch, op)(a, b)
    torch.testing.assert_close(backing.cpu(), expected)
    torch.testing.assert_close(scalar.cpu(), getattr(torch, op)(a, 1))
    assert actual.data_ptr() == out.data_ptr()


@pytest.mark.parametrize('src_dtype', [torch.float32, torch.int32, torch.int64, torch.bool])
@pytest.mark.parametrize('dst_dtype', [torch.float32, torch.int32, torch.int64, torch.bool])
def test_device_cast_and_partial_copy(src_dtype, dst_dtype):
    a = torch.tensor([-3, -1, 0, 1, 7, 99], dtype=src_dtype).reshape(2, 3)
    av = a.to('vulkan').t()
    backing = torch.ones(7, 5, dtype=dst_dtype).to('vulkan')
    out = backing[1:6:2, 1::2]
    vk.reset_execution_stats()
    out.copy_(av)
    cast = av.to(dtype=dst_dtype)
    resident()
    expected = torch.ones(7, 5, dtype=dst_dtype)
    expected[1:6:2, 1::2] = a.t()
    torch.testing.assert_close(backing.cpu(), expected)
    torch.testing.assert_close(cast.cpu(), a.t().to(dst_dtype))


@pytest.mark.parametrize('dtype', [torch.float32, torch.bool])
def test_random_integer_other_dtypes(dtype):
    g = torch.Generator(device='vulkan').manual_seed(901)
    state = g.get_state()
    backing = torch.ones(17, 19, dtype=dtype).to('vulkan')
    out = backing[1::2, 1::2].t()
    vk.reset_execution_stats()
    out.random_(0, 2, generator=g)
    resident()
    first = out.cpu()
    g.set_state(state)
    same = torch.empty(out.shape, dtype=dtype, device='vulkan').random_(0, 2, generator=g)
    torch.testing.assert_close(same.cpu(), first)
    expected = torch.ones(17, 19, dtype=dtype)
    expected[1::2, 1::2] = first.t()
    torch.testing.assert_close(backing.cpu(), expected)
    assert (first == 0).any() and (first == 1).any()


@pytest.mark.parametrize('op', ['add', 'mul', 'relu', 'copy', 'fill', 'random'])
def test_copy_on_write_preserves_gpu_dirty_source(op):
    original = torch.arange(24.).reshape(4, 6).to('vulkan') + 7
    clone = torch._lazy_clone(original)
    vk.reset_execution_stats()
    if op == 'add': result = clone + 2
    elif op == 'mul': result = clone * clone
    elif op == 'relu': result = clone.relu()
    elif op == 'copy': result = clone.copy_(torch.ones_like(clone))
    elif op == 'fill': result = clone.fill_(3)
    else: result = clone.random_(0, 2)
    resident()
    assert torch._C._is_cow_tensor(original)
    torch.testing.assert_close(original.cpu(), torch.arange(24.).reshape(4, 6) + 7)
    if op in ('add', 'mul', 'relu'):
        assert torch._C._is_cow_tensor(clone)
        ref = torch.arange(24.).reshape(4, 6) + 7
        expected = ref + 2 if op == 'add' else ref * ref if op == 'mul' else ref.relu()
        torch.testing.assert_close(result.cpu(), expected)
    else:
        assert not torch._C._is_cow_tensor(clone)


@pytest.mark.parametrize('dtype', [torch.float32, torch.int32, torch.int64])
@pytest.mark.parametrize('bounds', [(0, 0, 1), (1, 17, 3), (9, -11, -3)])
def test_arange_native(dtype, bounds):
    vk.reset_execution_stats()
    actual = torch.arange(*bounds, dtype=dtype, device='vulkan')
    resident()
    torch.testing.assert_close(actual.cpu(), torch.arange(*bounds, dtype=dtype))


def test_arange_exact_int64():
    start = 2**60 + 7
    vk.reset_execution_stats()
    result = torch.arange(start, start + 31, 3, device='vulkan')
    resident()
    torch.testing.assert_close(result.cpu(), torch.arange(start, start + 31, 3))


@pytest.mark.parametrize('steps', [0, 1, 2, 19, 513])
def test_linspace_native(steps):
    vk.reset_execution_stats()
    result = torch.linspace(-3.7, 11.2, steps, device='vulkan')
    resident()
    torch.testing.assert_close(result.cpu(), torch.linspace(-3.7, 11.2, steps))


@pytest.mark.parametrize('dtype', [torch.float32, torch.int32, torch.int64])
def test_randperm_native(dtype):
    generator = torch.Generator(device='vulkan').manual_seed(91)
    state = generator.get_state()
    vk.reset_execution_stats()
    a = torch.randperm(513, dtype=dtype, device='vulkan', generator=generator)
    resident()
    generator.set_state(state)
    b = torch.randperm(513, dtype=dtype, device='vulkan', generator=generator)
    torch.testing.assert_close(a.cpu(), b.cpu())
    torch.testing.assert_close(a.cpu().sort().values, torch.arange(513, dtype=dtype))


@pytest.mark.parametrize('name', ['isnan', 'isinf', 'isposinf', 'isneginf', 'logical_not'])
def test_unary_predicates_native(name):
    x = torch.tensor([float('nan'), float('inf'), -float('inf'), 0., -2.])
    v = x.to('vulkan')
    vk.reset_execution_stats()
    result = getattr(torch, name)(v)
    resident()
    torch.testing.assert_close(result.cpu(), getattr(torch, name)(x))


def test_metadata_unfold_and_set():
    x = torch.arange(19.).to('vulkan')
    out = torch.empty(0, device='vulkan')
    vk.reset_execution_stats()
    out.set_(x[2::3])
    windows = x.unfold(0, 5, 2)
    resident()
    assert out.untyped_storage().data_ptr() == x.untyped_storage().data_ptr()
    assert windows.untyped_storage().data_ptr() == x.untyped_storage().data_ptr()
    torch.testing.assert_close(out.cpu(), torch.arange(19.)[2::3])
    torch.testing.assert_close(windows.cpu(), torch.arange(19.).unfold(0, 5, 2))


@pytest.mark.parametrize('dims', [None, (0,), (1,), (0, 2)])
@pytest.mark.parametrize('order', [0, 1, 2, 3.5, -1, -2, float('inf'), -float('inf')])
def test_general_norm(dims, order):
    torch.manual_seed(99)
    x = torch.rand(3, 5, 7) + .1
    v = x.to('vulkan').transpose(0, 2)
    x = x.transpose(0, 2)
    vk.reset_execution_stats()
    result = torch.linalg.vector_norm(v, order, dims, keepdim=True)
    resident()
    torch.testing.assert_close(result.cpu(), torch.linalg.vector_norm(x, order, dims, keepdim=True))


@pytest.mark.parametrize('op', ['amax', 'amin'])
def test_extrema_nan_and_strides(op):
    x = torch.arange(105.).reshape(3, 5, 7)
    x[1, 1, 1] = float('nan')
    v = x.to('vulkan').transpose(0, 2)
    vk.reset_execution_stats()
    result = getattr(torch, op)(v, (0, 2))
    resident()
    torch.testing.assert_close(result.cpu(), getattr(torch, op)(x.transpose(0, 2), (0, 2)), equal_nan=True)


@pytest.mark.parametrize('name', ['sin', 'cos', 'exp', 'sqrt', 'rsqrt', 'neg', 'tanh', 'sigmoid', 'abs'])
def test_existing_native_out(name):
    x = torch.linspace(.1, .9, 15).reshape(3, 5)
    v = x.to('vulkan')
    backing = torch.full((3, 11), 17., device='vulkan')
    out = backing[:, 1::2]
    vk.reset_execution_stats()
    getattr(torch, name)(v, out=out)
    resident()
    expected = torch.full((3, 11), 17.)
    expected[:, 1::2] = getattr(torch, name)(x)
    torch.testing.assert_close(backing.cpu(), expected)


def test_integer_float_cast_rounding():
    # Adjacent integers around ties must round exactly like an IEEE conversion.
    values = [0, -2**63, 2**63-1]
    for bit in [24, 31, 32, 40, 60]:
        for delta in [-1, 0, 1]:
            v = 2**bit + 2**(bit-24) + delta
            values.extend([v, -v])
    x = torch.tensor(values, dtype=torch.int64)
    v = x.to('vulkan')
    vk.reset_execution_stats()
    result = v.float()
    resident()
    torch.testing.assert_close(result.cpu(), x.float(), atol=0, rtol=0)


@pytest.mark.parametrize('op', ['bitwise_and', 'bitwise_or', 'bitwise_xor'])
@pytest.mark.parametrize('dtype', [torch.bool, torch.int32, torch.int64])
def test_bitwise_broadcast_and_out(op, dtype):
    x = torch.tensor([-1, 0, 2, 7, 91], dtype=dtype).reshape(5, 1)
    y = torch.tensor([1, 3, 11], dtype=dtype)
    a, b = x.to('vulkan'), y.to('vulkan')
    out = torch.empty(5, 3, device='vulkan', dtype=dtype)
    vk.reset_execution_stats()
    result = getattr(torch, op)(a, b, out=out)
    neg = torch.bitwise_not(a)
    resident()
    torch.testing.assert_close(result.cpu(), getattr(torch, op)(x, y))
    torch.testing.assert_close(neg.cpu(), torch.bitwise_not(x))


@pytest.mark.parametrize('shape', [(1, 1), (3, 5), (5, 3), (9, 9), (2, 3, 7), (2, 7, 3), (0, 3), (3, 0), (0, 3, 5)])
@pytest.mark.parametrize('full', [False, True])
@pytest.mark.parametrize('kind', ['random', 'zero', 'rank_one'])
def test_svd_native(shape, full, kind):
    torch.manual_seed(537)
    x = torch.randn(shape)
    if kind == 'zero': x.zero_()
    if kind == 'rank_one': x = x[..., :, :1].expand(shape).clone() if shape[-1] else x
    a = x.to('vulkan')
    vk.reset_execution_stats()
    u, s, vh = torch.linalg.svd(a, full_matrices=full)
    stats = vk.execution_stats()
    assert stats['cpu_fallback_calls'] == 0 and stats['host_read_max_bytes'] <= 4, stats
    u, s, vh = u.cpu(), s.cpu(), vh.cpu()
    k = min(shape[-2:])
    torch.testing.assert_close(s, torch.linalg.svdvals(x), atol=3e-5, rtol=2e-5)
    torch.testing.assert_close((u[..., :k] * s.unsqueeze(-2)) @ vh[..., :k, :], x, atol=4e-5, rtol=2e-5)
    torch.testing.assert_close(u.mT @ u, torch.eye(u.shape[-1]).expand(*u.shape[:-2], u.shape[-1], u.shape[-1]), atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(vh @ vh.mT, torch.eye(vh.shape[-2]).expand(*vh.shape[:-2], vh.shape[-2], vh.shape[-2]), atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize('dtype', [torch.float32, torch.bool, torch.int32, torch.int64])
@pytest.mark.parametrize('pattern', ['adjacent', 'separated', 'mask', 'empty'])
def test_advanced_indexing_and_assignment(dtype, pattern):
    x = (torch.arange(60).reshape(3, 4, 5) % 7).to(dtype)
    if pattern == 'adjacent': index = (torch.tensor([0, -1, 1]), torch.tensor([3, 1, 0]))
    elif pattern == 'separated': index = (torch.tensor([2, 0, 1]), slice(None), torch.tensor([1, 3, 0]))
    elif pattern == 'mask': index = (torch.tensor([True, False, True]), slice(None), slice(None))
    else: index = (torch.empty(0, dtype=torch.long),)
    v = x.to('vulkan')
    iv = tuple(i.to('vulkan') if isinstance(i, torch.Tensor) else i for i in index)
    vk.reset_execution_stats()
    result = v[iv]
    value = torch.ones_like(result)
    v[iv] = value
    stats = vk.execution_stats()
    assert stats['cpu_fallback_calls'] == 0 and stats['host_read_max_bytes'] <= 4, stats
    torch.testing.assert_close(result.cpu(), x[index])
    expected = x.clone(); expected[index] = 1
    torch.testing.assert_close(v.cpu(), expected)


@pytest.mark.parametrize('dtype', [torch.float32, torch.int32, torch.int64, torch.bool])
def test_index_put_duplicate_accumulation(dtype):
    x = torch.ones(4, dtype=dtype)
    indices = torch.tensor([1, 1, -1, 1])
    values = torch.tensor([2, 3, 4, 5], dtype=dtype)
    v, i, vals = x.to('vulkan'), indices.to('vulkan'), values.to('vulkan')
    vk.reset_execution_stats()
    v.index_put_((i,), vals, accumulate=True)
    stats = vk.execution_stats()
    assert stats['cpu_fallback_calls'] == 0 and stats['host_read_max_bytes'] <= 4, stats
    torch.testing.assert_close(v.cpu(), x.index_put_((indices,), values, accumulate=True))


@pytest.mark.parametrize('dtype', [torch.float32, torch.int32, torch.int64, torch.bool])
@pytest.mark.parametrize('shape', [(), (0,), (3, 4)])
def test_nonzero_native(dtype, shape):
    x = torch.ones(shape, dtype=dtype)
    if x.ndim == 2: x[1, 2] = 0
    v = x.to('vulkan')
    if x.ndim == 2: x, v = x.t(), v.t()
    vk.reset_execution_stats()
    result = v.nonzero()
    stats = vk.execution_stats()
    assert stats['cpu_fallback_calls'] == 0 and stats['host_read_max_bytes'] <= 4, stats
    torch.testing.assert_close(result.cpu(), x.nonzero())


def test_inline_fallback_is_forbidden(monkeypatch):
    v = torch.ones(3, dtype=torch.float64).to('vulkan')
    monkeypatch.setenv('TORCH_VULKAN_FORBID_CPU_FALLBACK', '1')
    with pytest.raises(RuntimeError, match='CPU fallback forbidden'):
        torch.sigmoid(v)


def test_nll_device_target_validation():
    import torch.nn.functional as F
    x = torch.randn(137, 7).log_softmax(-1)
    target = torch.arange(137, dtype=torch.int64) % 7
    v = x.to('vulkan'); t = target.to('vulkan') + 0
    vk.reset_execution_stats()
    result = F.nll_loss(v, t)
    stats = vk.execution_stats()
    assert stats['cpu_fallback_calls'] == 0 and stats['host_read_max_bytes'] <= 4, stats
    torch.testing.assert_close(result.cpu(), F.nll_loss(x, target))


def test_boolean_arithmetic_and_isclose_helpers():
    x = torch.tensor([True, False, True, False])
    y = torch.tensor([True, True, False, False])
    a, b = x.to('vulkan'), y.to('vulkan')
    vals = torch.tensor([float('nan'), float('inf'), -float('inf'), -0., .25]).to('vulkan')
    vk.reset_execution_stats()
    product, total = a * b, a + b
    close = torch.isclose(vals, vals, equal_nan=True)
    finite = torch.isfinite(vals)
    resident()
    torch.testing.assert_close(product.cpu(), x * y)
    torch.testing.assert_close(total.cpu(), x + y)
    assert close.cpu().all()
    torch.testing.assert_close(finite.cpu(), torch.tensor([False, False, False, True, True]))


@pytest.mark.parametrize('left,right', [(torch.float32, torch.int64), (torch.int32, torch.bool), (torch.bool, torch.int64)])
def test_binary_dtype_promotion(left, right):
    a = torch.tensor([1, 2, -3], dtype=left)
    b = torch.tensor([3, -2, 1], dtype=right)
    av, bv = a.to('vulkan'), b.to('vulkan')
    vk.reset_execution_stats()
    total, product = av + bv, av * bv
    resident()
    torch.testing.assert_close(total.cpu(), a + b)
    torch.testing.assert_close(product.cpu(), a * b)


@pytest.mark.parametrize('name', ['reciprocal', 'log', 'log1p', 'expm1', 'floor', 'ceil', 'trunc', 'round', 'sign', 'log10', 'log2', 'exp2', 'sinh', 'cosh', 'tan', 'asin', 'acos', 'atan', 'asinh', 'acosh', 'atanh', 'erf', 'erfc'])
def test_pointwise_math_edges(name):
    x = torch.tensor([-float('inf'), -89., -3., -1., -.99999, -.5, -1e-7, -0., 0., 1e-7, .5, .99999, 1., 1.00001, 3., 89., float('inf'), float('nan')])
    v = x.to('vulkan')
    vk.reset_execution_stats()
    result = getattr(torch, name)(v)
    resident()
    torch.testing.assert_close(result.cpu(), getattr(torch, name)(x), equal_nan=True, rtol=1.3e-5, atol=2e-6)


@pytest.mark.parametrize('decimals', [-3, 0, 3, 99])
def test_round_decimals_native(decimals):
    x = torch.tensor([-.5, .5, 1.5, 2.5, 0.12345, 12345.])
    v = x.to('vulkan')
    vk.reset_execution_stats()
    result = torch.round(v, decimals=decimals)
    resident()
    torch.testing.assert_close(result.cpu(), torch.round(x, decimals=decimals), equal_nan=True)


@pytest.mark.parametrize('dtype', [torch.float32, torch.int32, torch.int64, torch.bool])
def test_native_cat_general_dtype(dtype):
    a = torch.tensor([[0, 2, 3], [1, 0, 4]], dtype=dtype).t()
    b = torch.tensor([[1], [3], [7]], dtype=dtype)
    av, bv = a.to('vulkan'), b.to('vulkan')
    vk.reset_execution_stats()
    result = torch.cat((av, bv), 1)
    resident()
    torch.testing.assert_close(result.cpu(), torch.cat((a, b), 1))


@pytest.mark.parametrize('dtype', [torch.uint8, torch.int8, torch.int16])
def test_subword_native_primitives(dtype):
    info = torch.iinfo(dtype)
    x = torch.tensor([info.min, info.max, 0, 1, 3, 17, 42, 5], dtype=dtype).reshape(2, 4)
    v = x.to('vulkan')
    mask = torch.tensor([[True, False, False, True], [False, True, True, False]])
    mv = mask.to('vulkan')
    vk.reset_execution_stats()
    outputs = [v.to(target) for target in (torch.float32, torch.int32, torch.int64, torch.bool)]
    outputs += [v + v, v * v, v.sum(), torch.where(mv, v, v.flip([1]))]
    resident()
    expected = [x.to(target) for target in (torch.float32, torch.int32, torch.int64, torch.bool)]
    expected += [x + x, x * x, x.sum(), torch.where(mask, x, x.flip([1]))]
    for a, b in zip(outputs, expected): torch.testing.assert_close(a.cpu(), b)
    base = torch.full((3, 9), 7, dtype=dtype, device='vulkan')
    ref = torch.full((3, 9), 7, dtype=dtype)
    vk.reset_execution_stats()
    base[:, 1:8:2].copy_(v[0].expand(3, 4))
    resident()
    ref[:, 1:8:2].copy_(x[0].expand(3, 4))
    torch.testing.assert_close(base.cpu(), ref)


def test_noncanonical_boolean_storage():
    raw = torch.tensor([0, 2, 255, 128, 0, 7, 3, 0, 42], dtype=torch.uint8)
    v = raw.to('vulkan')
    vk.reset_execution_stats()
    b = v.view(torch.bool)
    copied = b.clone().view(torch.uint8)
    floats = b.to(torch.float32)
    truth = b & True
    selected = torch.where(b, v, torch.zeros((), dtype=torch.uint8, device='vulkan'))
    random = torch.randint(2, 255, (1000,), dtype=torch.uint8, device='vulkan')
    resident()
    torch.testing.assert_close(copied.cpu(), raw)
    torch.testing.assert_close(floats.cpu(), raw.bool().float())
    torch.testing.assert_close(truth.cpu(), raw.bool())
    torch.testing.assert_close(selected.cpu(), raw)
    random = random.cpu()
    assert random.min() >= 2 and random.max() < 255


def test_inverse_trigonometric_dense_accuracy():
    x = torch.linspace(-1, 1, 10001)
    v = x.to('vulkan')
    for name in ('asin', 'acos', 'atan'):
        vk.reset_execution_stats()
        actual = getattr(torch, name)(v)
        resident()
        torch.testing.assert_close(actual.cpu(), getattr(torch, name)(x), rtol=1.3e-6, atol=2e-7)


@pytest.mark.parametrize('dtype', [torch.float32, torch.int64, torch.int32, torch.uint8, torch.bool])
def test_masked_fill_and_boolean_reductions(dtype):
    x = torch.tensor([[[0, 1, 2], [3, 0, 1]], [[1, 1, 1], [0, 0, 0]]], dtype=dtype)
    mask = torch.tensor([True, False, True])
    v, mv = x.to('vulkan'), mask.to('vulkan')
    scalar = torch.tensor(1, device='vulkan')
    vk.reset_execution_stats()
    outputs = [v.masked_fill(mv, 1), v.masked_fill(mv, scalar)]
    z = v.transpose(0, 1).clone(); z.masked_fill_(mv, 1)
    reductions = [getattr(torch, name)(v, dim=dims, keepdim=keep) for name in ('any', 'all') for dims in (None, (), (0, 2), (1,)) for keep in (False, True)]
    resident()
    for a in outputs: torch.testing.assert_close(a.cpu(), x.masked_fill(mask, 1))
    torch.testing.assert_close(z.cpu(), x.transpose(0, 1).masked_fill(mask, 1))
    expected = [getattr(torch, name)(x, dim=dims, keepdim=keep) for name in ('any', 'all') for dims in (None, (), (0, 2), (1,)) for keep in (False, True)]
    for a, b in zip(reductions, expected): torch.testing.assert_close(a.cpu(), b)


@pytest.mark.parametrize('dtype', [torch.float32, torch.int64, torch.int32, torch.int8])
def test_clamp_and_integer_arithmetic(dtype):
    x = torch.tensor([[-3, 0, 1, 4], [2, -1, 3, 0]], dtype=dtype)
    lo = torch.tensor([-2, 1, 0, 2], dtype=dtype)
    hi = torch.tensor([[2], [1]], dtype=dtype)
    v, lv, hv = [t.to('vulkan') for t in (x, lo, hi)]
    vk.reset_execution_stats()
    outputs = [v.clamp(-1, 2), v.clamp(lv, hv), v.clamp_min(1.5), v.abs(), -v, v - v.flip([1]), v / 2]
    z=v.clone(); z.mul_(2); z.sub_(1)
    resident()
    expected = [x.clamp(-1, 2), x.clamp(lo, hi), x.clamp_min(1.5), x.abs(), -x, x-x.flip([1]), x/2]
    for a, b in zip(outputs, expected): torch.testing.assert_close(a.cpu(), b)
    torch.testing.assert_close(z.cpu(), x*2-1)


def test_tensor_clamp_nan_and_addmv():
    x=torch.tensor([float('nan'), -2., 0., 2., float('inf')]); lo=torch.tensor([0., float('nan'), -1., 3., 0.]); hi=torch.tensor([1., 1., float('nan'), 2., 5.])
    a,b,c=[t.to('vulkan') for t in (x,lo,hi)]
    vk.reset_execution_stats(); result=a.clamp(b,c); resident()
    torch.testing.assert_close(result.cpu(), x.clamp(lo,hi), equal_nan=True)
    m=torch.randn(4,7); v=torch.randn(7); bias=torch.randn(4)
    mv,vv,bv=[t.to('vulkan') for t in (m,v,bias)]
    vk.reset_execution_stats(); result=torch.addmv(bv,mv,vv,beta=.3,alpha=-.7); resident()
    torch.testing.assert_close(result.cpu(),torch.addmv(bias,m,v,beta=.3,alpha=-.7))


def test_empty_broadcast_arithmetic_and_storage_clone():
    a=torch.empty((0,1,3),device='vulkan'); b=torch.empty((0,10,3),device='vulkan')
    x=torch.randn(5,7,device='vulkan')
    vk.reset_execution_stats(); y=a+b; cloned=x.untyped_storage().clone(); resident()
    assert y.shape==(0,10,3)
    z=torch.empty(0,device='vulkan').set_(cloned,0,x.shape,x.stride())
    torch.testing.assert_close(z.cpu(),x.cpu())


@pytest.mark.parametrize('ord', [0., 1., 2., float('inf'), -float('inf')])
def test_norm_nan_and_zero_semantics(ord):
    x=torch.tensor([[float('nan'), 0., 2.], [0., 0., 0.]])
    v=x.to('vulkan'); vk.reset_execution_stats()
    actual=torch.linalg.vector_norm(v,ord=ord,dim=1); resident()
    torch.testing.assert_close(actual.cpu(),torch.linalg.vector_norm(x,ord=ord,dim=1),equal_nan=True)


def test_division_integer_boundary_accuracy():
    generator=torch.Generator().manual_seed(457)
    denominator=torch.rand(400000,generator=generator)*8+.01
    integers=torch.randint(-20,21,denominator.shape,generator=generator).float()
    numerator=denominator*integers
    numerator=torch.cat((numerator,torch.nextafter(numerator,torch.full_like(numerator,float('-inf'))),torch.nextafter(numerator,torch.full_like(numerator,float('inf')))))
    denominator=denominator.repeat(3)
    a,b=numerator.to('vulkan'),denominator.to('vulkan')
    vk.reset_execution_stats(); actual=(a/b).trunc(); resident()
    torch.testing.assert_close(actual.cpu(),(numerator/denominator).trunc(),rtol=0,atol=0)


def test_division_full_float32_range():
    g=torch.Generator().manual_seed(951)
    bits=torch.randint(-(2**31),2**31,(20000,2),dtype=torch.int32,generator=g)
    a,b=bits.view(torch.float32).unbind(1)
    specials=torch.tensor([0.,-0.,float('inf'),-float('inf'),float('nan'),torch.finfo(torch.float32).tiny,torch.finfo(torch.float32).max])
    a=torch.cat((a,specials[:,None].expand(7,7).reshape(-1)))
    b=torch.cat((b,specials[None,:].expand(7,7).reshape(-1)))
    av,bv=a.to('vulkan'),b.to('vulkan');vk.reset_execution_stats();actual=av/bv;resident()
    reference=(a.double()/b.double()).float()
    torch.testing.assert_close(actual.cpu(),reference,rtol=0,atol=0,equal_nan=True)


@pytest.mark.parametrize('reduce', ['sum', 'prod', 'mean', 'amax', 'amin'])
@pytest.mark.parametrize('include', [True, False])
def test_scatter_reduce_native_forward_backward(reduce, include):
    torch.manual_seed(73)
    x=torch.randn(3,5,requires_grad=True); source=torch.randn(3,7,requires_grad=True)
    index=torch.tensor([[0,0,2,3,2,0,2], [1,1,3,1,3,3,1], [4,4,4,0,0,4,0]])
    xv=x.detach().to('vulkan').requires_grad_(); sv=source.detach().to('vulkan').requires_grad_(); iv=index.to('vulkan')
    expected=x.scatter_reduce(1,index,source,reduce=reduce,include_self=include)
    vk.reset_execution_stats(); actual=xv.scatter_reduce(1,iv,sv,reduce=reduce,include_self=include)
    stats=vk.execution_stats(); assert stats['cpu_fallback_calls']==0 and stats['host_read_max_bytes']<=4,stats
    torch.testing.assert_close(actual.cpu(),expected)
    actual.sum().backward(); expected.sum().backward()
    torch.testing.assert_close(xv.grad.cpu(),x.grad);torch.testing.assert_close(sv.grad.cpu(),source.grad)
    base=torch.zeros((5,3),device='vulkan'); out=base.t()
    torch.scatter_reduce(xv.detach(),1,iv,sv.detach(),reduce=reduce,include_self=include,out=out)
    torch.testing.assert_close(out.cpu(),expected.detach())


@pytest.mark.parametrize('reduce', ['prod', 'mean', 'amax', 'amin'])
@pytest.mark.parametrize('include', [True, False])
def test_index_reduce_native(reduce,include):
    x=torch.tensor([[1.,2.,3.],[4.,5.,6.],[7.,8.,9.],[10.,11.,12.]])
    s=torch.tensor([[2.,1.,-1.],[0.,3.,4.],[5.,-2.,1.]])
    index=torch.tensor([1,1,3])
    xv,sv,iv=[t.to('vulkan') for t in (x,s,index)]
    vk.reset_execution_stats(); actual=torch.index_reduce(xv,0,iv,sv,reduce,include_self=include)
    stats=vk.execution_stats(); assert stats['cpu_fallback_calls']==0 and stats['host_read_max_bytes']<=4,stats
    torch.testing.assert_close(actual.cpu(),torch.index_reduce(x,0,index,s,reduce,include_self=include))
    actual=xv.clone(); actual.index_reduce_(0,iv,sv,reduce,include_self=include)
    torch.testing.assert_close(actual.cpu(),torch.index_reduce(x,0,index,s,reduce,include_self=include))


def test_roll_nansum_std_and_index_copy():
    x=torch.arange(24.).reshape(2,3,4); v=x.to('vulkan')
    vk.reset_execution_stats()
    outputs=[torch.roll(v,(7,-2),(2,0)), torch.roll(v,-5), v.std(dim=(0,2),correction=0),torch.std_mean(v,dim=1,correction=1)]
    resident(); expected=[torch.roll(x,(7,-2),(2,0)),torch.roll(x,-5),x.std(dim=(0,2),correction=0),torch.std_mean(x,dim=1,correction=1)]
    for a,b in zip(outputs,expected):
        if isinstance(a,tuple):
            for aa,bb in zip(a,b):torch.testing.assert_close(aa.cpu(),bb)
        else:torch.testing.assert_close(a.cpu(),b)
    x[0,0,0]=float('nan');v=x.to('vulkan');vk.reset_execution_stats();result=v.nansum(dim=(0,2));resident();torch.testing.assert_close(result.cpu(),x.nansum(dim=(0,2)))
    index=torch.tensor([1,0]); src=torch.randn(2,3,4); iv=index.to('vulkan');sv=src.to('vulkan')
    vk.reset_execution_stats(); result=v.index_copy(0,iv,sv)
    stats=vk.execution_stats();assert stats['cpu_fallback_calls']==0 and stats['host_read_max_bytes']<=4,stats
    torch.testing.assert_close(result.cpu(),x.index_copy(0,index,src))


@pytest.mark.parametrize('n', [0,1,5,17])
@pytest.mark.parametrize('batch', [(),(2,)])
@pytest.mark.parametrize('upper', [False,True])
def test_cholesky_inverse_solve_native(n,batch,upper):
    torch.manual_seed(93)
    raw=torch.randn(*batch,n,n); a=raw@raw.transpose(-1,-2)+torch.eye(n)*.8
    rhs=torch.randn(*batch,n,3); av,bv=a.to('vulkan'),rhs.to('vulkan')
    vk.reset_execution_stats(); factor,info=torch.linalg.cholesky_ex(av,upper=upper)
    solution=torch.cholesky_solve(bv,factor,upper=upper); inverse=torch.cholesky_inverse(factor,upper=upper)
    stats=vk.execution_stats();assert stats['cpu_fallback_calls']==0 and stats['host_read_max_bytes']<=4,stats
    ref=torch.linalg.cholesky(a,upper=upper)
    torch.testing.assert_close(factor.cpu(),ref,rtol=1e-5,atol=3e-6); assert not info.cpu().any()
    torch.testing.assert_close(solution.cpu(),torch.cholesky_solve(rhs,ref,upper=upper),rtol=2e-5,atol=5e-6)
    torch.testing.assert_close(inverse.cpu(),torch.cholesky_inverse(ref,upper=upper),rtol=2e-5,atol=5e-6)
    out=torch.empty_like(av).transpose(-1,-2);status=torch.empty(batch,dtype=torch.int32,device='vulkan')
    torch.linalg.cholesky_ex(av,upper=upper,out=(out,status))
    torch.testing.assert_close(out.cpu(),ref,rtol=1e-5,atol=3e-6)


def test_linalg_inverse_multi_output_and_singular_status():
    a=torch.tensor([[[3.,1.],[1.,2.]],[[2.,-1.],[1.,4.]]]); v=a.to('vulkan')
    out=torch.empty_like(v).transpose(-1,-2); info=torch.empty(2,dtype=torch.int32,device='vulkan')
    vk.reset_execution_stats(); torch.linalg.inv_ex(v,out=(out,info))
    stats=vk.execution_stats();assert stats['cpu_fallback_calls']==0 and stats['host_read_max_bytes']<=4,stats
    torch.testing.assert_close(out.cpu(),torch.linalg.inv(a));assert not info.cpu().any()
    lu=torch.empty_like(v);piv=torch.empty(2,2,dtype=torch.int32,device='vulkan');status=torch.empty_like(info)
    torch.linalg.lu_factor_ex(v,out=(lu,piv,status)); ref=torch.linalg.lu_factor_ex(a)
    for x,y in zip((lu,piv,status),ref):torch.testing.assert_close(x.cpu(),y)
    singular=torch.zeros(3,3,device='vulkan'); factor,status=torch.linalg.cholesky_ex(singular)
    assert status.cpu().item()==1
    with pytest.raises(RuntimeError):torch.linalg.cholesky_ex(singular,check_errors=True)


@pytest.mark.parametrize('left', [True,False])
@pytest.mark.parametrize('upper', [True,False])
def test_triangular_solve_native(left,upper):
    torch.manual_seed(19);a=torch.randn(2,4,4);a.diagonal(dim1=-2,dim2=-1).add_(5)
    b=torch.randn(2,4,3) if left else torch.randn(2,3,4)
    av,bv=a.to('vulkan'),b.to('vulkan')
    vk.reset_execution_stats();actual=torch.linalg.solve_triangular(av,bv,upper=upper,left=left)
    stats=vk.execution_stats();assert stats['cpu_fallback_calls']==0 and stats['host_read_max_bytes']<=4,stats
    torch.testing.assert_close(actual.cpu(),torch.linalg.solve_triangular(a,b,upper=upper,left=left),rtol=1e-5,atol=3e-6)


@pytest.mark.parametrize('probability', [.1,.5,.9])
def test_alpha_dropout_distribution_and_gradient(probability):
    torch.manual_seed(504)
    x=torch.randn(200000).to('vulkan').requires_grad_()
    vk.reset_execution_stats(); y=torch.nn.functional.alpha_dropout(x,p=probability,training=True)
    y.sum().backward(); resident()
    host=y.detach().cpu(); grad=x.grad.cpu()
    # Alpha dropout preserves zero mean/unit variance for normalized inputs.
    assert abs(float(host.mean()))<.02
    assert abs(float(host.var())-1)<.04
    alpha=-1.7580993408473766
    scale=((1-probability)*(1+probability*alpha*alpha))**-.5
    assert torch.all((grad==0)|torch.isclose(grad,torch.tensor(scale),rtol=1e-6,atol=1e-7))
    assert abs(float((grad==0).float().mean())-probability)<.01


@pytest.mark.parametrize('dtype', [torch.bool,torch.int8,torch.int32,torch.int64])
def test_typed_gather_scatter_native(dtype):
    x=torch.tensor([[1,2,3,4],[5,6,7,8]],dtype=dtype)
    if dtype==torch.int64:x=x*2**40+13
    index=torch.tensor([[3,0,3],[1,1,0]])
    source=torch.tensor([[2,3,5],[7,11,13]],dtype=dtype)
    xv,iv,sv=[t.to('vulkan') for t in (x,index,source)]
    vk.reset_execution_stats()
    actual=[xv.gather(1,iv),xv.scatter_add(1,iv,sv),xv.scatter(1,iv,sv),xv.scatter(1,iv,1)]
    stats=vk.execution_stats();assert stats['cpu_fallback_calls']==0 and stats['host_read_max_bytes']<=4,stats
    expected=[x.gather(1,index),x.scatter_add(1,index,source),x.scatter(1,index,source),x.scatter(1,index,1)]
    for a,b in zip(actual,expected):torch.testing.assert_close(a.cpu(),b)

@pytest.mark.parametrize('shape', [(), (13,), (3, 5)])
def test_noncanonical_boolean_sum(shape):
    raw = torch.full(shape, 37, dtype=torch.uint8)
    x = raw.view(torch.bool)
    v = raw.to('vulkan').view(torch.bool)
    vk.reset_execution_stats()
    outputs = [v.sum(), v.nansum(), v.sum(dtype=torch.float32)]
    resident()
    for actual, expected in zip(outputs, [x.sum(), x.nansum(), x.sum(dtype=torch.float32)]):
        torch.testing.assert_close(actual.cpu(), expected)

@pytest.mark.parametrize('dtype', [torch.float32, torch.int32, torch.int64, torch.bool])
def test_mask_selection_broadcast_and_scatter(dtype):
    x = torch.arange(15).reshape(3, 5).to(dtype)
    mask = torch.tensor([[True, False, True, False, True]])
    source = torch.arange(20, 40).to(dtype)
    a, m, s = [t.to('vulkan') for t in (x, mask, source)]
    vk.reset_execution_stats()
    selected, scattered = a.masked_select(m), a.masked_scatter(m, s)
    assert vk.execution_stats()['cpu_fallback_calls'] == 0
    torch.testing.assert_close(selected.cpu(), x.masked_select(mask))
    torch.testing.assert_close(scattered.cpu(), x.masked_scatter(mask, source))
    view = torch.zeros(3, 10, dtype=dtype, device='vulkan')[:, ::2]
    view.copy_(a); view.masked_scatter_(m, s)
    torch.testing.assert_close(view.cpu(), x.masked_scatter(mask, source))

@pytest.mark.parametrize('dtype', [torch.float32, torch.int32, torch.int64, torch.uint8, torch.int16, torch.bool])
def test_general_extrema(dtype):
    x = torch.tensor([[3, 7, 7], [-4, 0, 9]]).to(dtype)
    if dtype == torch.int64: x = x * 2**40 + 3
    if dtype == torch.float32: x[1, 1] = float('nan')
    v = x.to('vulkan').t()
    vk.reset_execution_stats()
    pairs = [(v.max(), x.t().max()), (v.min(), x.t().min()), (v.amax((0, 1)), x.t().amax((0, 1)))]
    for actual, expected in zip(v.max(0), x.t().max(0)): pairs.append((actual, expected))
    if dtype != torch.bool: pairs.append((v.argmin(), x.t().argmin()))
    resident()
    for actual, expected in pairs: torch.testing.assert_close(actual.cpu(), expected, equal_nan=True)

@pytest.mark.parametrize('conjugate', [False, True])
@pytest.mark.parametrize('negative', [False, True])
def test_complex_copy_predicates_and_magnitude(conjugate, negative):
    x = torch.tensor([[1+2j, -3+4j], [complex(float('inf'), 2), complex(0, float('nan'))]], dtype=torch.complex64)
    v = x.to('vulkan')
    if conjugate: x, v = x.conj(), v.conj()
    if negative: x, v = torch._neg_view(x), torch._neg_view(v)
    vk.reset_execution_stats()
    actual = [v.clone(), v.abs(), v == v, torch.isinf(v), v.to(torch.bool), v.to(torch.float32)]
    resident()
    expected = [x.clone(), x.abs(), x == x, torch.isinf(x), x.to(torch.bool), x.to(torch.float32)]
    for a, b in zip(actual, expected): torch.testing.assert_close(a.cpu(), b, equal_nan=True)


def test_efficient_zero_device_materialization():
    z = torch._efficientzerotensor((3, 4), device='vulkan', dtype=torch.float32)
    x = torch.arange(12, dtype=torch.float32).reshape(3, 4).to('vulkan')
    vk.reset_execution_stats()
    actual = z + x
    resident()
    torch.testing.assert_close(actual.cpu(), x.cpu())

@pytest.mark.parametrize('dtype', [torch.int32, torch.int64])
@pytest.mark.parametrize('reduce', ['sum', 'prod', 'mean', 'amax', 'amin'])
@pytest.mark.parametrize('include', [True, False])
def test_integer_scatter_reductions(dtype, reduce, include):
    x = torch.tensor([[-17, 31, 9], [7, -8, 9]], dtype=dtype)
    src = torch.tensor([[-7, 4, 3, -8], [12, -19, 11, 9]], dtype=dtype)
    if dtype == torch.int64:
        x = x * 2**40 + 1
        src = src * 2**36 - 3
    index = torch.tensor([[0, 0, 2, 2], [1, 2, 1, 2]])
    a, i, s = [t.to('vulkan') for t in (x, index, src)]
    vk.reset_execution_stats()
    actual = a.scatter_reduce(1, i, s, reduce, include_self=include)
    assert vk.execution_stats()['cpu_fallback_calls'] == 0
    torch.testing.assert_close(actual.cpu(), x.scatter_reduce(1, index, src, reduce, include_self=include))

@pytest.mark.parametrize('dtype', [torch.float32, torch.int64, torch.bool])
def test_index_fill_general(dtype):
    x = torch.arange(12).reshape(3, 4).to(dtype)
    index = torch.tensor([0, 2])
    a, i = x.to('vulkan'), index.to('vulkan')
    torch.testing.assert_close(a.index_fill(0, i, 7).cpu(), x.index_fill(0, index, 7))
    a.index_fill_(1, i, 0)
    torch.testing.assert_close(a.cpu(), x.index_fill(1, index, 0))

@pytest.mark.parametrize('shape', [(3, 5), (5, 3), (4, 4), (2, 4, 3), (0, 3), (3, 0), (0, 3, 3)])
def test_lu_unpack_general(shape):
    torch.manual_seed(912)
    x = torch.randn(shape)
    v = x.to('vulkan')
    vk.reset_execution_stats()
    p, l, u = torch.linalg.lu(v)
    assert vk.execution_stats()['cpu_fallback_calls'] == 0
    torch.testing.assert_close((p @ l @ u).cpu(), x, atol=2e-5, rtol=2e-5)
    expected = torch.linalg.lu(x)
    for a, b in zip((p, l, u), expected): torch.testing.assert_close(a.cpu(), b, atol=2e-5, rtol=2e-5)

@pytest.mark.parametrize('dtype', [torch.float32, torch.int32, torch.int64, torch.int16, torch.bool])
@pytest.mark.parametrize('dim', [0, 1, -1])
def test_general_scans(dtype, dim):
    x = torch.tensor([[2, -1, -1, 0], [3, 0, 4, 4], [-2, 5, 1, -3]]).to(dtype).t()
    if dtype == torch.int64: x = x * 2**32 + 1
    if dtype == torch.float32: x[1, 1] = float('nan')
    v = x.to('vulkan')
    vk.reset_execution_stats()
    actual = [v.cumsum(dim), v.cumprod(dim), *v.cummax(dim), *v.cummin(dim)]
    resident()
    expected = [x.cumsum(dim), x.cumprod(dim), *x.cummax(dim), *x.cummin(dim)]
    for a, b in zip(actual, expected): torch.testing.assert_close(a.cpu(), b, equal_nan=True)

@pytest.mark.parametrize('reduction', ['none', 'mean', 'sum'])
@pytest.mark.parametrize('shape', [(2, 4, 3, 5), (0, 4, 3, 5), (2, 4, 0, 5)])
def test_nll_spatial_and_empty(shape, reduction):
    torch.manual_seed(991)
    x = torch.randn(shape, requires_grad=True)
    target = torch.randint(0, 4, (shape[0], shape[2], shape[3]))
    weight = torch.rand(4)
    if target.numel(): target.flatten()[0] = -100
    v = x.detach().to('vulkan').requires_grad_()
    t, w = target.to('vulkan'), weight.to('vulkan')
    vk.reset_execution_stats()
    actual = torch.nn.functional.nll_loss(v, t, w, reduction=reduction)
    actual.sum().backward()
    assert vk.execution_stats()['cpu_fallback_calls'] == 0
    expected = torch.nn.functional.nll_loss(x, target, weight, reduction=reduction)
    expected.sum().backward()
    torch.testing.assert_close(actual.cpu(), expected, equal_nan=True)
    torch.testing.assert_close(v.grad.cpu(), x.grad, equal_nan=True)


def test_selection_alias_and_misc_compositions():
    x = torch.arange(6, dtype=torch.float32, device='vulkan')
    mask = torch.tensor([True, False, True, False, False, True], device='vulkan')
    with pytest.raises(RuntimeError, match='unsupported operation'):
        torch.masked_select(x, mask, out=x)
    a = torch.tensor([float('nan'), 1., -2., float('inf')])
    b = torch.tensor([2., float('nan'), -3., float('inf')])
    av, bv = a.to('vulkan'), b.to('vulkan')
    torch.testing.assert_close(torch.fmax(av, bv).cpu(), torch.fmax(a, b), equal_nan=True)
    torch.testing.assert_close(torch.fmin(av, bv).cpu(), torch.fmin(a, b), equal_nan=True)
    for actual, expected in zip(torch.aminmax(av), torch.aminmax(a)): torch.testing.assert_close(actual.cpu(), expected, equal_nan=True)
    c, d = torch.randn(2, 3), torch.randn(1, 3)
    torch.testing.assert_close(torch.linalg.cross(c.to('vulkan'), d.to('vulkan')).cpu(), torch.linalg.cross(c, d))
    m, n = torch.randn(2, 3, 4), torch.randn(2, 4, 5)
    bias = torch.full((5,), float('nan'))
    torch.testing.assert_close(torch.baddbmm(bias.to('vulkan'), m.to('vulkan'), n.to('vulkan'), beta=0, alpha=.5).cpu(), torch.baddbmm(bias, m, n, beta=0, alpha=.5))
    torch.testing.assert_close(torch.count_nonzero(av).cpu(), torch.count_nonzero(a))

@pytest.mark.parametrize('operation', ['fmod', 'remainder', 'floor_divide'])
def test_binary_remainder_full_fp32_range(operation):
    generator = torch.Generator().manual_seed(4423)
    a = torch.randint(-(2**31), 2**31 - 1, (30000,), dtype=torch.int32, generator=generator).view(torch.float32)
    b = torch.randint(-(2**31), 2**31 - 1, (30000,), dtype=torch.int32, generator=generator).view(torch.float32)
    a = torch.cat((a, torch.tensor([0., -0., float('inf'), -float('inf'), float('nan'), 1e-45, -1e-45])))
    b = torch.cat((b, torch.tensor([-1., 2., 3., -4., 5., 1e-45, 2e-45])))
    av, bv = a.to('vulkan'), b.to('vulkan')
    vk.reset_execution_stats()
    actual = getattr(torch, operation)(av, bv)
    resident()
    # CPU vectorized FP32 fmod overflows for large finite quotients. CUDA
    # agrees exactly with the FP64 remainder rounded to FP32 on this corpus.
    expected = getattr(torch, operation)(a, b) if operation == 'floor_divide' else getattr(torch, operation)(a.double(), b.double()).float()
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0, equal_nan=True)
    zeros = (expected == 0)
    torch.testing.assert_close(torch.signbit(actual.cpu()[zeros]), torch.signbit(expected[zeros]))

@pytest.mark.parametrize('operation', ['hypot', 'copysign', 'logaddexp', 'logaddexp2', 'xlogy', 'nextafter', 'pow'])
def test_binary_math_ieee_cases(operation):
    x = torch.tensor([0., -0., 1., -1., 2., -2., float('inf'), -float('inf'), float('nan')])
    y = torch.tensor([0., -0., .5, -.5, 1., 2., 3., -3., float('inf'), -float('inf'), float('nan')])
    a, b = torch.broadcast_tensors(x[:, None], y[None, :])
    av, bv = a.to('vulkan'), b.to('vulkan')
    vk.reset_execution_stats()
    actual = getattr(torch, operation)(av, bv)
    resident()
    expected = getattr(torch, operation)(a, b)
    torch.testing.assert_close(actual.cpu(), expected, equal_nan=True)

@pytest.mark.parametrize('dtype', [torch.float32, torch.int32, torch.int64, torch.bool, torch.int16])
def test_unique_inverse_counts_and_sort(dtype):
    x = torch.tensor([[4, 2, -2, 4, 7], [2, 0, -2, 8, 8]], dtype=dtype)
    if dtype == torch.int64: x = x * 2**40 + 3
    v = x.to('vulkan')
    vk.reset_execution_stats()
    actual = torch.unique(v, return_inverse=True, return_counts=True)
    assert vk.execution_stats()['cpu_fallback_calls'] == 0
    expected = torch.unique(x, return_inverse=True, return_counts=True)
    for a, b in zip(actual, expected): torch.testing.assert_close(a.cpu(), b)
    for a, b in zip(v.sort(dim=0, descending=True), x.sort(dim=0, descending=True)): torch.testing.assert_close(a.cpu(), b)


def test_magnitude_extreme_finite_inputs():
    limit = torch.finfo(torch.float32).max
    a = torch.tensor([limit, limit, 1., 1e-38, 1e-45])
    b = torch.tensor([limit, 1., limit, 1e-38, 1e-45])
    torch.testing.assert_close(torch.hypot(a.to('vulkan'), b.to('vulkan')).cpu(), torch.hypot(a, b), rtol=1e-6, atol=0)
    c = torch.complex(a, b)
    torch.testing.assert_close(c.to('vulkan').abs().cpu(), c.abs(), rtol=1e-6, atol=0)

@pytest.mark.parametrize('dtype', [torch.int32, torch.int64])
@pytest.mark.parametrize('op', ['fmod', 'remainder', 'floor_divide', 'trunc', 'pow'])
def test_exact_integer_math(dtype, op):
    gen = torch.Generator().manual_seed(988)
    lo, hi = -(2**30), 2**30
    a = torch.randint(lo, hi, (4000,), generator=gen, dtype=dtype)
    b = torch.randint(-999, 999, (4000,), generator=gen, dtype=dtype)
    b[b == 0] = 1
    if dtype == torch.int64: a = a * 2**32 + 31; b = b * 2**27 + 1
    if op == 'pow': b = torch.randint(-3, 8, (4000,), generator=gen, dtype=dtype)
    av, bv = a.to('vulkan'), b.to('vulkan')
    fn = (lambda x, y: torch.div(x, y, rounding_mode='trunc')) if op == 'trunc' else getattr(torch, op)
    vk.reset_execution_stats(); actual = fn(av, bv)
    assert vk.execution_stats()['cpu_fallback_calls'] == 0
    torch.testing.assert_close(actual.cpu(), fn(a, b), rtol=0, atol=0)

@pytest.mark.parametrize('name', ['lgamma', 'digamma', 'polygamma1', 'polygamma2', 'polygamma3', 'polygamma5'])
def test_gamma_functions_and_gradients(name):
    x = torch.cat((torch.linspace(.001, 20, 3001), torch.linspace(-20.7, -.3, 2001), torch.tensor([1e-38, 1e-45, -1e-38, -1e-45, 0., -0., -1., -2., float('inf'), -float('inf'), float('nan')])))
    fn = getattr(torch, name) if not name.startswith('polygamma') else lambda t: torch.polygamma(int(name[-1]), t)
    v = x.to('vulkan'); vk.reset_execution_stats(); actual = fn(v); resident()
    torch.testing.assert_close(actual.cpu(), fn(x), rtol=2e-5, atol=2e-5, equal_nan=True)
    xg = torch.linspace(.2, 4, 71, requires_grad=True); vg = xg.detach().to('vulkan').requires_grad_()
    fn(xg).sum().backward(); fn(vg).sum().backward()
    torch.testing.assert_close(vg.grad.cpu(), xg.grad, rtol=2e-5, atol=2e-5, equal_nan=True)

@pytest.mark.parametrize('consecutive', [True, False])
@pytest.mark.parametrize('dim', [None, 0, 1])
@pytest.mark.parametrize('dtype', [torch.float32, torch.int64, torch.bool])
def test_unique_rows_and_consecutive(dtype, dim, consecutive):
    x = torch.tensor([[1, 2, 1, 2], [1, 2, 1, 2], [3, 0, 3, 0], [1, 2, 1, 2]], dtype=dtype)
    if dtype == torch.int64: x = x * 2**40 + 7
    fn = torch.unique_consecutive if consecutive else torch.unique
    v = x.to('vulkan'); vk.reset_execution_stats()
    actual = fn(v, dim=dim, return_inverse=True, return_counts=True)
    assert vk.execution_stats()['cpu_fallback_calls'] == 0
    expected = fn(x, dim=dim, return_inverse=True, return_counts=True)
    for a, b in zip(actual, expected): torch.testing.assert_close(a.cpu(), b)

@pytest.mark.parametrize('rank', [2, 3])
@pytest.mark.parametrize('batched', [True, False])
def test_adaptive_max_and_unpool(rank, batched):
    torch.manual_seed(1212)
    shape = ((2, 3) if batched else (3,)) + (6,) * rank
    x = torch.randn(shape, requires_grad=True); v = x.detach().to('vulkan').requires_grad_()
    fn = getattr(torch.nn.functional, f'adaptive_max_pool{rank}d')
    vk.reset_execution_stats(); actual, indices = fn(v, (3,) * rank, return_indices=True)
    actual.sum().backward(); assert vk.execution_stats()['cpu_fallback_calls'] == 0
    expected, reference_indices = fn(x, (3,) * rank, return_indices=True); expected.sum().backward()
    torch.testing.assert_close(actual.cpu(), expected); torch.testing.assert_close(indices.cpu(), reference_indices); torch.testing.assert_close(v.grad.cpu(), x.grad)
    unpool = getattr(torch.nn.functional, f'max_unpool{rank}d')
    actual_unpool = unpool(actual.detach(), indices, 2, output_size=shape[-rank:])
    expected_unpool = unpool(expected.detach(), reference_indices, 2, output_size=shape[-rank:])
    torch.testing.assert_close(actual_unpool.cpu(), expected_unpool)


def test_logcumsumexp_and_product():
    x = torch.tensor([[float('-inf'), -1000., -40., 1., 40., 1000.], [0., 1., float('nan'), 4., 5., 6.]])
    v = x.to('vulkan'); vk.reset_execution_stats(); actual = v.logcumsumexp(1); resident()
    torch.testing.assert_close(actual.cpu(), x.logcumsumexp(1), equal_nan=True)
    a = torch.tensor([[2, -3, 4], [7, 8, 9]], dtype=torch.int64)
    torch.testing.assert_close(a.to('vulkan').prod(1).cpu(), a.prod(1))

@pytest.mark.parametrize('name', ['elu', 'celu', 'selu', 'softplus', 'hardshrink', 'softshrink', 'hardsigmoid', 'hardswish', 'mish', 'logit', 'glu'])
def test_general_activation_forward_and_gradient(name):
    torch.manual_seed(992)
    x = torch.randn(3, 8, 5).transpose(1, 2).requires_grad_()
    if name == 'logit': x = x.sigmoid().detach().requires_grad_()
    v = x.detach().to('vulkan').requires_grad_()
    fn = torch.logit if name == 'logit' else getattr(torch.nn.functional, name)
    options = {'dim': 2} if name == 'glu' else {}
    vk.reset_execution_stats(); a = fn(v, **options); a.sum().backward(); resident()
    b = fn(x, **options); b.sum().backward()
    torch.testing.assert_close(a.cpu(), b, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(v.grad.cpu(), x.grad, rtol=2e-5, atol=2e-6)

@pytest.mark.parametrize('name', ['elu', 'softplus', 'hardshrink', 'softshrink', 'hardsigmoid', 'hardswish', 'mish'])
def test_activation_nonfinite_and_boundary_values(name):
    x = torch.tensor([-float('inf'), -100., -3., -.5, -0., 0., .5, 3., 100., float('inf'), float('nan')], requires_grad=True)
    v = x.detach().to('vulkan').requires_grad_(); fn = getattr(torch.nn.functional, name)
    a = fn(v); b = fn(x); a.sum().backward(); b.sum().backward()
    torch.testing.assert_close(a.cpu(), b, equal_nan=True)
    torch.testing.assert_close(v.grad.cpu(), x.grad, equal_nan=True)


def test_general_selection_primitives():
    x = torch.arange(24).reshape(2, 3, 4).transpose(0, 2)
    index = torch.tensor([[-1, 0], [3, -5]])
    torch.testing.assert_close(torch.take(x.to('vulkan'), index.to('vulkan')).cpu(), torch.take(x, index))
    src = torch.tensor([3, 4, 5, 6]); idx = index.reshape(-1)
    a = x.to('vulkan'); b = x.clone(); a.put_(idx.to('vulkan'), src.to('vulkan'), accumulate=True); b.put_(idx, src, accumulate=True)
    torch.testing.assert_close(a.cpu(), b)
    for size in [0, 2, 30]: torch.testing.assert_close(torch.nonzero_static(a, size=size, fill_value=7).cpu(), torch.nonzero_static(b, size=size, fill_value=7))
    z = torch.randn(2, 6, 3, 4)
    torch.testing.assert_close(torch.channel_shuffle(z.to('vulkan'), 3).cpu(), torch.channel_shuffle(z, 3))

@pytest.mark.parametrize('n', [0, 1, 3, 9])
@pytest.mark.parametrize('batch', [(), (2,), (2, 1)])
@pytest.mark.parametrize('uplo', ['L', 'U'])
def test_eigh_and_determinant(n, batch, uplo):
    torch.manual_seed(132); x = torch.randn(batch + (n, n)); x = (x + x.mT) * .5
    v = x.to('vulkan'); vk.reset_execution_stats(); w, q = torch.linalg.eigh(v, UPLO=uplo)
    assert vk.execution_stats()['cpu_fallback_calls'] == 0
    torch.testing.assert_close(w.cpu(), torch.linalg.eigvalsh(x, UPLO=uplo), rtol=2e-5, atol=2e-6)
    torch.testing.assert_close((q @ torch.diag_embed(w) @ q.mT).cpu(), x, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(torch.linalg.det(v).cpu(), torch.linalg.det(x), rtol=3e-5, atol=2e-5)
    a = torch.linalg.slogdet(v); b = torch.linalg.slogdet(x)
    for aa, bb in zip(a, b): torch.testing.assert_close(aa.cpu(), bb, rtol=2e-5, atol=2e-5)

@pytest.mark.parametrize('shape', [(0, 3), (3, 0), (1, 1), (7, 3), (3, 7), (2, 5, 4)])
@pytest.mark.parametrize('mode', ['reduced', 'complete', 'r'])
def test_householder_qr_and_factor(shape, mode):
    torch.manual_seed(341); x = torch.randn(shape); v = x.to('vulkan')
    vk.reset_execution_stats(); q, r = torch.linalg.qr(v, mode=mode)
    assert vk.execution_stats()['cpu_fallback_calls'] == 0
    qc, rc = torch.linalg.qr(x, mode=mode)
    torch.testing.assert_close(r.cpu(), rc, rtol=2e-5, atol=3e-6)
    if mode != 'r':
        torch.testing.assert_close(q.cpu(), qc, rtol=2e-5, atol=3e-6)
        torch.testing.assert_close((q @ r).cpu(), x, rtol=2e-5, atol=3e-6)
    factor, tau = torch.geqrf(v); fc, tc = torch.geqrf(x)
    torch.testing.assert_close(factor.cpu(), fc, rtol=2e-5, atol=3e-6)
    torch.testing.assert_close(tau.cpu(), tc, rtol=2e-5, atol=3e-6)

@pytest.mark.parametrize('transpose', [False, True])
@pytest.mark.parametrize('left', [False, True])
def test_ormqr_general(left, transpose):
    torch.manual_seed(342); x = torch.randn(2, 6, 4); b = torch.randn((2, 6, 3) if left else (2, 3, 6))
    f, t = torch.geqrf(x); fv, tv = torch.geqrf(x.to('vulkan'))
    actual = torch.ormqr(fv, tv, b.to('vulkan'), left=left, transpose=transpose)
    torch.testing.assert_close(actual.cpu(), torch.ormqr(f, t, b, left=left, transpose=transpose), rtol=2e-5, atol=3e-6)

@pytest.mark.parametrize('op', ['det', 'slogdet'])
@pytest.mark.parametrize('layout', ['contiguous', 'transpose', 'slice'])
def test_determinant_gradients(op, layout):
    torch.manual_seed(779); x = torch.randn(5, 5) + torch.eye(5) * 3
    if layout == 'transpose': x = x.t()
    if layout == 'slice': x = torch.stack((x, x), -1)[..., 0]
    v = torch.empty_strided(x.shape, x.stride(), device='vulkan').copy_(x).requires_grad_(); x.requires_grad_()
    a, b = getattr(torch.linalg, op)(v), getattr(torch.linalg, op)(x)
    if op == 'slogdet': a, b = a.logabsdet, b.logabsdet
    a.sum().backward(); b.sum().backward()
    torch.testing.assert_close(v.grad.cpu(), x.grad, rtol=2e-5, atol=2e-6)


def test_complex_random_and_construction():
    radius, angle = torch.randn(3, 1).abs(), torch.randn(1, 4)
    torch.testing.assert_close(torch.polar(radius.to('vulkan'), angle.to('vulkan')).cpu(), torch.polar(radius, angle))
    v = torch.empty(100000, dtype=torch.complex64, device='vulkan').normal_(3, 2).cpu()
    assert abs(v.real.mean().item() - 3) < .03 and abs(v.imag.mean().item() - 3) < .03
    assert abs(v.real.var().item() - 2) < .06 and abs(v.imag.var().item() - 2) < .06

@pytest.mark.parametrize('dtype', [torch.float32, torch.int64])
def test_indexing_empty_scalar_and_negative_indices(dtype):
    x = torch.arange(6, dtype=dtype).reshape(2, 3); v = x.to('vulkan')
    for shape in [(0,), (2, 0)]:
        i = torch.empty(shape, dtype=torch.uint8)
        torch.testing.assert_close(torch.gather(v, 0, i.to('vulkan')).cpu(), torch.gather(x, 0, i))
    i = torch.tensor([-1, -3])
    torch.testing.assert_close(torch.index_fill(v, 1, i.to('vulkan'), 9).cpu(), torch.index_fill(x, 1, i, 9))
    s = torch.tensor(1, dtype=dtype); src = torch.tensor([9], dtype=dtype); idx = torch.tensor([0])
    torch.testing.assert_close(torch.index_copy(s.to('vulkan'), 0, idx.to('vulkan'), src.to('vulkan')).cpu(), torch.index_copy(s, 0, idx, src))


def test_complex_reduction_and_approximate_gelu():
    x = torch.randn(3, 7, dtype=torch.complex64)
    for d in [0, 1]: torch.testing.assert_close(x.to('vulkan').sum(d).cpu(), x.sum(d))
    z = torch.linspace(-10, 10, 2001, requires_grad=True); v = z.detach().to('vulkan').requires_grad_()
    a = torch.nn.functional.gelu(v, approximate='tanh'); b = torch.nn.functional.gelu(z, approximate='tanh')
    a.sum().backward(); b.sum().backward()
    torch.testing.assert_close(a.cpu(), b); torch.testing.assert_close(v.grad.cpu(), z.grad)

@pytest.mark.parametrize('name', ['sin', 'cos'])
def test_trigonometry_full_fp32_range(name):
    gen = torch.Generator().manual_seed(161); bits = torch.randint(-(2**31), 2**31, (20000,), dtype=torch.int32, generator=gen)
    x = bits.view(torch.float32); x = torch.cat((x, torch.tensor([-0., 0., -float('inf'), float('inf'), float('nan'), 1e20, -1e20, torch.finfo(torch.float32).max])))
    v = x.to('vulkan'); vk.reset_execution_stats(); actual = getattr(torch, name)(v); resident()
    torch.testing.assert_close(actual.cpu(), getattr(torch, name)(x), rtol=2e-6, atol=1e-7, equal_nan=True)


def test_compensated_reduction_cancellation_and_nonfinite():
    x = torch.tensor([[1e10, 1., -1e10, 2.], [float('inf'), 1., 2., 3.], [float('-inf'), float('inf'), 1., 2.]])
    actual = x.to('vulkan').sum(1).cpu()
    torch.testing.assert_close(actual, x.double().sum(1).float(), rtol=0, atol=0, equal_nan=True)


def test_zero_tensor_read_primitives():
    z = torch._efficientzerotensor((3, 5), device='vulkan')
    assert torch.isclose(z, z).all().item()
    torch.testing.assert_close((z + 2).cpu(), torch.full((3, 5), 2.))
    torch.testing.assert_close(torch.sin(z).cpu(), torch.zeros(3, 5))

@pytest.mark.parametrize('dtype', [torch.int32, torch.int64])
@pytest.mark.parametrize('batched', [False, True])
def test_exact_integer_matmul(dtype, batched):
    torch.manual_seed(681); a = torch.randint(-100, 100, (2, 7, 3) if batched else (7, 3), dtype=dtype); b = torch.randint(-100, 100, (2, 3, 5) if batched else (3, 5), dtype=dtype)
    if dtype == torch.int64: a = a * 2**40 + 9
    fn = torch.bmm if batched else torch.mm
    torch.testing.assert_close(fn(a.to('vulkan'), b.to('vulkan')).cpu(), fn(a, b), rtol=0, atol=0)

@pytest.mark.parametrize('shape', [(3, 7), (7, 3), (2, 4, 4)])
def test_polar_matrix_decomposition(shape):
    torch.manual_seed(682); x = torch.randn(shape); v = x.to('vulkan')
    if shape[-2] < shape[-1]:
        for data in [x, v]:
            with pytest.raises(RuntimeError, match='at least as many rows as columns'): torch.linalg.polar(data)
        return
    u, h = torch.linalg.polar(v)
    torch.testing.assert_close((u @ h).cpu(), x, rtol=2e-5, atol=3e-6)
    torch.testing.assert_close(h.cpu(), h.mT.cpu(), rtol=2e-5, atol=3e-6)
    a, b = torch.linalg.polar(x)
    torch.testing.assert_close(u.cpu(), a, rtol=2e-5, atol=3e-6);torch.testing.assert_close(h.cpu(), b, rtol=2e-5, atol=3e-6)
