"""Independent semantic checks for failures exposed by the upstream audit."""
import warnings
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

@pytest.mark.parametrize('dtype', [torch.float32, torch.int64, torch.int32])
@pytest.mark.parametrize('right', [False, True])
@pytest.mark.parametrize('batched', [False, True])
def test_binary_search_sorted_boundaries(dtype, right, batched):
    boundaries = torch.tensor([[-9, -3, -3, 0, 7, 99], [-1, 0, 3, 8, 8, 15]], dtype=dtype)
    query = torch.tensor([[-99, -3, -1, 7, 100], [-2, 0, 5, 8, 99]], dtype=dtype)
    if not batched: boundaries = boundaries[0]
    if dtype == torch.int64: boundaries = boundaries * 2**40 + 7; query = query * 2**40 + 7
    bv, qv = boundaries.to('vulkan'), query.to('vulkan')
    for out_int32 in [False, True]:
        vk.reset_execution_stats(); actual = torch.searchsorted(bv, qv, right=right, out_int32=out_int32); resident()
        torch.testing.assert_close(actual.cpu(), torch.searchsorted(boundaries, query, right=right, out_int32=out_int32))
    if not batched: torch.testing.assert_close(torch.bucketize(qv, bv, right=right).cpu(), torch.bucketize(query, boundaries, right=right))


def test_searchsorted_sorter_and_nonfinite():
    x = torch.tensor([8., -4., float('inf'), 0., -float('inf')]); order = x.argsort(); q = torch.tensor([-float('inf'), -1., 0., 7., float('inf'), float('nan')])
    for right in [False, True]:
        actual = torch.searchsorted(x.to('vulkan'), q.to('vulkan'), sorter=order.to('vulkan'), right=right)
        torch.testing.assert_close(actual.cpu(), torch.searchsorted(x, q, sorter=order, right=right))


def test_atan2_axes_signed_zero_extremes():
    axis = torch.tensor([-float('inf'), -torch.finfo(torch.float32).max, -1., -1e-38, -1e-45, -0., 0., 1e-45, 1e-38, 1., torch.finfo(torch.float32).max, float('inf'), float('nan')])
    y, x = axis[:, None], axis[None, :]
    torch.testing.assert_close(torch.atan2(y.to('vulkan'), x.to('vulkan')).cpu(), torch.atan2(y, x), rtol=2e-6, atol=1e-7, equal_nan=True)
    parts = torch.complex(x.expand(13, 13), y.expand(13, 13))
    torch.testing.assert_close(torch.angle(parts.to('vulkan')).cpu(), torch.angle(parts), rtol=2e-6, atol=1e-7, equal_nan=True)


def test_frexp_exact_float_bits():
    gen = torch.Generator().manual_seed(393); bits = torch.randint(-(2**31), 2**31, (20000,), dtype=torch.int32, generator=gen)
    x = bits.view(torch.float32); actual = torch.frexp(x.to('vulkan')); expected = torch.frexp(x)
    for a, b in zip(actual, expected): torch.testing.assert_close(a.cpu(), b, rtol=0, atol=0, equal_nan=True)

@pytest.mark.parametrize('name', ['sinc', 'frac', 'special.entr'])
def test_additional_pointwise_functions(name):
    x = torch.cat((torch.linspace(-10, 10, 2001), torch.tensor([-float('inf'), float('inf'), float('nan'), -0., 1e-45, -1e-45, 1e20])))
    fn = torch.special.entr if name == 'special.entr' else getattr(torch, name)
    torch.testing.assert_close(fn(x.to('vulkan')).cpu(), fn(x), rtol=2e-5, atol=2e-6, equal_nan=True)

@pytest.mark.parametrize('mode', ['nearest', 'nearest-exact', 'linear'])
@pytest.mark.parametrize('length,output', [(1, 7), (3, 3), (3, 6), (7, 3), (8, 13)])
def test_resampling_1d_forward_backward(mode, length, output):
    torch.manual_seed(539); x = torch.randn(2, 3, length, requires_grad=True); v = x.detach().to('vulkan').requires_grad_()
    options = {'align_corners': False} if mode == 'linear' else {}
    a = torch.nn.functional.interpolate(v, size=output, mode=mode, **options); b = torch.nn.functional.interpolate(x, size=output, mode=mode, **options)
    a.sum().backward(); b.sum().backward()
    torch.testing.assert_close(a.cpu(), b, rtol=2e-5, atol=2e-6); torch.testing.assert_close(v.grad.cpu(), x.grad, rtol=2e-5, atol=2e-6)

@pytest.mark.parametrize('rank', [2, 3])
@pytest.mark.parametrize('mode', ['nearest', 'nearest-exact'])
def test_resampling_nearest_all_spatial_ranks(rank, mode):
    torch.manual_seed(545); x = torch.randn((2, 3) + (7,) * rank, requires_grad=True); v = x.detach().to('vulkan').requires_grad_()
    size = (3, 5) if rank == 2 else (3, 5, 9)
    a = torch.nn.functional.interpolate(v, size=size, mode=mode); b = torch.nn.functional.interpolate(x, size=size, mode=mode)
    a.sum().backward(); b.sum().backward()
    torch.testing.assert_close(a.cpu(), b); torch.testing.assert_close(v.grad.cpu(), x.grad)

@pytest.mark.parametrize('dtype', [torch.float32, torch.int64])
def test_searchsorted_noncontiguous_sorter_lifetime(dtype):
    boundaries = torch.tensor([8, -4, 6, 1, -9, 3, 0, 11, 7, 2], dtype=dtype)
    query = torch.arange(-11, 13, dtype=dtype)
    storage = torch.empty(20, dtype=torch.int64, device='vulkan')
    sorter = storage[1::2]
    sorter.copy_(boundaries.argsort())
    bv, qv = boundaries.to('vulkan'), query.to('vulkan')
    for right in [False, True]:
        torch.testing.assert_close(torch.searchsorted(bv, qv, sorter=sorter, right=right).cpu(), torch.searchsorted(boundaries, query, sorter=boundaries.argsort(), right=right))

@pytest.mark.parametrize('eps', [-.5, float('nan'), float('inf')])
def test_normalization_unrestricted_epsilon(eps):
    x = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4) / 10
    for fn in [lambda v: torch.nn.functional.layer_norm(v, (4,), eps=eps), lambda v: torch.nn.functional.group_norm(v, 1, eps=eps)]:
        torch.testing.assert_close(fn(x.to('vulkan')).cpu(), fn(x), equal_nan=True)

@pytest.mark.parametrize('shape,normalized', [((0, 1), (1,)), ((2, 0), (0,)), ((2, 0, 3), (0, 3))])
def test_layer_norm_empty_forward_backward(shape, normalized):
    x = torch.empty(shape, requires_grad=True); v = x.detach().to('vulkan').requires_grad_()
    a = torch.native_layer_norm(v, normalized, None, None, 1e-5); b = torch.native_layer_norm(x, normalized, None, None, 1e-5)
    for av, bv in zip(a, b): torch.testing.assert_close(av.cpu(), bv, equal_nan=True)
    a[0].sum().backward(); b[0].sum().backward();torch.testing.assert_close(v.grad.cpu(), x.grad)


def test_variance_forward_ad_decomposition():
    x = torch.arange(25, dtype=torch.float32).reshape(5, 5) / 10; tangent = x.square().sub(1)
    results = []
    for device in ['cpu', 'vulkan']:
        with torch.autograd.forward_ad.dual_level():
            dual = torch.autograd.forward_ad.make_dual(x.to(device), tangent.to(device))
            result = torch.autograd.forward_ad.unpack_dual(torch.var(dual, unbiased=True))
            assert result.tangent is not None
            results.append((result.primal.cpu(), result.tangent.cpu()))
    torch.testing.assert_close(results[0], results[1])

@pytest.mark.parametrize('name', ['lerp', 'addcmul', 'addcdiv', 'ldexp'])
def test_inplace_broadcast_must_not_resize(name):
    x = torch.ones(5, device='vulkan'); larger = torch.ones(5, 5, device='vulkan')
    before = x.clone()
    with pytest.raises(RuntimeError):
        if name == 'lerp': x.lerp_(larger, .5)
        elif name == 'ldexp': x.ldexp_(larger.to(torch.int64))
        else: getattr(x, name + '_')(larger, larger)
    assert x.shape == (5,)
    torch.testing.assert_close(x.cpu(), before.cpu())

@pytest.mark.parametrize('dtype', [torch.float32, torch.int32, torch.int64])
@pytest.mark.parametrize('dim', [0, 1, -1])
@pytest.mark.parametrize('keep', [False, True])
def test_order_selection_and_indices(dtype, dim, keep):
    x = torch.tensor([[9, 1, 1, 4, 0], [2, 3, 2, 2, 3], [8, 4, 4, 4, 1]], dtype=dtype)
    if dtype == torch.int64: x = x * 2**40 + 7
    v = x.to('vulkan'); d = dim % 2
    for name in ['median', 'nanmedian', 'mode', 'kthvalue']:
        args = (1, dim, keep) if name == 'kthvalue' else (dim, keep)
        av, ai = getattr(torch, name)(v, *args); bv, bi = getattr(torch, name)(x, *args)
        torch.testing.assert_close(av.cpu(), bv)
        index = ai.cpu() if keep else ai.cpu().unsqueeze(d)
        reconstructed = x.gather(d, index)
        if not keep: reconstructed = reconstructed.squeeze(d)
        torch.testing.assert_close(av.cpu(), reconstructed)


def test_nanmedian_mixed_empty_and_scalar():
    x = torch.tensor([[float('nan'), 1., 3., float('nan')], [float('nan')] * 4, [1., 2., 3., 4.]])
    av, ai = torch.nanmedian(x.to('vulkan'), 1); bv, _ = torch.nanmedian(x, 1)
    torch.testing.assert_close(av.cpu(), bv, equal_nan=True)
    torch.testing.assert_close(torch.median(x.to('vulkan')).cpu(), torch.median(x), equal_nan=True)
    for name in ['median', 'nanmedian']:
        for value in [torch.tensor(2.), torch.empty(0)]: torch.testing.assert_close(getattr(torch, name)(value.to('vulkan')).cpu(), getattr(torch, name)(value), equal_nan=True)

@pytest.mark.parametrize('dtype', [torch.float32, torch.int32, torch.int64])
@pytest.mark.parametrize('invert', [False, True])
def test_isin_exact_ordered_search(dtype, invert):
    x = torch.tensor([1, 3, 7, -4, 0, 9, 9], dtype=dtype); values = torch.tensor([9, -4, 9, 0], dtype=dtype)
    if dtype == torch.int64: x = x * 2**40 + 1; values = values * 2**40 + 1
    torch.testing.assert_close(torch.isin(x.to('vulkan'), values.to('vulkan'), invert=invert).cpu(), torch.isin(x, values, invert=invert))
    torch.testing.assert_close(torch.isin(x.to('vulkan'), values[0].item(), invert=invert).cpu(), torch.isin(x, values[0].item(), invert=invert))

@pytest.mark.parametrize('dtype', [torch.int32, torch.int64])
def test_repeat_interleave_dynamic_indices(dtype):
    repeats = torch.tensor([2, 0, 3, 1, 0, 0, 4], dtype=dtype)
    v = torch.zeros(14, dtype=dtype, device='vulkan')[1::2]; v.copy_(repeats)
    for size in [None, 10]: torch.testing.assert_close(torch.repeat_interleave(v, output_size=size).cpu(), torch.repeat_interleave(repeats, output_size=size))
    with pytest.raises(RuntimeError): torch.repeat_interleave(v, output_size=9)
    with pytest.raises(RuntimeError): torch.repeat_interleave(torch.tensor([1, -1], dtype=dtype, device='vulkan'))

@pytest.mark.parametrize('p', [1., 2., float('inf')])
def test_renorm_general_norms(p):
    x = torch.arange(60, dtype=torch.float32).reshape(3, 4, 5).div(10).sub(3)
    for dim in [0, 1, -1]: torch.testing.assert_close(torch.renorm(x.to('vulkan'), p, dim, 2).cpu(), torch.renorm(x, p, dim, 2))

@pytest.mark.parametrize('name', ['smooth_l1_loss', 'huber_loss', 'binary_cross_entropy'])
@pytest.mark.parametrize('reduction', ['none', 'mean', 'sum'])
def test_additional_losses_forward_gradients(name, reduction):
    x = torch.tensor([[0., .1, .5, .9, 1.], [.3, .5, .7, .8, .2]], requires_grad=True)
    target = torch.tensor([[0., 1., 0., 1., 1.], [1., 1., 0., .5, .2]])
    v = x.detach().to('vulkan').requires_grad_(); fn = getattr(torch.nn.functional, name)
    options = {'weight': torch.tensor([1., .5, 2., 0., 1.])} if name == 'binary_cross_entropy' else {}
    a = fn(v, target.to('vulkan'), reduction=reduction, **{k: t.to('vulkan') for k, t in options.items()}); b = fn(x, target, reduction=reduction, **options)
    a.sum().backward(); b.sum().backward()
    torch.testing.assert_close(a.cpu(), b); torch.testing.assert_close(v.grad.cpu(), x.grad)

@pytest.mark.parametrize('name', ['erfinv', 'i0', 'special_i0e', 'special_i1', 'special_i1e', 'special_ndtri', 'special_erfcx', 'special_log_ndtr', 'special_airy_ai', 'special_bessel_j0', 'special_bessel_j1', 'special_bessel_y0', 'special_bessel_y1', 'special_spherical_bessel_j0', 'special_modified_bessel_i0', 'special_modified_bessel_i1', 'special_modified_bessel_k0', 'special_modified_bessel_k1', 'special_scaled_modified_bessel_k0', 'special_scaled_modified_bessel_k1'])
@pytest.mark.parametrize('noncontiguous', [False, True])
def test_additional_special_unary_functions(name, noncontiguous):
    x = torch.cat((torch.linspace(-10, 10, 2001), torch.linspace(-1.01, 1.01, 1001), torch.tensor([float('nan'), float('inf'), -float('inf'), -0., 0., 1e-45, 1e-38, -1e-38, 20., 40., 80., 100., 1e5, 1e20, -1e20])))
    fn = getattr(torch, name) if name in ['erfinv', 'i0'] else getattr(torch.special, name.removeprefix('special_'))
    v = x.to('vulkan')
    if noncontiguous:
        storage = torch.empty(x.numel()*2+3, device='vulkan');v=storage[3::2];v.copy_(x)
    vk.reset_execution_stats(); actual = fn(v); resident()
    torch.testing.assert_close(actual.cpu(), fn(x), equal_nan=True)
    output = torch.empty_like(v); fn(v, out=output)
    torch.testing.assert_close(output.cpu(), fn(x), equal_nan=True)

@pytest.mark.parametrize('name', ['special_chebyshev_polynomial_t', 'special_chebyshev_polynomial_u', 'special_chebyshev_polynomial_v', 'special_chebyshev_polynomial_w', 'special_hermite_polynomial_h', 'special_hermite_polynomial_he', 'special_laguerre_polynomial_l', 'special_legendre_polynomial_p', 'special_shifted_chebyshev_polynomial_t', 'special_shifted_chebyshev_polynomial_u', 'special_shifted_chebyshev_polynomial_v', 'special_shifted_chebyshev_polynomial_w', 'special_zeta'])
def test_additional_special_binary_functions(name):
    fn = getattr(torch.special, name.removeprefix('special_'))
    x = torch.linspace(-2, 2, 201)[:, None]; n = torch.arange(-2, 16, dtype=torch.float32)[None, :]
    if name == 'special_zeta': x=x+3.01;n=n+3.03
    xv, nv = x.to('vulkan'), n.to('vulkan')
    vk.reset_execution_stats();actual=fn(xv,nv);resident()
    torch.testing.assert_close(actual.cpu(),fn(x,n),equal_nan=True)
    torch.testing.assert_close(fn(xv,3.).cpu(),fn(x,3.),equal_nan=True)
    torch.testing.assert_close(fn(.25,nv).cpu(),fn(.25,n),equal_nan=True)

@pytest.mark.parametrize('values', [[1.25, -2.5, 0.], [float('inf'), -float('inf'), float('nan')]])
def test_fp32_tensor_formatting_without_double_fallback(values):
    value=torch.tensor(values,device='vulkan');vk.reset_execution_stats()
    text=repr(value)
    assert 'vulkan' in text
    assert vk.execution_stats()['cpu_fallback_calls']==0
    import torch._tensor_str as tensor_str
    assert tensor_str.tensor_totype(torch.tensor(1.)).dtype==torch.float64


def test_inplace_renorm_keeps_native_gradients():
    x=torch.arange(30,dtype=torch.float32).reshape(2,3,5).div(10).sub(1).requires_grad_();v=x.detach().to('vulkan').requires_grad_()
    for p in [2.,float('inf')]:
        a=v.clone().renorm_(p,1,.5);b=x.clone().renorm_(p,1,.5)
        ag=torch.autograd.grad(a.sum(),v)[0];bg=torch.autograd.grad(b.sum(),x)[0]
        torch.testing.assert_close(a.cpu(),b);torch.testing.assert_close(ag.cpu(),bg)

@pytest.mark.parametrize('dtype', [torch.int32, torch.int64])
def test_gcd_lcm_exact_signed_extremes(dtype):
    x = torch.tensor([0, 1, -1, 24, -30, 2147483647, -2147483648, 0], dtype=dtype)
    if dtype == torch.int64: x[-2] = -(2**63); x[-3] = 2**61+3
    y = torch.tensor([0, 0, 0, -18, 42, 87, 0, -91], dtype=dtype)
    for fn in [torch.gcd, torch.lcm]:
        vk.reset_execution_stats();value=fn(x.to('vulkan'),y.to('vulkan'))
        assert vk.execution_stats()['cpu_fallback_calls'] == 0
        torch.testing.assert_close(value.cpu(),fn(x,y))

@pytest.mark.parametrize('dtype', [torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64])
def test_signed_bit_shifts_width_and_negative_counts(dtype):
    x = torch.tensor([0, 1, 3, 127], dtype=dtype)
    if dtype != torch.uint8: x[-2:] = torch.tensor([-3, -1],dtype=dtype)
    if dtype == torch.int64: x[1] = 2**60+3
    counts = torch.tensor([-1, 0, 1, 7, 8, 15, 16, 31, 32, 33, 63, 64, 127], dtype=torch.int64)
    for fn in [torch.bitwise_left_shift, torch.bitwise_right_shift]:
        a,b=x[:,None],counts[None,:]
        torch.testing.assert_close(fn(a.to('vulkan'),b.to('vulkan')).cpu(),fn(a,b))
        torch.testing.assert_close(fn(3,counts.to('vulkan')).cpu(),fn(3,counts))

@pytest.mark.parametrize('shape,hermitian', [((3, 3), False), ((5, 3), False), ((3, 5), False), ((2, 3, 3), False), ((3, 3), True), ((2, 3, 3), True)])
def test_pseudoinverse_and_matrix_rank_native(shape, hermitian):
    torch.manual_seed(709); x = torch.randn(shape)
    if hermitian: x = x + x.mT
    v=x.to('vulkan')
    for options in [{}, {'atol': .1}, {'rtol': .1}, {'atol': .02, 'rtol': .01}]:
        a=torch.linalg.pinv(v,hermitian=hermitian,**options); b=torch.linalg.pinv(x,hermitian=hermitian,**options)
        torch.testing.assert_close(a.cpu(),b)
        torch.testing.assert_close(torch.linalg.matrix_rank(v,hermitian=hermitian,**options).cpu(),torch.linalg.matrix_rank(x,hermitian=hermitian,**options))
    torch.testing.assert_close((v@torch.linalg.pinv(v)@v).cpu(),x,rtol=2e-5,atol=3e-6)

@pytest.mark.parametrize('name', ['exponential_', 'cauchy_', 'geometric_', 'log_normal_'])
def test_device_distributions_reproducibility_and_statistics(name):
    shape=(200000,);g=torch.Generator(device='vulkan').manual_seed(829)
    x=torch.empty(shape,device='vulkan'); getattr(x,name)(.5,generator=g) if name=='geometric_' else getattr(x,name)(generator=g)
    g.manual_seed(829);y=torch.empty(shape,device='vulkan');getattr(y,name)(.5,generator=g) if name=='geometric_' else getattr(y,name)(generator=g)
    torch.testing.assert_close(x.cpu(),y.cpu(),rtol=0,atol=0)
    values=x.cpu();assert torch.isfinite(values).all()
    if name=='cauchy_': assert abs(values.median().item())<.03
    elif name=='geometric_': assert (values>=1).all(); assert abs(values.mean().item()-2)<.03
    elif name=='exponential_': assert (values>=0).all();assert abs(values.mean().item()-1)<.03
    else: assert (values>0).all(); assert abs(values.log().mean().item()-1)<.03;assert abs(values.log().std().item()-2)<.03
    assert vk.execution_stats()['cpu_fallback_calls']==0


def test_tensor_normal_parameters_and_integer_bernoulli():
    mean=torch.arange(3000,dtype=torch.float32).reshape(1000,3).div(1000);std=torch.tensor([0.,.5,1.])
    vmean,vstd=mean.to('vulkan'),std.to('vulkan');g=torch.Generator(device='vulkan').manual_seed(97)
    result=torch.normal(vmean,vstd,generator=g).cpu();torch.testing.assert_close(result[:,0],mean[:,0]); assert abs((result[:,1]-mean[:,1]).std().item()-.5)<.06
    for dtype in [torch.bool,torch.int32,torch.int64]:
        out=torch.empty(2,3,dtype=dtype,device='vulkan');probs=torch.tensor([[0.,1.,0.],[1.,0.,1.]],device='vulkan');out.bernoulli_(probs,generator=g);torch.testing.assert_close(out.cpu(),probs.cpu().to(dtype))

@pytest.mark.parametrize('hermitian', [False, True])
@pytest.mark.parametrize('rtol', [1., .01])
def test_pseudoinverse_backward_and_forward_ad(hermitian, rtol):
    torch.manual_seed(711)
    x = torch.randn(5, 5)
    if hermitian: x = x + x.mT
    tangent = torch.randn_like(x)
    host = x.requires_grad_(); device = x.detach().to('vulkan').requires_grad_()
    cpu = torch.linalg.pinv(host, rtol=rtol, hermitian=hermitian)
    gpu = torch.linalg.pinv(device, rtol=rtol, hermitian=hermitian)
    torch.testing.assert_close(torch.autograd.grad(gpu.sum(), device)[0].cpu(), torch.autograd.grad(cpu.sum(), host)[0])
    results=[]
    for value, direction in [(x, tangent), (x.to('vulkan'), tangent.to('vulkan'))]:
        with torch.autograd.forward_ad.dual_level():
            dual=torch.autograd.forward_ad.make_dual(value,direction)
            result=torch.linalg.pinv(dual,rtol=rtol,hermitian=hermitian)
            results.append(torch.autograd.forward_ad.unpack_dual(result).tangent.cpu())
    torch.testing.assert_close(results[1],results[0])

@pytest.mark.parametrize('shape', [(5,3), (3,5), (2,5,3), (2,3,5)])
@pytest.mark.parametrize('driver', ['gels', 'gelsy', 'gelsd', 'gelss'])
@pytest.mark.parametrize('vector', [False, True])
def test_least_squares_driver_shapes_and_values(shape, driver, vector):
    torch.manual_seed(719)
    a=torch.randn(shape); b=torch.randn(shape[:-1] if vector else (*shape[:-1],2))
    actual=torch.linalg.lstsq(a.to('vulkan'),b.to('vulkan'),driver=driver)
    expected=torch.linalg.lstsq(a,b,driver=driver)
    for x,y in zip(actual,expected):torch.testing.assert_close(x.cpu(),y)

@pytest.mark.parametrize('upper', [False, True])
def test_incomplete_gamma_domains_and_parameters(upper):
    a=torch.tensor([0.,.0001,.01,.1,.5,1.,2.,3.,10.,19.,20.,21.,50.,100.,199.,200.,201.,500.,1000.,1e4,1e6,float('inf'),float('nan'),-1.])[:,None]
    x=torch.cat((torch.tensor([-1.,0.,1e-6,.01,.1,.5,1.,1.1,2.,3.,10.,50.,100.,500.,1000.,1e4,1e6,float('inf'),float('nan')]),torch.linspace(15,25,19)))[None,:]
    fn=torch.special.gammaincc if upper else torch.special.gammainc
    actual=fn(a.to('vulkan'),x.to('vulkan'));expected=fn(a,x)
    torch.testing.assert_close(actual.cpu(),expected,equal_nan=True)

@pytest.mark.parametrize('batched', [False, True])
@pytest.mark.parametrize('kernel,dilation,padding,stride', [((2,3),(1,1),(0,0),(1,1)),((3,2),(2,1),(1,2),(2,3)),((1,1),(1,1),(0,0),(3,2))])
def test_image_columns_forward_fold_and_gradients(batched,kernel,dilation,padding,stride):
    torch.manual_seed(727)
    x=torch.randn((2,3,10,11) if batched else (3,10,11)).transpose(-1,-2)
    x=x[...,1:,:].detach().requires_grad_();v=x.detach().to('vulkan').requires_grad_()
    options=dict(kernel_size=kernel,dilation=dilation,padding=padding,stride=stride)
    a=torch.nn.functional.unfold(v,**options);b=torch.nn.functional.unfold(x,**options)
    torch.testing.assert_close(a.cpu(),b)
    f=torch.nn.functional.fold(a,x.shape[-2:],**options);g=torch.nn.functional.fold(b,x.shape[-2:],**options)
    torch.testing.assert_close(f.cpu(),g)
    torch.testing.assert_close(torch.autograd.grad(f.square().sum(),v)[0].cpu(),torch.autograd.grad(g.square().sum(),x)[0])

@pytest.mark.parametrize('shape', [(0,0),(1,1),(4,4),(2,4,4)])
def test_matrix_functions_values_and_gradients(shape):
    torch.manual_seed(733)
    source=torch.randn(shape)*.2
    for fn,x in [(torch.matrix_exp,source),(torch.linalg.matrix_sqrth,source@source.mT+torch.eye(shape[-1])*.5)]:
        host=x.detach().requires_grad_();dev=x.to('vulkan').requires_grad_()
        a=fn(dev);b=fn(host);torch.testing.assert_close(a.cpu(),b)
        torch.testing.assert_close(torch.autograd.grad(a.sum(),dev)[0].cpu(),torch.autograd.grad(b.sum(),host)[0])

@pytest.mark.parametrize('upper,transpose,unit', [(u,t,d) for u in [False,True] for t in [False,True] for d in [False,True]])
def test_legacy_triangular_solve_options(upper,transpose,unit):
    torch.manual_seed(739);a=torch.randn(2,4,4)+torch.eye(4)*4;b=torch.randn(2,4,3)
    actual=torch.triangular_solve(b.to('vulkan'),a.to('vulkan'),upper=upper,transpose=transpose,unitriangular=unit)
    expected=torch.triangular_solve(b,a,upper=upper,transpose=transpose,unitriangular=unit)
    for x,y in zip(actual,expected):torch.testing.assert_close(x.cpu(),y)

@pytest.mark.parametrize('p',[0.,.5,1.,1.5,2.,3.,float('inf')])
def test_pairwise_distances_and_gradients(p):
    torch.manual_seed(743);x=torch.randn(2,5,4,requires_grad=True);y=torch.randn(1,3,4,requires_grad=True)
    a=x.detach().to('vulkan').requires_grad_();b=y.detach().to('vulkan').requires_grad_()
    actual=torch.cdist(a,b,p=p);expected=torch.cdist(x,y,p=p)
    torch.testing.assert_close(actual.cpu(),expected)
    actual_grad=torch.autograd.grad(actual.sum(),(a,b));expected_grad=torch.autograd.grad(expected.sum(),(x,y))
    for a,b in zip(actual_grad,expected_grad):torch.testing.assert_close(a.cpu(),b)

@pytest.mark.parametrize('base',[2.,10.,.5])
def test_logspace_scalar_and_tensor_endpoints(base):
    for start,end,n in [(-2.,3.,0),(-2.,3.,1),(-2.,3.,31),(3.,-2.,32)]:
        expected=torch.logspace(start,end,n,base=base)
        for a,b in [(start,end),(torch.tensor(start,device='vulkan'),torch.tensor(end,device='vulkan'))]:
            torch.testing.assert_close(torch.logspace(a,b,n,base=base,device='vulkan').cpu(),expected)

@pytest.mark.parametrize('dtype',[torch.float32,torch.int64,torch.bool])
def test_heaviside_broadcast_and_special_values(dtype):
    x=torch.tensor([-1,0,1],dtype=dtype).reshape(3,1);v=torch.tensor([0,1,1],dtype=dtype)
    torch.testing.assert_close(torch.heaviside(x.to('vulkan'),v.to('vulkan')).cpu(),torch.heaviside(x,v))

@pytest.mark.parametrize('axis,size,step', [(0,2,1),(1,3,2),(2,2,3),(1,0,1)])
def test_tensor_unfold_backward_overlapping_windows(axis,size,step):
    torch.manual_seed(751);x=torch.randn(4,5,6,requires_grad=True);v=x.detach().to('vulkan').requires_grad_()
    a=v.unfold(axis,size,step);b=x.unfold(axis,size,step)
    torch.testing.assert_close(a.cpu(),b)
    torch.testing.assert_close(torch.autograd.grad(a.square().sum(),v)[0].cpu(),torch.autograd.grad(b.square().sum(),x)[0])

@pytest.mark.parametrize('ignore_nan',[False,True])
@pytest.mark.parametrize('interpolation',['linear','lower','higher','midpoint','nearest'])
def test_quantiles_shapes_nan_and_gradients(ignore_nan,interpolation):
    torch.manual_seed(757)
    fn=torch.nanquantile if ignore_nan else torch.quantile
    source=torch.randn(3,4,5).transpose(0,2);source[1,2,0]=float('nan');source[2,:,1]=float('nan')
    for dim,keep in [(None,False),(0,False),(1,True),(-1,True)]:
        for q in [torch.tensor(.37),torch.tensor([0.,.1,.5,.9,1.]),torch.empty(0)]:
            host=source.detach().requires_grad_();dev=source.to('vulkan').requires_grad_()
            a=fn(dev,q.to('vulkan'),dim=dim,keepdim=keep,interpolation=interpolation)
            b=fn(host,q,dim=dim,keepdim=keep,interpolation=interpolation)
            torch.testing.assert_close(a.cpu(),b,equal_nan=True)
            ga=torch.autograd.grad(a.sum(),dev)[0];gb=torch.autograd.grad(b.sum(),host)[0]
            # Equivalent NaNs can receive different tie indices in CPU selection,
            # CUDA sort and Vulkan sort. Preserve finite gradients and the total
            # gradient routed to NaNs within each reduction, independently of ties.
            nan=source.isnan();ga=ga.cpu()
            torch.testing.assert_close(ga.masked_fill(nan,0),gb.masked_fill(nan,0),equal_nan=True)
            torch.testing.assert_close(ga.masked_fill(~nan,0).sum(dim),gb.masked_fill(~nan,0).sum(dim),equal_nan=True)

@pytest.mark.parametrize('mode',[0,1,2,3,4])
def test_quantile_rank_precision_without_fp64(mode):
    torch.manual_seed(761)
    qbits=torch.randint(0,0x3f800001,(10000,),dtype=torch.int32)
    q=torch.cat((qbits.view(torch.float32),torch.rand(10000),torch.tensor([0.,.5,1.])))
    n=torch.randint(1,2**32,(q.numel(),),dtype=torch.int64)
    lower,upper,weight=torch.ops.torch_vulkan._quantile_ranks(q.to('vulkan'),n.to('vulkan'),mode)
    rank=q.double()*(n-1).double();lo=rank.floor().long();hi=rank.ceil().long()
    weights=(rank-lo).float()
    if mode==1:hi=lo;weights.zero_()
    elif mode==2:lo=hi;weights.zero_()
    elif mode==3:weights.fill_(.5)
    elif mode==4:lo=rank.round().long();hi=lo;weights.zero_()
    torch.testing.assert_close(lower.cpu(),lo);torch.testing.assert_close(upper.cpu(),hi)
    torch.testing.assert_close(weight.cpu(),weights,rtol=0,atol=0)


def test_quantile_probability_gradient_and_scalar_precision():
    torch.manual_seed(769);x=torch.randn(2,101);q=torch.tensor([.01,.15,.5,.99],requires_grad=True)
    vq=q.detach().to('vulkan').requires_grad_()
    actual=torch.quantile(x.to('vulkan'),vq,dim=1);expected=torch.quantile(x,q,dim=1)
    torch.testing.assert_close(actual.cpu(),expected)
    torch.testing.assert_close(torch.autograd.grad(actual.sum(),vq)[0].cpu(),torch.autograd.grad(expected.sum(),q)[0])
    for scalar in [0.,.01,.15,.99,1.]:torch.testing.assert_close(torch.quantile(x.to('vulkan'),scalar,dim=1).cpu(),torch.quantile(x,scalar,dim=1))

@pytest.mark.parametrize('shape,axis,size,step', [((),0,1,1),((5,5,5),2,3,2)])
def test_unfold_scalar_and_variant_backward(shape,axis,size,step):
    x=torch.randn(shape,requires_grad=True);v=x.detach().to('vulkan').requires_grad_()
    a=v.unfold(axis,size,step);b=x.unfold(axis,size,step)
    torch.testing.assert_close(torch.autograd.grad(a.sum(),v)[0].cpu(),torch.autograd.grad(b.sum(),x)[0])

@pytest.mark.parametrize('reduction',['sum','mean','max','min','prod'])
@pytest.mark.parametrize('use_offsets',[False,True])
@pytest.mark.parametrize('axis',[0,1])
@pytest.mark.parametrize('initial',[None,0.,.3])
def test_segment_reductions_forward_backward(reduction,use_offsets,axis,initial):
    torch.manual_seed(773)
    lengths=torch.tensor([2,0,4]) if axis==0 else torch.tensor([[2,0,4],[1,3,2]])
    data=torch.randn(6,3) if axis==0 else torch.randn(2,6,3)
    data[...,0]=0
    if use_offsets:
        shape=list(lengths.shape);shape[-1]=1
        indices=torch.cat((torch.zeros(shape,dtype=lengths.dtype),lengths.cumsum(-1)),dim=-1);name='offsets'
    else:indices=lengths;name='lengths'
    host=data.detach().requires_grad_();dev=data.to('vulkan').requires_grad_()
    a=torch.segment_reduce(dev,reduction,axis=axis,initial=initial,**{name:indices.to('vulkan')})
    b=torch.segment_reduce(host,reduction,axis=axis,initial=initial,**{name:indices})
    torch.testing.assert_close(a.cpu(),b,equal_nan=True)
    direction=torch.randn_like(b)
    ga=torch.autograd.grad(a,dev,direction.to('vulkan'))[0];gb=torch.autograd.grad(b,host,direction)[0]
    torch.testing.assert_close(ga.cpu(),gb,equal_nan=True)

@pytest.mark.parametrize('reduction',['sum','mean','max','min','prod'])
def test_segment_reductions_nan_and_zero_products(reduction):
    x=torch.tensor([0.,2.,0.,0.,float('nan'),2.,2.,float('nan')],requires_grad=True)
    v=x.detach().to('vulkan').requires_grad_();lengths=torch.tensor([2,2,2,2])
    a=torch.segment_reduce(v,reduction,lengths=lengths.to('vulkan'))
    b=torch.segment_reduce(x,reduction,lengths=lengths)
    torch.testing.assert_close(a.cpu(),b,equal_nan=True)
    torch.testing.assert_close(torch.autograd.grad(a.sum(),v)[0].cpu(),torch.autograd.grad(b.sum(),x)[0],equal_nan=True)

@pytest.mark.parametrize('rank',[0,1,2])
@pytest.mark.parametrize('power',[1,2])
@pytest.mark.parametrize('reduction',['none','mean','sum'])
@pytest.mark.parametrize('weighted',[False,True])
def test_multi_margin_loss_values_and_gradients(rank,power,reduction,weighted):
    torch.manual_seed(787);classes=1 if rank==0 else 7;rows=3
    x=torch.randn(4)[2] if rank==0 else torch.randn(classes*2+1)[1::2] if rank==1 else torch.randn(rows+1,classes*2)[1:,1::2]
    target=torch.tensor(0) if rank==0 else torch.tensor(3) if rank==1 else torch.tensor([0,3,6])
    weight=torch.rand(classes*2)[::2] if weighted else None
    host=x.detach().requires_grad_();dev=x.detach().to('vulkan').requires_grad_()
    options=dict(p=power,margin=.7,reduction=reduction)
    a=torch.nn.functional.multi_margin_loss(dev,target.to('vulkan'),weight=weight.to('vulkan') if weighted else None,**options)
    b=torch.nn.functional.multi_margin_loss(host,target,weight=weight,**options)
    torch.testing.assert_close(a.cpu(),b)
    direction=torch.randn_like(b)
    torch.testing.assert_close(torch.autograd.grad(a,dev,direction.to('vulkan'))[0].cpu(),torch.autograd.grad(b,host,direction)[0])

@pytest.mark.parametrize('rank',[0,1,2])
@pytest.mark.parametrize('reduction',['none','mean','sum'])
def test_multilabel_margin_loss_values_masks_and_gradients(rank,reduction):
    torch.manual_seed(797);classes=1 if rank==0 else 7;rows=3
    x=torch.randn(4)[2] if rank==0 else torch.randn(classes*2+1)[1::2] if rank==1 else torch.randn(rows+1,classes*2)[1:,1::2]
    target=torch.tensor(0) if rank==0 else torch.tensor([1,3,-1,-1,-1,-1,-1]) if rank==1 else torch.tensor([[1,3,-1,-1,-1,-1,-1],[6,-1,-1,-1,-1,-1,-1],[0,1,4,-1,-1,-1,-1]])
    host=x.detach().requires_grad_();dev=x.detach().to('vulkan').requires_grad_()
    a=torch.nn.functional.multilabel_margin_loss(dev,target.to('vulkan'),reduction=reduction);b=torch.nn.functional.multilabel_margin_loss(host,target,reduction=reduction)
    torch.testing.assert_close(a.cpu(),b)
    direction=torch.randn_like(b)
    torch.testing.assert_close(torch.autograd.grad(a,dev,direction.to('vulkan'))[0].cpu(),torch.autograd.grad(b,host,direction)[0])
    ag=torch.ops.aten.multilabel_margin_loss_forward(dev.detach(),target.to('vulkan'),0)
    bg=torch.ops.aten.multilabel_margin_loss_forward(host.detach(),target,0)
    for u,v in zip(ag,bg):torch.testing.assert_close(u.cpu(),v)

@pytest.mark.parametrize('multilabel',[False,True])
def test_margin_loss_shape_validation_precedes_target_dtype(multilabel):
    x=torch.empty(5,0,device='vulkan')
    target=torch.empty(5,4,device='vulkan') if multilabel else torch.empty(5,device='vulkan')
    fn=torch.nn.functional.multilabel_margin_loss if multilabel else torch.nn.functional.multi_margin_loss
    with pytest.raises(RuntimeError,match='Expected non-empty vector or matrix'):
        fn(x,target)

@pytest.mark.parametrize('dtype',[torch.float32,torch.int64,torch.int32,torch.bool,torch.uint8,torch.int8,torch.int16])
@pytest.mark.parametrize('shape',[(),(4097,),(3,4,5)])
def test_tensor_hash_types_axes_and_unsigned_output(dtype,shape):
    torch.manual_seed(809)
    if dtype==torch.float32:x=torch.randn(shape)
    elif dtype==torch.bool:x=torch.randint(0,2,shape).bool()
    else:x=torch.randint(0,91,shape,dtype=dtype)
    if len(shape)>1:x=x.transpose(0,2)
    axes=[None] if len(shape)<2 else [None,0,1,-1,(0,2)]
    v=x.to('vulkan')
    for dim in axes:
        for keep in [False,True]:
            options={'keepdim':keep}
            if dim is not None:options['dim']=dim
            actual=torch.hash_tensor(v,**options);expected=torch.hash_tensor(x,**options)
            assert actual.dtype==torch.uint64
            assert actual.shape==expected.shape
            assert torch.equal(actual.cpu(),expected)
            clone=actual.clone();assert torch.equal(clone.cpu(),expected)
            out=torch.empty(expected.shape,dtype=torch.uint64,device='vulkan')
            torch.hash_tensor(v,out=out,**options);assert torch.equal(out.cpu(),expected)


def test_tensor_hash_preserves_float_bits_and_signed_extremes():
    torch.manual_seed(811)
    bits=torch.randint(-(2**31),2**31-1,(20000,),dtype=torch.int32)
    x=bits.view(torch.float32)
    x=torch.cat((x,torch.tensor([0.,-0.,float('inf'),-float('inf'),float('nan')]),torch.tensor([1,0x7fffff,-2147483647],dtype=torch.int32).view(torch.float32)))
    integers=torch.tensor([-(2**63),2**63-1,-1,0,2**60+3,-(2**60+5)],dtype=torch.int64)
    for value in [x,integers]:assert torch.equal(torch.hash_tensor(value.to('vulkan')).cpu(),torch.hash_tensor(value))

@pytest.mark.parametrize('dtype',[torch.uint8,torch.int8,torch.int16,torch.int32,torch.int64])
@pytest.mark.parametrize('weighted',[False,True])
def test_bincount_indices_weights_and_minimum_length(dtype,weighted):
    torch.manual_seed(821);x=torch.randint(0,31,(401,),dtype=dtype)[1::2]
    w=torch.randn(401)[1::2] if weighted else None
    for minimum in [0,50]:
        a=torch.bincount(x.to('vulkan'),weights=w.to('vulkan') if weighted else None,minlength=minimum)
        b=torch.bincount(x,weights=w,minlength=minimum)
        torch.testing.assert_close(a.cpu(),b)


def test_bincount_empty_and_invalid_indices():
    for n in [0,7]:
        actual=torch.bincount(torch.empty(0,dtype=torch.int64,device='vulkan'),weights=torch.empty(0,device='vulkan'),minlength=n)
        torch.testing.assert_close(actual.cpu(),torch.zeros(n,dtype=torch.int64))
    with pytest.raises(RuntimeError,match='non-negative'):
        torch.bincount(torch.tensor([0,-1,2],device='vulkan'))
    with pytest.raises(RuntimeError,match='minlength'):
        torch.bincount(torch.tensor([0,1],device='vulkan'),minlength=-1)

@pytest.mark.parametrize('value',[-1,2**64-1,3.5])
def test_unsigned64_fill_preserves_scalar_conversion(value):
    cpu=torch.empty(5,dtype=torch.uint64).fill_(value)
    gpu=torch.empty(5,dtype=torch.uint64,device='vulkan').fill_(value)
    assert torch.equal(gpu.cpu(),cpu)


def test_unsigned64_comparisons_cover_sign_boundary():
    values=[0,1,2**63-1,2**63,2**64-1]
    x=torch.tensor(values,dtype=torch.uint64,device='vulkan')
    for name,fn in [('eq',lambda a,b:a==b),('ne',lambda a,b:a!=b),('lt',lambda a,b:a<b),('le',lambda a,b:a<=b),('gt',lambda a,b:a>b),('ge',lambda a,b:a>=b)]:
        expected=torch.tensor([[fn(a,b) for b in values] for a in values])
        assert torch.equal(getattr(torch,name)(x[:,None],x[None,:]).cpu(),expected)


def test_unsigned64_bounded_generator_reproducibility():
    generator=torch.Generator(device='vulkan').manual_seed(823)
    x=torch.empty(10000,dtype=torch.uint64,device='vulkan').random_(5,100,generator=generator)
    generator.manual_seed(823)
    y=torch.empty_like(x).random_(5,100,generator=generator)
    assert torch.equal(x.cpu(),y.cpu())
    values=x.cpu().tolist();assert all(5<=value<100 for value in values)
    assert abs(sum(values)/len(values)-52)<1


def test_unsigned64_signed_storage_casts_preserve_all_bits():
    signed=torch.tensor([-(2**63),-1,0,1,2**63-1],dtype=torch.int64)
    unsigned=signed.to('vulkan').to(torch.uint64)
    assert torch.equal(unsigned.cpu(),signed.to(torch.uint64))
    assert torch.equal(unsigned.to(torch.int64).cpu(),signed)

@pytest.mark.parametrize('shape', [(), (31,), (3,4,5), (3,0,5)])
@pytest.mark.parametrize('weighted,density',[(False,False),(True,False),(False,True),(True,True)])
def test_histogram_native_edges_weights_and_density(shape,weighted,density):
    torch.manual_seed(741)
    x=torch.randn(shape);weight=torch.randn(shape) if weighted else None
    for bins in [1,7,torch.tensor([-3.,-1.,-.25,0.,.25,1.,3.])]:
        kw={'weight':weight,'density':density}
        expected=torch.histogram(x,bins,**kw)
        actual=torch.histogram(x.to('vulkan'),bins.to('vulkan') if isinstance(bins,torch.Tensor) else bins,weight=weight.to('vulkan') if weighted else None,density=density)
        for a,b in zip(actual,expected):torch.testing.assert_close(a.cpu(),b,equal_nan=True)

@pytest.mark.parametrize('dimensions',[0,1,2,4])
@pytest.mark.parametrize('weighted,density',[(False,False),(True,False),(False,True),(True,True)])
def test_histogramdd_native_mixed_bins_and_strides(dimensions,weighted,density):
    torch.manual_seed(712)
    x=torch.randn(dimensions,2,19).permute(1,2,0)
    weight=torch.randn(2,19) if weighted else None
    counts=[2+d for d in range(dimensions)]
    for bins in [counts,[torch.linspace(-2,2,n+1) for n in counts]]:
        expected=torch.histogramdd(x,bins,weight=weight,density=density)
        actual=torch.histogramdd(x.to('vulkan'),[b.to('vulkan') for b in bins] if bins and isinstance(bins[0],torch.Tensor) else bins,weight=weight.to('vulkan') if weighted else None,density=density)
        torch.testing.assert_close(actual.hist.cpu(),expected.hist,equal_nan=True)
        for a,b in zip(actual.bin_edges,expected.bin_edges):torch.testing.assert_close(a.cpu(),b)

@pytest.mark.parametrize('shape', [(), (101,), (3,7), (0,5)])
@pytest.mark.parametrize('bounds',[(0.,0.),(-2.,2.),(-100.,0.)])
def test_histc_native_linear_bin_selection(shape,bounds):
    torch.manual_seed(772)
    x=torch.randn(shape)
    for n in [1,3,10,100]:
        expected=torch.histc(x,n,*bounds)
        actual=torch.histc(x.to('vulkan'),n,*bounds)
        torch.testing.assert_close(actual.cpu(),expected)


def test_histogram_native_boundaries_nan_and_degenerate_range():
    for x in [torch.tensor([-2.,-1.,0.,1.,2.,float('nan')]),torch.ones(17),torch.empty(0)]:
        v=x.to('vulkan')
        for n in [1,4]:
            if not torch.isnan(x).any():
                for a,b in zip(torch.histogram(v,n),torch.histogram(x,n)):torch.testing.assert_close(a.cpu(),b)
            for a,b in zip(torch.histogram(v,n,range=(-1.,1.)),torch.histogram(x,n,range=(-1.,1.))):torch.testing.assert_close(a.cpu(),b)
    with pytest.raises(RuntimeError,match='not finite'):torch.histogram(torch.tensor([float('nan')],device='vulkan'),4)
    with pytest.raises(RuntimeError,match='innermost dimension'):torch.histogramdd(torch.ones(5,6,device='vulkan'),[1]*5)

@pytest.mark.parametrize('upper',[False,True])
@pytest.mark.parametrize('dtype',[torch.int32,torch.int64])
@pytest.mark.parametrize('shape',[(0,5),(5,0),(1,1),(3,7),(7,3),(37,61),(65537,1),(1,65537)])
def test_triangular_indices_native_shapes_offsets_and_storage(upper,dtype,shape):
    op=torch.triu_indices if upper else torch.tril_indices
    for offset in [-2**63,-100,-3,-1,0,1,3,100,2**63-1]:
        # Pinned CPU size arithmetic overflows at the signed offset limits.
        # Use the integer inequality itself as an independent oracle there.
        if offset in [-2**63,2**63-1]:
            pairs=[(r,c) for r in range(shape[0]) for c in range(shape[1]) if (c>=r+offset if upper else c<=r+offset)]
            expected=torch.tensor(pairs,dtype=dtype).reshape(-1,2).T.contiguous()
        else:
            expected=op(*shape,offset,dtype=dtype)
        vk.reset_execution_stats()
        actual=op(*shape,offset,dtype=dtype,device='vulkan')
        resident()
        assert torch.equal(actual.cpu(),expected)
        target=torch.empty((2,actual.size(1)*2),dtype=dtype,device='vulkan')[:,::2]
        assert (torch.ops.aten.triu_indices.out if upper else torch.ops.aten.tril_indices.out)(*shape,offset,out=target) is target
        assert torch.equal(target.cpu(),expected)

@pytest.mark.parametrize('p',[0.,.5,1.,2.,3.,float('inf')])
@pytest.mark.parametrize('shape',[(0,3),(1,3),(7,0),(6,4)])
def test_pdist_native_forward_backward(p,shape):
    torch.manual_seed(873)
    x=torch.randn(shape,requires_grad=True);v=x.detach().to('vulkan').requires_grad_()
    expected=torch.pdist(x,p);actual=torch.pdist(v,p)
    torch.testing.assert_close(actual.cpu(),expected)
    g=torch.randn_like(expected)
    # Pinned CPU pdist backward segfaults for empty p<1 inputs; the empty
    # derivative is independently zero, including tensors with no features.
    if expected.numel() and x.numel():
        expected.backward(g)
        gradient=x.grad
    else:
        gradient=torch.zeros_like(x)
    actual.backward(g.to('vulkan'))
    torch.testing.assert_close(v.grad.cpu(),gradient)

@pytest.mark.parametrize('causal',[False,True])
@pytest.mark.parametrize('mask_type',['none','2d','4d','all_masked'])
def test_cpu_flash_aten_contract_uses_vulkan_attention(causal,mask_type):
    torch.manual_seed(967)
    q=torch.randn(2,3,5,4,requires_grad=True);k=torch.randn(2,3,7,4,requires_grad=True);v=torch.randn(2,3,7,4,requires_grad=True)
    mask=None if mask_type=='none' else torch.randn(5,7) if mask_type=='2d' else torch.randn(1,3,5,7) if mask_type=='4d' else torch.full((5,7),float('-inf'))
    op=torch.ops.aten._scaled_dot_product_flash_attention_for_cpu
    expected,lse=op(q,k,v,is_causal=causal,attn_mask=mask,scale=.3)
    inputs=[t.detach().to('vulkan').requires_grad_() for t in (q,k,v)]
    vk.reset_execution_stats()
    actual,lsev=op(*inputs,is_causal=causal,attn_mask=None if mask is None else mask.to('vulkan'),scale=.3)
    # Uploads and scalar parameter setup do not download tensor data.
    resident()
    torch.testing.assert_close(actual.cpu(),expected)
    torch.testing.assert_close(lsev.cpu(),lse)
    grad=torch.randn_like(expected);expected.backward(grad);actual.backward(grad.to('vulkan'))
    for a,b in zip(inputs,[q,k,v]):torch.testing.assert_close(a.grad.cpu(),b.grad)

@pytest.mark.parametrize('dimensions',[2,3])
@pytest.mark.parametrize('batched',[False,True])
@pytest.mark.parametrize('values',['random','ties','nan','negative_infinity'])
def test_fractional_pool_native_random_intervals_indices_and_gradient(dimensions,batched,values):
    torch.manual_seed(613)
    shape=([2,3] if batched else [3])+([8,11] if dimensions==2 else [7,8,11])
    x=torch.randn(shape)
    if values=='ties':x.round_()
    if values=='nan':x.reshape(-1)[::31]=float('nan')
    if values=='negative_infinity':x.fill_(float('-inf'))
    x.requires_grad_();v=x.detach().to('vulkan').requires_grad_()
    kernel=(3,2) if dimensions==2 else (2,3,2)
    samples=torch.rand(3,3,dimensions);sv=samples.to('vulkan')
    output=(4,5) if dimensions==2 else (3,3,5)
    op=torch.ops.aten.fractional_max_pool2d if dimensions==2 else torch.ops.aten.fractional_max_pool3d
    expected,indices=op(x,kernel,output,samples)
    vk.reset_execution_stats();actual,indicesv=op(v,kernel,output,sv);resident()
    torch.testing.assert_close(actual.cpu(),expected,equal_nan=True)
    assert torch.equal(indicesv.cpu(),indices)
    g=torch.randn_like(expected);expected.backward(g);actual.backward(g.to('vulkan'))
    torch.testing.assert_close(v.grad.cpu(),x.grad)

@pytest.mark.parametrize('layout',['contiguous','permute','stepped'])
@pytest.mark.parametrize('rank',[9,16,32,64])
def test_general_iteration_native_rank_stride_and_broadcast(rank,layout):
    shape=[1]*(rank-4)+[2,3,2,6]
    torch.manual_seed(226)
    x=torch.randn(shape);v=x.to('vulkan')
    if layout=='permute':
        dims=list(range(rank));dims[-4:]=[rank-1,rank-3,rank-4,rank-2]
        x=x.permute(dims);v=v.permute(dims)
    if layout=='stepped':x=x[...,1::2];v=v[...,1::2]
    rhs=torch.randn(x.size(-1));rv=rhs.to('vulkan')
    vk.reset_execution_stats()
    actual=[v.sigmoid(),v.abs(),v.sgn(),v/3.,v-rv,v/rv,torch.nn.functional.leaky_relu(v,.13),torch.nn.functional.logsigmoid(v),v.mean()]
    resident()
    expected=[x.sigmoid(),x.abs(),x.sgn(),x/3.,x-rhs,x/rhs,torch.nn.functional.leaky_relu(x,.13),torch.nn.functional.logsigmoid(x),x.mean()]
    for a,b in zip(actual,expected):torch.testing.assert_close(a.cpu(),b)
    vk.reset_execution_stats();v.add_(rv,alpha=.25);v.sub_(rv,alpha=-.3);v.mul_(rv);v.div_(rv);v.add_(.15);resident()
    x.add_(rhs,alpha=.25);x.sub_(rhs,alpha=-.3);x.mul_(rhs);x.div_(rhs);x.add_(.15)
    torch.testing.assert_close(v.cpu(),x)


def test_general_iteration_native_uncoalesced_axes_and_autograd():
    torch.manual_seed(267)
    x=torch.randn([2]*9).permute(8,6,4,2,0,7,5,3,1).detach().requires_grad_()
    v=x.detach().contiguous().to('vulkan').permute(8,6,4,2,0,7,5,3,1).detach().requires_grad_()
    # Match logical values while preserving a noncontiguous rank-nine layout.
    expected=v.detach().cpu().requires_grad_()
    vk.reset_execution_stats()
    result=(v.sigmoid()+torch.nn.functional.logsigmoid(v)+torch.nn.functional.leaky_relu(v,.2)).mean()
    resident()
    reference=(expected.sigmoid()+torch.nn.functional.logsigmoid(expected)+torch.nn.functional.leaky_relu(expected,.2)).mean()
    result.backward();reference.backward()
    torch.testing.assert_close(v.grad.cpu(),expected.grad)


def test_general_iteration_native_overlapping_fill_and_lazy_negative_view():
    cpu=torch.arange(64,dtype=torch.float32);gpu=cpu.to('vulkan')
    a=cpu.as_strided((8,8),(1,1),3);v=gpu.as_strided((8,8),(1,1),3)
    vk.reset_execution_stats();v.fill_(1.75);resident();a.fill_(1.75)
    torch.testing.assert_close(gpu.cpu(),cpu)
    shape=[1]*8+[4];x=torch.tensor([-2.,-0.,1.,3.]).reshape(shape);v=x.to('vulkan')
    neg=torch._neg_view(v);expected=torch._neg_view(x)
    vk.reset_execution_stats();actual=neg.sigmoid();neg.add_(.25);resident()
    torch.testing.assert_close(actual.cpu(),expected.sigmoid())
    expected.add_(.25);torch.testing.assert_close(v.cpu(),x)

@pytest.mark.parametrize('replacement',[False,True])
@pytest.mark.parametrize('batched',[False,True])
def test_multinomial_native_generator_and_zero_weights(replacement,batched):
    weights=torch.tensor([[0.,1.,3.,0.,2.],[2.,0.,0.,1.,4.]])
    if not batched:weights=weights[0]
    x=weights.to('vulkan');g=torch.Generator(device='vulkan').manual_seed(611)
    samples=4 if not replacement else 20000
    vk.reset_execution_stats();a=torch.multinomial(x,samples,replacement,generator=g)
    assert vk.execution_stats()['cpu_fallback_calls']==0
    g.manual_seed(611);b=torch.multinomial(x,samples,replacement,generator=g)
    assert torch.equal(a.cpu(),b.cpu())
    if not replacement:
        rows=a.cpu().reshape(-1,samples)
        assert all(len(set(row.tolist()))==samples for row in rows)
    else:
        rows=a.cpu().reshape(-1,samples);expected=weights.reshape(-1,5)
        for row,prob in zip(rows,expected):
            counts=torch.bincount(row,minlength=5).float()/samples
            torch.testing.assert_close(counts,prob/prob.sum(),atol=.015,rtol=0)
            assert torch.equal(counts[prob==0],torch.zeros_like(counts[prob==0]))


def test_multinomial_native_invalid_probabilities_and_out():
    for values in [[0.,0.,0.],[1.,-1.,1.],[1.,float('nan'),1.],[1.,float('inf'),1.]]:
        for replacement in [False,True]:
            with pytest.raises(RuntimeError,match='probability|probabilities'):
                torch.multinomial(torch.tensor(values,device='vulkan'),2,replacement)
    x=torch.tensor([0.,0.,1.,0.],device='vulkan');out=torch.empty(20,dtype=torch.long,device='vulkan')[::2]
    assert torch.multinomial(x,10,True,out=out) is out
    assert out.cpu().tolist()==[2]*10
    with pytest.raises(RuntimeError,match='Long tensor out'):torch.multinomial(x,2,out=torch.empty(2,device='vulkan'))

@pytest.mark.parametrize('target_format',['padded','concatenated'])
@pytest.mark.parametrize('tensor_lengths',[False,True])
@pytest.mark.parametrize('reduction',['none','mean','sum'])
@pytest.mark.parametrize('zero_infinity',[False,True])
def test_ctc_native_lengths_repeats_zero_lengths_and_gradients(target_format,tensor_lengths,reduction,zero_infinity):
    torch.manual_seed(633)
    lengths=[0,7,13];target_lengths=[0,3,5]
    targets=torch.tensor([[1,1,2,2,1],[1,1,2,2,1],[1,1,2,2,1]],dtype=torch.int64)
    if target_format=='concatenated':targets=torch.cat([row[:n] for row,n in zip(targets,target_lengths)])
    data=torch.randn(15,8,3).log_softmax(1).permute(0,2,1)
    x=data.detach().requires_grad_();v=data.to('vulkan').detach().requires_grad_()
    il=torch.tensor(lengths) if tensor_lengths else lengths;tl=torch.tensor(target_lengths) if tensor_lengths else target_lengths
    op=torch.nn.functional.ctc_loss
    expected=op(x,targets,il,tl,reduction=reduction,zero_infinity=zero_infinity)
    vk.reset_execution_stats()
    actual=op(v,targets.to('vulkan'),il.to('vulkan') if tensor_lengths else il,tl.to('vulkan') if tensor_lengths else tl,reduction=reduction,zero_infinity=zero_infinity)
    assert vk.execution_stats()['cpu_fallback_calls']==0
    torch.testing.assert_close(actual.cpu(),expected,equal_nan=True)
    grad=torch.randn_like(expected);expected.backward(grad);actual.backward(grad.to('vulkan'))
    torch.testing.assert_close(v.grad.cpu(),x.grad,equal_nan=True)

@pytest.mark.parametrize('dtype',[torch.int32,torch.int64])
@pytest.mark.parametrize('blank',[0,3])
def test_ctc_native_long_sequences_cpu_targets_and_alphas(dtype,blank):
    torch.manual_seed(642)
    targets=torch.randint(0,6,(2,17),dtype=dtype);targets=targets+(targets>=blank).to(dtype)
    x=torch.randn(81,2,7).log_softmax(-1).detach().requires_grad_();v=x.detach().to('vulkan').requires_grad_()
    expected,alpha=torch.ops.aten._ctc_loss(x,targets,[81,73],[17,11],blank,False)
    actual,alphav=torch.ops.aten._ctc_loss(v,targets,[81,73],[17,11],blank,False)
    torch.testing.assert_close(actual.cpu(),expected)
    for n,(length,labels) in enumerate([(81,17),(73,11)]):torch.testing.assert_close(alphav[n,:length,:2*labels+1].cpu(),alpha[n,:length,:2*labels+1])
    grad=torch.tensor([.2,-.3]);expected.backward(grad);actual.backward(grad.to('vulkan'))
    torch.testing.assert_close(v.grad.cpu(),x.grad)

@pytest.mark.parametrize('dtype',[torch.bool,torch.uint8,torch.int8,torch.int16,torch.int32,torch.int64])
def test_integer_sigmoid_and_powers_remain_native(dtype):
    x=torch.tensor([[-3,-1,0,1,3],[7,8,9,10,11]],dtype=torch.int64).to(dtype)
    v=x.to('vulkan').t()
    for exponent in [0,1,2,5,63]:
        vk.reset_execution_stats();actual=torch.pow(v,exponent);resident()
        assert torch.equal(actual.cpu(),torch.pow(x.t(),exponent))
    vk.reset_execution_stats();actual=v.sigmoid();resident()
    torch.testing.assert_close(actual.cpu(),x.t().sigmoid())
    if dtype!=torch.bool:
        with pytest.raises(RuntimeError,match='negative integer powers'):torch.pow(v,-1)


def test_noncanonical_bool_sigmoid_and_square():
    raw=torch.tensor([0,1,2,255,128,0,7,64],dtype=torch.uint8)
    v=raw.to('vulkan').view(torch.bool);cpu=raw.view(torch.bool)
    torch.testing.assert_close(v.sigmoid().cpu(),cpu.sigmoid())
    assert torch.equal(v.square().cpu(),cpu.square())
    with pytest.raises(RuntimeError,match='Optional dtype must be either a floating point or complex dtype'):
        torch.ones(3,4,device='vulkan').mean(dtype=torch.long)

@pytest.mark.parametrize('n',[0,1,2,5,11])
@pytest.mark.parametrize('batch',[(),(2,),(2,1)])
@pytest.mark.parametrize('hermitian',[False,True])
def test_ldl_native_factors_pivots_and_solve(n,batch,hermitian):
    torch.manual_seed(844)
    source=torch.randn(batch+(n,n));a=(source+source.transpose(-1,-2))*.5
    if n==2:a=torch.tensor([[0.,1.],[1.,0.]]).expand(batch+(2,2)).clone()
    expected,pivots,info=torch.linalg.ldl_factor_ex(a,hermitian=hermitian)
    v=a.to('vulkan');vk.reset_execution_stats()
    actual,pv,iv=torch.linalg.ldl_factor_ex(v,hermitian=hermitian)
    resident();torch.testing.assert_close(actual.cpu(),expected)
    assert torch.equal(pv.cpu(),pivots);assert torch.equal(iv.cpu(),info)
    b=torch.randn(batch+(n,3))
    result=torch.linalg.ldl_solve(actual,pv,b.to('vulkan'),hermitian=hermitian)
    torch.testing.assert_close(result.cpu(),torch.linalg.ldl_solve(expected,pivots,b,hermitian=hermitian))

@pytest.mark.parametrize('n',[7,17,129])
def test_ldl_native_solve_accepts_cpu_factors_and_batch_broadcast(n):
    torch.manual_seed(862)
    source=torch.randn(2,n,n);a=source+source.transpose(-1,-2)+torch.eye(n)*.25
    ld,pivots=torch.linalg.ldl_factor(a)
    b=torch.randn(1,n,4)
    expected=torch.linalg.ldl_solve(ld,pivots,b)
    actual=torch.linalg.ldl_solve(ld.to('vulkan'),pivots.to('vulkan'),b.to('vulkan'))
    torch.testing.assert_close(actual.cpu(),expected)
    # Factorization uses the lower triangle, including a non-symmetric upper half.
    data=a+torch.triu(torch.randn_like(a),1)
    actual,pv,iv=torch.linalg.ldl_factor_ex(data.to('vulkan'))
    ld,pivots,info=torch.linalg.ldl_factor_ex(data)
    # CUDA and Vulkan use different floating-point update schedules from CPU
    # LAPACK at larger sizes. Validate the factorization through an independent
    # CPU FP64 solve and a normalized equation residual, while retaining exact
    # pivot/info checks and the small-matrix factor comparisons above.
    solution=torch.linalg.ldl_solve(actual.cpu().double(),pv.cpu(),b.double())
    effective=torch.tril(data).double()+torch.tril(data,-1).double().transpose(-1,-2)
    residual=(effective@solution-b.double()).abs().amax(-2)
    scale=effective.abs().sum(-1).amax(-1).unsqueeze(-1)*solution.abs().amax(-2)+b.double().abs().amax(-2)
    assert (residual/scale).max()<4*torch.finfo(torch.float32).eps
    assert torch.equal(pv.cpu(),pivots);assert torch.equal(iv.cpu(),info)


def test_ldl_native_singular_info_and_errors():
    a=torch.zeros(3,3,device='vulkan')
    result,pivots,info=torch.linalg.ldl_factor_ex(a)
    assert info.cpu().item()==1
    # Pinned CPU ATen raises RuntimeError with an unknown-info message for LDL.
    with pytest.raises(RuntimeError,match='Unknown error code: 1'):torch.linalg.ldl_factor_ex(a,check_errors=True)
    with pytest.raises(RuntimeError,match='square'):torch.linalg.ldl_factor_ex(torch.ones(2,3,device='vulkan'))


def test_ldl_solve_single_rounding_across_fp32_exponents():
    # In a 2x2 unit-diagonal LDL solve, the last result is exactly c - a*b.
    # Compare that operation to the host C99 single-rounded fmaf oracle. This
    # includes cancellation, halfway rounding, overflow and subnormal values.
    import ctypes
    import ctypes.util
    import random
    import struct
    libm=ctypes.CDLL(ctypes.util.find_library('m') or None)
    fmaf=libm.fmaf;fmaf.argtypes=[ctypes.c_float]*3;fmaf.restype=ctypes.c_float
    def fp32(raw):return struct.unpack('f',struct.pack('I',raw))[0]
    directed=[(0x3f800001,0x3f7ffffe,0x3f800000),
              (0x7f7fffff,0x40000000,0x7f7fffff),
              (0x00800000,0x3f000000,0x00800001),
              (0x00000001,0x3f000000,0x00000001),
              (0x00000003,0x3f000000,0x00000002),
              (0x3f800001,0x3f800001,0x3f800002)]
    rng=random.Random(863)
    bits=directed+[tuple(rng.getrandbits(32) for _ in range(3)) for _ in range(512)]
    triples=[tuple(fp32(raw) for raw in row) for row in bits
             if all((raw&0x7f800000)!=0x7f800000 for raw in row)]
    values=torch.tensor(triples);ld=torch.eye(2).repeat(len(triples),1,1)
    ld[:,1,0]=values[:,0];pivots=torch.tensor([1,2],dtype=torch.int32).expand(len(triples),2)
    rhs=values[:,1:].reshape(-1,2,1)
    expected=torch.tensor([fmaf(-a,b,c) for a,b,c in triples])
    result=torch.linalg.ldl_solve(ld.to('vulkan'),pivots.to('vulkan'),rhs.to('vulkan')).cpu()[:,1,0]
    torch.testing.assert_close(result,expected,rtol=0,atol=0,equal_nan=True)


@pytest.mark.parametrize('order',[-float('inf'),-2.,-.5,0.,.5,1.,2.,3.5,float('inf')])
@pytest.mark.parametrize('features',[1,5,129])
@pytest.mark.parametrize('index_dtype',[torch.int32,torch.int64])
def test_embedding_renorm_native_strides_duplicates_and_aliases(order,features,index_dtype):
    torch.manual_seed(871)
    source=torch.randn(2*features+1,15)
    cpu=source[1::2,1::2].transpose(0,1)
    storage=source.to('vulkan');weight=storage[1::2,1::2].transpose(0,1)
    aliases=weight[:,::2];pointer=weight.data_ptr()
    cpu[0].zero_();weight[0].zero_()
    indices=torch.tensor([[4,2,4],[0,6,2]],dtype=index_dtype).t()
    expected=cpu.clone();torch.embedding_renorm_(expected,indices,.75,order)
    vk.reset_execution_stats()
    assert torch.embedding_renorm_(weight,indices.to('vulkan'),.75,order) is weight
    assert vk.execution_stats()['cpu_fallback_calls']==0
    assert weight.data_ptr()==pointer
    torch.testing.assert_close(weight.cpu(),expected,equal_nan=True)
    torch.testing.assert_close(aliases.cpu(),expected[:,::2],equal_nan=True)


@pytest.mark.parametrize('order',[0.,1.,2.,float('inf'),-1.,-float('inf')])
@pytest.mark.parametrize('maximum',[-.5,0.,2.])
def test_embedding_renorm_native_nonfinite_and_negative_indices(order,maximum):
    source=torch.tensor([[0.,0.,0.],[1.,float('inf'),2.],[float('nan'),3.,4.],[1e-30,2e-30,3e-30],[1e30,-2e30,3e30]])
    indices=torch.tensor([-1,0,1,2,3,-1])
    expected=source.clone();torch.embedding_renorm_(expected,indices,maximum,order)
    weight=source.to('vulkan');torch.embedding_renorm_(weight,indices.to('vulkan'),maximum,order)
    torch.testing.assert_close(weight.cpu(),expected,equal_nan=True)


def test_embedding_renorm_native_training_and_bounds():
    torch.manual_seed(872)
    cpu=torch.randn(17,9,requires_grad=True);weight=cpu.detach().to('vulkan').requires_grad_()
    indices=torch.tensor([[1,5,1],[3,0,16]])
    expected=torch.nn.functional.embedding(indices,cpu,max_norm=.5,norm_type=1.5,scale_grad_by_freq=True)
    vk.reset_execution_stats()
    actual=torch.nn.functional.embedding(indices.to('vulkan'),weight,max_norm=.5,norm_type=1.5,scale_grad_by_freq=True)
    expected.square().sum().backward();actual.square().sum().backward()
    assert vk.execution_stats()['cpu_fallback_calls']==0
    torch.testing.assert_close(actual.cpu(),expected)
    torch.testing.assert_close(weight.grad.cpu(),cpu.grad)
    for invalid in [-18,17,2**32]:
        with pytest.raises(IndexError):torch.embedding_renorm_(weight.detach(),torch.tensor([invalid],device='vulkan'),1.,2.)


@pytest.mark.parametrize('mode,weighted',[('sum',False),('sum',True),('mean',False)])
@pytest.mark.parametrize('include_last_offset',[False,True])
@pytest.mark.parametrize('padding_idx',[None,3])
def test_embedding_bag_native_sparse_weight_gradients(mode,weighted,include_last_offset,padding_idx):
    torch.manual_seed(873)
    cpu=torch.randn(13,7,requires_grad=True);weight=cpu.detach().to('vulkan').requires_grad_()
    indices=torch.tensor([3,1,3,2,1,3,9,9,2]);offsets=torch.tensor([0,3,3,7]+([9] if include_last_offset else []))
    ps=torch.randn(indices.numel(),requires_grad=True) if weighted else None
    psv=ps.detach().to('vulkan').requires_grad_() if weighted else None
    kwargs=dict(mode=mode,sparse=True,include_last_offset=include_last_offset,padding_idx=padding_idx)
    expected=torch.nn.functional.embedding_bag(indices,cpu,offsets,per_sample_weights=ps,**kwargs)
    vk.reset_execution_stats()
    actual=torch.nn.functional.embedding_bag(indices.to('vulkan'),weight,offsets.to('vulkan'),per_sample_weights=psv,**kwargs)
    grad=torch.randn_like(expected);expected.backward(grad);actual.backward(grad.to('vulkan'))
    assert weight.grad.layout==torch.sparse_coo
    assert vk.execution_stats()['cpu_fallback_calls']==0
    torch.testing.assert_close(actual.cpu(),expected)
    torch.testing.assert_close(weight.grad.to_dense().cpu(),cpu.grad.to_dense())
    if weighted:torch.testing.assert_close(psv.grad.cpu(),ps.grad)


@pytest.mark.parametrize('dtype',[torch.bool,torch.uint8,torch.int8,torch.int16,torch.int32,torch.int64,torch.float32,torch.complex64])
@pytest.mark.parametrize('sparse_dim',[1,2,3])
def test_sparse_coo_typed_dense_roundtrip_mask_and_duplicate_coalesce(dtype,sparse_dim):
    raw=torch.arange(2*3*5,dtype=torch.int64).reshape(2,3,5)
    data=((raw%7)-3).to(dtype);data[raw%3==0]=0
    if dtype==torch.int64:data[data!=0]+=2**54
    if dtype==torch.complex64:data=data+data.flip(-1)*1j
    view=data.transpose(0,2);v=data.to('vulkan').transpose(0,2)
    # Transfer a non-contiguous view so packing, coordinate extraction and exact
    # value storage all participate in the conversion.
    expected=view.to_sparse(sparse_dim)
    vk.reset_execution_stats();actual=v.to_sparse(sparse_dim)
    assert vk.execution_stats()['cpu_fallback_calls']==0
    torch.testing.assert_close(actual.to_dense().cpu(),view,rtol=0,atol=0)
    torch.testing.assert_close(actual.indices().cpu(),expected.indices(),rtol=0,atol=0)
    torch.testing.assert_close(actual.values().cpu(),expected.values(),rtol=0,atol=0)
    indices=torch.cat([expected.indices(),expected.indices().flip(1)],dim=1)
    values=torch.cat([expected.values(),expected.values().flip(0)],dim=0)
    reference=torch.sparse_coo_tensor(indices,values,view.shape).coalesce()
    duplicate=torch.sparse_coo_tensor(indices.to('vulkan'),values.to('vulkan'),view.shape).coalesce()
    torch.testing.assert_close(duplicate.indices().cpu(),reference.indices(),rtol=0,atol=0)
    torch.testing.assert_close(duplicate.values().cpu(),reference.values(),rtol=0,atol=0)
    torch.testing.assert_close(duplicate.to_dense().cpu(),reference.to_dense(),rtol=0,atol=0)
    masked=v.sparse_mask(actual)
    torch.testing.assert_close(masked.values().cpu(),expected.values(),rtol=0,atol=0)


def test_sparse_coo_typed_noncanonical_bool_and_int64_overflow():
    bits=torch.tensor([[0,3,128,0,254],[2,0,0,129,7]],dtype=torch.uint8)
    value=bits.view(torch.bool).to('vulkan').to_sparse()
    torch.testing.assert_close(value.to_dense().cpu(),bits.bool())
    idx=torch.tensor([[0,0,1,1],[1,1,0,0]])
    values=torch.tensor([2**63-1,3,-2**63,-5])
    expected=torch.sparse_coo_tensor(idx,values,(2,2)).coalesce()
    actual=torch.sparse_coo_tensor(idx.to('vulkan'),values.to('vulkan'),(2,2)).coalesce()
    torch.testing.assert_close(actual.values().cpu(),expected.values(),rtol=0,atol=0)
    torch.testing.assert_close(actual.to_dense().cpu(),expected.to_dense(),rtol=0,atol=0)


@pytest.mark.parametrize('dtype',[torch.bool,torch.uint8,torch.int8,torch.int16,torch.int32,torch.int64,torch.float32,torch.complex64])
def test_sparse_coo_zero_sparse_dim_constructor(dtype):
    indices=torch.empty(0,1,dtype=torch.long);values=torch.arange(6).reshape(1,2,3).to(dtype)
    expected=torch.sparse_coo_tensor(indices,values,(2,3),check_invariants=True)
    actual=torch.sparse_coo_tensor(indices.to('vulkan'),values.to('vulkan'),(2,3),check_invariants=True)
    torch.testing.assert_close(actual.to_dense().cpu(),expected.to_dense(),rtol=0,atol=0)
    dense=values[0].to('vulkan')
    torch.testing.assert_close(dense.sparse_mask(actual).values().cpu(),values,rtol=0,atol=0)


def test_sparse_coo_conversion_zero_sparse_dim_error():
    for device in ['cpu','vulkan']:
        with pytest.raises(RuntimeError,match='sparse_dim argument must be in >0'):
            torch.ones(2,3,device=device).to_sparse(0)

@pytest.mark.parametrize('order',[-float('inf'),-2.,-.5,0.,.5,2.,float('inf')])
def test_embedding_renorm_native_empty_features(order):
    cpu=torch.empty(3,0);weight=cpu.to('vulkan');indices=torch.tensor([1,1])
    if order<0 or abs(order)==float('inf'):
        for x,idx in [(cpu,indices),(weight,indices.to('vulkan'))]:
            with pytest.raises(RuntimeError,match='operation does not have an identity'):torch.embedding_renorm_(x,idx,1.,order)
    else:
        torch.embedding_renorm_(cpu,indices,1.,order);torch.embedding_renorm_(weight,indices.to('vulkan'),1.,order)
        torch.testing.assert_close(weight.cpu(),cpu)


@pytest.mark.parametrize('layout',[torch.sparse_csr,torch.sparse_csc,torch.sparse_bsr,torch.sparse_bsc])
@pytest.mark.parametrize('dtype',[torch.float32,torch.bool,torch.int64,torch.complex64])
@pytest.mark.parametrize('batch_dense',[((),0),((2,),0),((2,),1)])
def test_sparse_compressed_native_formats_batches_dense_blocks(layout,dtype,batch_dense):
    batch,dense=batch_dense
    torch.manual_seed(881)
    shape=batch+(6,8)+((3,) if dense else ())
    data=torch.randint(1,4,shape).to(dtype)
    if dtype==torch.int64:data[data!=0]+=2**54
    # Each batch has the same sparsity pattern; values can differ.
    mask=(torch.arange(48).reshape(6,8)%5!=0)
    if dense:mask=mask.unsqueeze(-1)
    data=data.masked_fill(mask,0)
    kwargs={'layout':layout,'dense_dim':dense}
    if layout in [torch.sparse_bsr,torch.sparse_bsc]:kwargs['blocksize']=(2,2)
    expected=data.to_sparse(**kwargs)
    vk.reset_execution_stats();actual=data.to('vulkan').to_sparse(**kwargs)
    torch.testing.assert_close(actual.to_dense().cpu(),data,rtol=0,atol=0)
    assert vk.execution_stats()['cpu_fallback_calls']==0
    assert actual.layout==layout
    cp='crow_indices' if layout in [torch.sparse_csr,torch.sparse_bsr] else 'ccol_indices'
    ip='col_indices' if layout in [torch.sparse_csr,torch.sparse_bsr] else 'row_indices'
    torch.testing.assert_close(getattr(actual,cp)().cpu(),getattr(expected,cp)(),rtol=0,atol=0)
    torch.testing.assert_close(getattr(actual,ip)().cpu(),getattr(expected,ip)(),rtol=0,atol=0)
    torch.testing.assert_close(actual.values().cpu(),expected.values(),rtol=0,atol=0)
    torch.testing.assert_close(actual.cpu().to_dense(),data,rtol=0,atol=0)
    torch.testing.assert_close(actual.transpose(len(batch),len(batch)+1).to_dense().cpu(),data.transpose(len(batch),len(batch)+1),rtol=0,atol=0)


@pytest.mark.parametrize('reduce',['sum','mean','amax','amin'])
@pytest.mark.parametrize('index_dtype',[torch.int32,torch.int64])
@pytest.mark.parametrize('shape',[(5,9,7),(137,29,3),(3,4,0),(0,4,3),(3,0,4)])
def test_sparse_csr_native_mm_reductions_and_gradients(reduce,index_dtype,shape):
    rows,cols,width=shape;torch.manual_seed(882)
    dense=torch.randn(rows,cols);dense[dense.abs()<.9]=0
    if rows:dense[0].zero_()
    reference=dense.to_sparse_csr()
    cpu=torch.sparse_csr_tensor(reference.crow_indices().to(index_dtype),reference.col_indices().to(index_dtype),reference.values(),dense.shape,check_invariants=True).requires_grad_()
    actual=torch.sparse_csr_tensor(cpu.crow_indices().to('vulkan'),cpu.col_indices().to('vulkan'),cpu.values().detach().to('vulkan'),dense.shape,check_invariants=True).requires_grad_()
    assert actual.is_leaf and cpu.is_leaf
    source=torch.randn(width,cols+1).t()[1:];b=source.detach().requires_grad_();bv=source.to('vulkan').detach().requires_grad_()
    expected=torch.sparse.mm(cpu,b,reduce=reduce)
    vk.reset_execution_stats();result=torch.sparse.mm(actual,bv,reduce=reduce)
    grad=torch.randn_like(expected);expected.backward(grad);result.backward(grad.to('vulkan'))
    assert vk.execution_stats()['cpu_fallback_calls']==0
    torch.testing.assert_close(result.cpu(),expected)
    torch.testing.assert_close(bv.grad.cpu(),b.grad)
    torch.testing.assert_close(actual.grad.to_dense().cpu(),cpu.grad.to_dense())


@pytest.mark.parametrize('shape',[(5,7,9),(0,3,4),(4,0,3),(4,3,0)])
@pytest.mark.parametrize('alpha,beta',[(.6,.2),(0.,1.),(1.,0.)])
def test_sparse_csr_native_sampled_addmm_training(shape,alpha,beta):
    rows,cols,width=shape;torch.manual_seed(883)
    dense=torch.randn(rows,cols);dense[dense.abs()<1.1]=0
    cpu=dense.to_sparse_csr().detach().requires_grad_();a=cpu.to('vulkan').detach().requires_grad_()
    b=torch.randn(rows,width,requires_grad=True);c=torch.randn(width,cols,requires_grad=True)
    bv=b.detach().to('vulkan').requires_grad_();cv=c.detach().to('vulkan').requires_grad_()
    expected=torch.sparse.sampled_addmm(cpu,b,c,alpha=alpha,beta=beta)
    vk.reset_execution_stats();actual=torch.sparse.sampled_addmm(a,bv,cv,alpha=alpha,beta=beta)
    expected.to_dense().square().sum().backward();actual.to_dense().square().sum().backward()
    assert vk.execution_stats()['cpu_fallback_calls']==0
    torch.testing.assert_close(actual.to_dense().cpu(),expected.to_dense())
    torch.testing.assert_close(a.grad.to_dense().cpu(),cpu.grad.to_dense())
    torch.testing.assert_close(bv.grad.cpu(),b.grad);torch.testing.assert_close(cv.grad.cpu(),c.grad)


@pytest.mark.parametrize('spatial',[2,3])
@pytest.mark.parametrize('ceil',[False,True])
@pytest.mark.parametrize('include',[False,True])
@pytest.mark.parametrize('divisor',[None,3,-3,2**54+3])
def test_average_pool_long_native_exact_strides_divisors_and_backward(spatial,ceil,include,divisor):
    torch.manual_seed(891)
    shape=(2,3)+((11,15) if spatial==2 else (9,11,15))
    raw=torch.randint(-1023,1024,shape,dtype=torch.long);raw+=raw.sign()*(2**54)
    base=raw.to('vulkan');index=(slice(None),slice(None))+(slice(1,None,2),)*spatial
    x=raw[index];v=base[index]
    kernel=(3,2) if spatial==2 else (2,3,2);stride=(2,2) if spatial==2 else (2,2,2);padding=(1,1) if spatial==2 else (0,1,1)
    fn=getattr(torch.nn.functional,'avg_pool'+str(spatial)+'d')
    expected=fn(x,kernel,stride,padding,ceil,include,divisor)
    vk.reset_execution_stats();actual=fn(v,kernel,stride,padding,ceil,include,divisor)
    assert vk.execution_stats()['cpu_fallback_calls']==0
    torch.testing.assert_close(actual.cpu(),expected,rtol=0,atol=0)
    grad=torch.randint(-10001,10002,expected.shape,dtype=torch.long);grad+=grad.sign()*(2**54)
    backward=getattr(torch.ops.aten,'avg_pool'+str(spatial)+'d_backward')
    expected_grad=backward(grad,x,kernel,stride,padding,ceil,include,divisor)
    actual_grad=backward(grad.to('vulkan'),v,kernel,stride,padding,ceil,include,divisor)
    torch.testing.assert_close(actual_grad.cpu(),expected_grad,rtol=0,atol=0)
    out=torch.empty(tuple(2*n for n in actual.shape),dtype=torch.long,device='vulkan')[(slice(None,None,2),)*actual.dim()]
    getattr(torch.ops.aten,'avg_pool'+str(spatial)+'d').out(v,kernel,stride,padding,ceil,include,divisor,out=out)
    torch.testing.assert_close(out.cpu(),expected,rtol=0,atol=0)


def test_average_pool_long_native_1d_and_wraparound():
    x=torch.tensor([[[2**63-1,2**63-1,-2**63,-2**63,-3,2,7]]])
    expected=torch.nn.functional.avg_pool1d(x,2,1)
    actual=torch.nn.functional.avg_pool1d(x.to('vulkan'),2,1)
    torch.testing.assert_close(actual.cpu(),expected,rtol=0,atol=0)


@pytest.mark.parametrize('size',[(0,3),(2,0),(0,0)])
def test_adaptive_average_pool_empty_output_forward_and_backward_contract(size):
    cpu=torch.randn(2,3,5,7,requires_grad=True);v=cpu.detach().to('vulkan').requires_grad_()
    expected=torch.nn.functional.adaptive_avg_pool2d(cpu,size);actual=torch.nn.functional.adaptive_avg_pool2d(v,size)
    torch.testing.assert_close(actual.cpu(),expected)
    for output in [expected,actual]:
        with pytest.raises(RuntimeError,match='Expected grad_output to have non-zero size'):output.sum().backward()


@pytest.mark.parametrize('n', [0, 1, 2, 5])
@pytest.mark.parametrize('kind', ['random', 'symmetric', 'diagonal', 'triangular', 'zeros', 'rotation'])
def test_real_general_eigenvalues_and_vectors(n, kind):
    from itertools import permutations
    torch.manual_seed(722)
    a = torch.randn(2, n, n)
    if kind == 'symmetric': a = a + a.mT
    if kind == 'diagonal': a = a.diagonal(dim1=-2, dim2=-1).diag_embed()
    if kind == 'triangular': a = a.triu()
    if kind == 'zeros': a.zero_()
    if kind == 'rotation':
        a.zero_()
        if n > 1: a[:, 0, 1] = -1; a[:, 1, 0] = 1
    av = a.to('vulkan').mT
    vk.reset_execution_stats()
    w, v = torch.linalg.eig(av)
    only = torch.linalg.eigvals(av)
    stats = vk.execution_stats()
    assert stats['cpu_fallback_calls'] == 0, stats
    assert stats['host_read_max_bytes'] <= 8, stats  # Per-matrix convergence integers.
    w, v, only = w.cpu(), v.cpu(), only.cpu()
    torch.testing.assert_close(only, w)
    torch.testing.assert_close(a.mT.to(torch.complex64) @ v, v * w.unsqueeze(-2), atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(v.abs().square().sum(-2), torch.ones_like(w.real), atol=2e-5, rtol=2e-5)
    expected = torch.linalg.eigvals(a.mT)
    # Eigenvalue ordering is unspecified; independently match the full multiset.
    for actual, reference in zip(w, expected):
        choices = list(permutations(range(n)))
        order = min(choices, key=lambda p: (actual - reference[list(p)]).abs().sum())
        torch.testing.assert_close(actual, reference[list(order)], atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize('left', [False, True])
@pytest.mark.parametrize('adjoint', [False, True])
@pytest.mark.parametrize('n', [0, 1, 5])
def test_complex_lu_solve_and_broadcast(left, adjoint, n):
    torch.manual_seed(726)
    a = torch.randn(1, n, n, dtype=torch.complex64) + torch.eye(n) * 3
    b = torch.randn((2, n, 3) if left else (2, 3, n), dtype=torch.complex64)
    av, bv = a.to('vulkan').mT, b.to('vulkan')
    vk.reset_execution_stats()
    lu, piv, info = torch.linalg.lu_factor_ex(av)
    out = torch.linalg.lu_solve(lu, piv, bv, left=left, adjoint=adjoint)
    assert vk.execution_stats()['cpu_fallback_calls'] == 0
    lc, pc = torch.linalg.lu_factor(a.mT)
    expected = torch.linalg.lu_solve(lc, pc, b, left=left, adjoint=adjoint)
    torch.testing.assert_close(out.cpu(), expected, atol=1e-5, rtol=1e-5)
    assert not info.cpu().any()


@pytest.mark.parametrize('scale', [2.**-140, 2.**120])
def test_general_eig_extreme_fp32_scale(scale):
    a = torch.tensor([[1., -2., 3.], [4., 5., -6.], [-7., 8., 9.]]) * scale
    w, v = torch.linalg.eig(a.to('vulkan'))
    w, v = w.cpu(), v.cpu()
    denominator = max(a.abs().max().item(), torch.finfo(torch.float32).tiny)
    torch.testing.assert_close((a.to(torch.complex64) / denominator) @ v,
                               v * (w / denominator).unsqueeze(-2), atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize('value', [float('inf'), -float('inf'), float('nan')])
def test_general_eig_rejects_nonfinite(value):
    a = torch.eye(3); a[1, 2] = value
    with pytest.raises(RuntimeError, match='should not contain infs or NaNs'):
        torch.linalg.eig(a.to('vulkan'))


@pytest.mark.parametrize('shape', [(3, 3), (2, 3, 3), (0, 3, 3)])
def test_real_eig_phase_invariant_gradients(shape):
    torch.manual_seed(731)
    a = torch.randn(shape)
    cpu = a.clone().requires_grad_(); gpu = a.to('vulkan').detach().requires_grad_()
    grads = []
    for x in [cpu, gpu]:
        w, v = torch.linalg.eig(x)
        loss = w.abs().square().sum() + v.abs().pow(4).sum()
        grads.append(torch.autograd.grad(loss, x)[0].cpu())
    torch.testing.assert_close(grads[1], grads[0], atol=1e-4, rtol=1e-4)


@pytest.mark.parametrize('batched', [False, True])
@pytest.mark.parametrize('conjugate', [False, True])
def test_complex_matmul_strides_gradients(batched, conjugate):
    torch.manual_seed(732)
    shape = (2,) if batched else ()
    a = torch.randn(*shape, 4, 3, dtype=torch.complex64)
    b = torch.randn(*shape, 4, 5, dtype=torch.complex64)
    results = []
    for device in ['cpu', 'vulkan']:
        x, y = [t.to(device).detach().requires_grad_() for t in (a, b)]
        xx = x.mH if conjugate else x.mT
        out = xx @ y
        gx, gy = torch.autograd.grad(out.abs().square().sum(), (x, y))
        results.append([t.cpu() for t in (out, gx, gy)])
    for actual, expected in zip(results[1], results[0]):
        torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)


@pytest.mark.parametrize('batched', [False, True])
def test_matmul_lazy_negative_views(batched):
    torch.manual_seed(733)
    shape = (2,) if batched else ()
    a, b = torch.randn(*shape, 4, 3), torch.randn(*shape, 3, 5)
    av, bv = a.to('vulkan'), b.to('vulkan')
    actual = torch._neg_view(av) @ torch._neg_view(bv)
    torch.testing.assert_close(actual.cpu(), a @ b)


@pytest.mark.parametrize('lower', [False, True])
def test_general_eig_triangular_isolated_rows(lower):
    torch.manual_seed(735)
    a = torch.randn(3, 8, 8).triu()
    if lower: a = a.mT
    w, v = torch.linalg.eig(a.to('vulkan'))
    w, v = w.cpu(), v.cpu()
    torch.testing.assert_close(w.imag, torch.zeros_like(w.imag), atol=0, rtol=0)
    torch.testing.assert_close(w.real.sort(-1).values, a.diagonal(dim1=-2, dim2=-1).sort(-1).values, atol=0, rtol=0)
    torch.testing.assert_close(a.to(torch.complex64) @ v, v*w.unsqueeze(-2), atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize('rank', [1, 2, 3])
@pytest.mark.parametrize('groups', [1, 2])
@pytest.mark.parametrize('dilation', [1, 2])
def test_integer_convolution_exact_large_values(rank, groups, dilation):
    spatial = (5,) * rank
    x = (torch.arange(2*2*5**rank, dtype=torch.int64).reshape(2, 2, *spatial)*257 + 2**56)
    w = torch.arange(4*(2//groups)*3**rank, dtype=torch.int64).reshape(4, 2//groups, *((3,)*rank)).remainder(7).sub(3)
    bias = torch.arange(4, dtype=torch.int64) + 2**60
    # Sliced storage and nonzero offsets must preserve all 64 value bits.
    backing = torch.empty(*x.shape[:-1], x.size(-1)*2+1, dtype=torch.int64)
    backing[..., 1::2] = x
    xv = backing.to('vulkan')[..., 1::2]
    op = getattr(torch.nn.functional, f'conv{rank}d')
    vk.reset_execution_stats()
    actual = op(xv, w.to('vulkan'), bias.to('vulkan'), stride=2, padding=2, dilation=dilation, groups=groups)
    assert vk.execution_stats()['cpu_fallback_calls'] == 0
    expected = op(x, w, bias, stride=2, padding=2, dilation=dilation, groups=groups)
    torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0)


@pytest.mark.parametrize('ceil', [False, True])
@pytest.mark.parametrize('dilation', [1, 2])
def test_integer_max_pool2d_exact_values_indices(ceil, dilation):
    x = torch.arange(2*3*7*9, dtype=torch.int64).reshape(2, 3, 7, 9) + 2**56
    x[..., 1:3, 2:5] = -2**63
    actual = torch.nn.functional.max_pool2d(x.to('vulkan').mT, 3, stride=2, padding=1,
                                           dilation=dilation, ceil_mode=ceil, return_indices=True)
    expected = torch.nn.functional.max_pool2d(x.mT, 3, stride=2, padding=1,
                                             dilation=dilation, ceil_mode=ceil, return_indices=True)
    for a, e in zip(actual, expected): torch.testing.assert_close(a.cpu(), e, atol=0, rtol=0)
    smallest = torch.full((1, 1, 3, 3), -2**63, dtype=torch.int64)
    a = torch.nn.functional.max_pool2d(smallest.to('vulkan'), 3, padding=1, return_indices=True)
    e = torch.nn.functional.max_pool2d(smallest, 3, padding=1, return_indices=True)
    for av, ev in zip(a, e): torch.testing.assert_close(av.cpu(), ev, atol=0, rtol=0)


@pytest.mark.parametrize('dtype', [torch.int32, torch.int64])
@pytest.mark.parametrize('start,end,step', [(.9, 4.1, 1.5), (.9, 4.1, .5), (-.9, 4.1, .5), (4.1, -.9, -.5)])
def test_integer_arange_fractional_scalar_parameters(dtype, start, end, step):
    expected = torch.arange(start, end, step, dtype=dtype)
    backing = torch.empty(expected.numel()*2+1, dtype=dtype, device='vulkan')
    out = backing[1::2]
    torch.arange(start, end, step, dtype=dtype, device='vulkan', out=out)
    torch.testing.assert_close(out.cpu(), expected, atol=0, rtol=0)


@pytest.mark.parametrize('dtype', [torch.int32, torch.int64])
@pytest.mark.parametrize('value', [.5, -2, 3])
def test_integer_addcmul_wraparound_broadcast_out(dtype, value):
    x = torch.tensor([[1], [-3]], dtype=dtype)
    a = torch.tensor([[2**30, -3, 7]], dtype=dtype)
    b = torch.tensor([[17], [-9]], dtype=dtype)
    out = torch.empty(3, 2, dtype=dtype, device='vulkan').T
    torch.addcmul(x.to('vulkan'), a.to('vulkan'), b.to('vulkan'), value=value, out=out)
    torch.testing.assert_close(out.cpu(), torch.addcmul(x, a, b, value=value), atol=0, rtol=0)


@pytest.mark.parametrize('rank', [1, 2, 3])
@pytest.mark.parametrize('mode', ['reflect', 'replicate'])
def test_integer_padding_preserves_value_bits(rank, mode):
    shape = (2, 3) + (5,)*rank
    a = torch.arange(torch.tensor(shape).prod().item(), dtype=torch.int64).reshape(shape) + 2**57
    a[..., 1:3] = -2**63
    padding = (1, 2)*rank
    actual = torch.nn.functional.pad(a.to('vulkan').transpose(0, 1), padding, mode=mode)
    expected = torch.nn.functional.pad(a.transpose(0, 1), padding, mode=mode)
    torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0)


@pytest.mark.parametrize('dtype', [torch.int32, torch.int64, torch.bool])
def test_chunk_cat_rejects_unsafe_output_cast(dtype):
    values = [torch.tensor([[[1.5, -2.5, 0.], [7.25, 8.5, 9.]]])]*2
    expected = torch.empty(0, dtype=dtype)
    with pytest.raises(TypeError, match="input types can.t be cast"):
        torch._chunk_cat(values, -1, 5, out=expected)
    actual = torch.empty(0, dtype=dtype, device='vulkan')
    with pytest.raises(TypeError, match="input types can.t be cast"):
        torch._chunk_cat([v.to('vulkan') for v in values], -1, 5, out=actual)


def test_boolean_eye_noncanonical_storage():
    backing = torch.full((5, 7), 255, dtype=torch.uint8, device='vulkan').view(torch.bool)
    out = backing[1:4, 1:6]
    torch.eye(3, 5, out=out)
    expected = torch.ones(5, 7, dtype=torch.bool); expected[1:4, 1:6] = torch.eye(3, 5, dtype=torch.bool)
    torch.testing.assert_close(backing.cpu(), expected)


@pytest.mark.parametrize('scale', [2.**-120, 2.**120, 3.e38])
def test_complex_lu_solve_scaled_singleton(scale):
    a = torch.tensor([[complex(scale, scale)]], dtype=torch.complex64)
    b = a.clone()
    lu, piv = torch.linalg.lu_factor(a.to('vulkan'))
    actual = torch.linalg.lu_solve(lu, piv, b.to('vulkan'))
    # CPU FP32 LU itself overflows for 3e38+3e38j; use a higher precision
    # reference for this finite, well-conditioned singleton solve.
    expected = torch.linalg.solve(a.to(torch.complex128), b.to(torch.complex128)).to(torch.complex64)
    torch.testing.assert_close(actual.cpu(), expected)


@pytest.mark.parametrize('dtype', [torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64])
@pytest.mark.parametrize('bounds', [(0.1, 9.9), (1, 5), (9, 1), (7, 7)])
def test_integer_linspace_endpoint_casts_and_strided_output(dtype, bounds):
    for steps in [0, 1, 2, 5, 8, 33]:
        expected = torch.linspace(*bounds, steps, dtype=dtype)
        backing = torch.empty(steps*2+1, dtype=dtype, device='vulkan')
        out = backing[1::2]
        torch.linspace(*bounds, steps, dtype=dtype, device='vulkan', out=out)
        torch.testing.assert_close(out.cpu(), expected, atol=0, rtol=0)


@pytest.mark.parametrize('bounds', [(2**54, 2**54+32), (-(2**54+3), -(2**54)+73),
                                    (-2**62, 2**62), (2**63-1, 2**63-9), (-2**63, -2**63+4096)])
def test_integer_linspace_large_endpoint_precision(bounds):
    for steps in [1, 2, 3, 5, 8, 17, 127]:
        actual = torch.linspace(*bounds, steps, dtype=torch.int64, device='vulkan')
        expected = torch.linspace(*bounds, steps, dtype=torch.int64)
        torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0)


@pytest.mark.parametrize('dtype', [torch.uint8, torch.int8, torch.int16])
def test_short_integer_arange_and_eye(dtype):
    for start, end, step in [(259, 269, 1), (4.1, -.9, -.5), (-.9, 4.1, .5)]:
        actual = torch.arange(start, end, step, dtype=dtype, device='vulkan')
        expected = torch.arange(start, end, step, dtype=dtype)
        torch.testing.assert_close(actual.cpu(), expected, atol=0, rtol=0)
    torch.testing.assert_close(torch.eye(3, 5, dtype=dtype, device='vulkan').cpu(), torch.eye(3, 5, dtype=dtype))


@pytest.mark.parametrize('ceil', [False, True])
@pytest.mark.parametrize('dilation', [1, 2])
def test_integer_max_pool3d_exact_values_indices(ceil, dilation):
    a = torch.arange(2*3*5*7*9, dtype=torch.int64).reshape(2, 3, 5, 7, 9) + 2**56
    a[..., 1:3, 2:5] = -2**63
    for x in [a, torch.full_like(a, -2**63)]:
        actual = torch.nn.functional.max_pool3d(x.to('vulkan').mT, 3, stride=2, padding=1,
                                               dilation=dilation, ceil_mode=ceil, return_indices=True)
        expected = torch.nn.functional.max_pool3d(x.mT, 3, stride=2, padding=1,
                                                 dilation=dilation, ceil_mode=ceil, return_indices=True)
        for av, ev in zip(actual, expected): torch.testing.assert_close(av.cpu(), ev, atol=0, rtol=0)


@pytest.mark.parametrize('shape', [(), (0, 3), (2, 3), (7, 11)])
@pytest.mark.parametrize('op', ['silu', 'exp', 'neg'])
def test_complex_unary_silu_exp_neg_strides(shape, op):
    torch.manual_seed(752)
    x = torch.randn(shape, dtype=torch.complex64)*3
    if x.dim() > 1: x = x.mT
    fn = torch.nn.functional.silu if op == 'silu' else getattr(torch, op)
    xv = x.to('vulkan')
    vk.reset_execution_stats(); actual = fn(xv); resident()
    torch.testing.assert_close(actual.cpu(), fn(x))


def test_complex_gradient_accumulation_alias_and_conjugation():
    torch.manual_seed(753)
    x = torch.randn(2, 3, 4, dtype=torch.complex64)
    outputs = []
    for device in ['cpu', 'vulkan']:
        t = x.to(device).detach().requires_grad_()
        out = t*t.conj()
        grad = torch.autograd.grad(out.abs().sum(), t)[0]
        outputs.append(grad.cpu())
    torch.testing.assert_close(outputs[1], outputs[0])
    target = x.to('vulkan').mT
    expected = x.mT.clone(); expected.add_(expected, alpha=.25+.5j)
    target.add_(target, alpha=.25+.5j)
    torch.testing.assert_close(target.cpu(), expected)


@pytest.mark.parametrize('exponent', [-3., -.5, 0., .5, 1., 2., 3., 2.+1.j])
def test_complex_scalar_power_general(exponent):
    torch.manual_seed(755)
    x = (torch.randn(3, 5, dtype=torch.complex64)+2).mT
    vk.reset_execution_stats();actual = x.to('vulkan').pow(exponent);resident()
    torch.testing.assert_close(actual.cpu(), x.pow(exponent), rtol=2e-5, atol=2e-5)


def test_complex_image_columns_training():
    torch.manual_seed(756)
    x = torch.randn(2, 3, 5, 7, dtype=torch.complex64)
    values=[]
    for device in ['cpu', 'vulkan']:
        leaf=x.to(device).detach().requires_grad_(); t=leaf.mT
        columns=torch.nn.functional.unfold(t,3,padding=1,stride=2)
        out=torch.nn.functional.fold(columns,t.shape[-2:],3,padding=1,stride=2)
        grad=torch.autograd.grad(out.abs().square().sum(),leaf)[0]
        values.append([v.detach().cpu() for v in (columns,out,grad)])
    for a,e in zip(values[1],values[0]):torch.testing.assert_close(a,e)


@pytest.mark.parametrize('op', ['silu', 'exp', 'neg'])
def test_complex_unary_inplace_alias(op):
    torch.manual_seed(757)
    x=torch.randn(3,5,dtype=torch.complex64).mT
    actual=x.to('vulkan');pointer=actual.data_ptr()
    if op=='silu':
        expected=torch.nn.functional.silu(x,inplace=False)
        torch.nn.functional.silu(actual,inplace=True)
    else:
        expected=getattr(torch,op)(x)
        getattr(actual,op+'_')()
    assert actual.data_ptr()==pointer
    torch.testing.assert_close(actual.cpu(),expected)


def test_complex_silu_preserves_upstream_autograd_rejection():
    for device in ['cpu','vulkan']:
        x=torch.ones(3,dtype=torch.complex64,device=device,requires_grad=True)
        with pytest.raises(RuntimeError,match='automatic differentiation'):
            torch.nn.functional.silu(x)


@pytest.mark.parametrize('dtype', [torch.float32, torch.complex64])
@pytest.mark.parametrize('shape,grad_shape,dim,step', [((10,20),(1,6,10),1,2), ((2,3,20,4),(1,3,6,1,10),2,2), ((10,20),(10,1,10),1,2), ((10,20),(10,6,10),1,2), ((10,20),(1,8,10),1,2), ((10,20),(1,1,1),1,10)])
def test_unfold_backward_broadcasts_non_window_axes(dtype, shape, grad_shape, dim, step):
    torch.manual_seed(758)
    grad=torch.randn(grad_shape,dtype=dtype)
    if grad_shape==(10,6,10):
        # A broadcast window is a size-10, zero-stride view. A raw size-1
        # window is outside the overlapping kernel contract in PyTorch.
        grad=grad[..., :1].expand(grad_shape)
    expected=torch.ops.aten.unfold_backward(grad,list(shape),dim,10,step)
    actual=torch.ops.aten.unfold_backward(grad.to('vulkan'),list(shape),dim,10,step)
    torch.testing.assert_close(actual.cpu(),expected)


@pytest.mark.parametrize('dtype', [torch.float32, torch.complex64])
def test_unfold_backward_empty_broadcast_window(dtype):
    for shape,dim,grad_shape in [((3,8),1,(1,1,1)),((),0,())]:
        grad=torch.ones(grad_shape,dtype=dtype)
        expected=torch.ops.aten.unfold_backward(grad,list(shape),dim,0,2)
        actual=torch.ops.aten.unfold_backward(grad.to('vulkan'),list(shape),dim,0,2)
        torch.testing.assert_close(actual.cpu(),expected)


@pytest.mark.parametrize('dtype', [torch.float32, torch.complex64])
def test_unfold_backward_nonoverlap_scalar_broadcast(dtype):
    grad=torch.tensor(2.,dtype=dtype)
    expected=torch.ops.aten.unfold_backward(grad,[3,20],1,10,10)
    actual=torch.ops.aten.unfold_backward(grad.to('vulkan'),[3,20],1,10,10)
    torch.testing.assert_close(actual.cpu(),expected)


@pytest.mark.parametrize('dtype', [torch.float32,torch.uint8,torch.int8,torch.int16,torch.int32,torch.int64,torch.bool,torch.complex64])
@pytest.mark.parametrize('length', [0,2])
def test_narrow_copy_out_native_typed(dtype,length):
    x=torch.arange(60).reshape(3,4,5).to(dtype).transpose(0,2)
    expected=torch.narrow_copy(x,1,-3,length)
    actual=x.to('vulkan')
    for out in [torch.empty(0,dtype=dtype,device='vulkan'),torch.empty((3,length,5),dtype=dtype,device='vulkan').transpose(0,2)]:
        result=torch.narrow_copy(actual,1,-3,length,out=out)
        assert result is out
        torch.testing.assert_close(result.cpu(),expected)


@pytest.mark.parametrize('alias', ['exact','partial'])
def test_narrow_copy_out_native_alias(alias):
    x=torch.arange(12.).reshape(3,4)
    expected=x.narrow(1,1,2).clone()
    actual=x.to('vulkan');out=actual if alias=='exact' else actual[:,:2]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',UserWarning)
        result=torch.narrow_copy(actual,1,1,2,out=out)
    assert result is out
    torch.testing.assert_close(result.cpu(),expected)


@pytest.mark.parametrize('dtype', [torch.int64,torch.complex64])
def test_narrow_copy_out_native_rejects_dtype_change(dtype):
    for device in ['cpu','vulkan']:
        with pytest.raises(RuntimeError,match=r'self.dtype\(\) == output.dtype\(\)'):
            torch.narrow_copy(torch.ones(3,4,device=device),1,0,2,out=torch.empty(0,dtype=dtype,device=device))
