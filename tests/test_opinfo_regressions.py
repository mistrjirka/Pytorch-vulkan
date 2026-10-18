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
