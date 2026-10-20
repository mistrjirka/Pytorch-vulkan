"""Generic primitive parity, views, nonfinite values, aliases and autograd."""
import os
import pytest
import torch
import torch.nn.functional as F
import torch_vulkan


def view(x, layout):
    if layout == "transpose":
        return x.transpose(0, 1)
    if layout == "offset":
        return x[1:, 1:]
    if layout == "stepped":
        return x[:, ::2]
    if layout == "expanded":
        return x[:1, :].expand(5, -1)
    return x


@pytest.mark.parametrize("op", [torch.sigmoid, torch.abs, torch.sgn, F.logsigmoid])
@pytest.mark.parametrize("layout", ["contiguous", "transpose", "offset", "stepped", "expanded"])
def test_unary_views(op, layout):
    torch.manual_seed(7)
    x = torch.randn(7, 11)
    a, b = view(x, layout), view(x.to("vulkan"), layout)
    torch.testing.assert_close(op(b).cpu(), op(a), rtol=3e-6, atol=2e-7)


@pytest.mark.parametrize("op", [torch.sigmoid, torch.abs, torch.sgn, F.logsigmoid])
def test_unary_nonfinite_and_empty(op):
    a = torch.tensor([-float("inf"), -100., -20., -2., -0., 0., 2., 20.,
                      100., float("inf"), float("nan")])
    torch.testing.assert_close(op(a.to("vulkan")).cpu(), op(a),
                               rtol=3e-6, atol=2e-7, equal_nan=True)
    a = torch.empty(2, 0, 3)
    torch.testing.assert_close(op(a.to("vulkan")).cpu(), op(a))


@pytest.mark.parametrize("op", [torch.sub, torch.div])
@pytest.mark.parametrize("shapes", [((3, 1, 5), (1, 7, 1)), ((), (2, 3)),
                                    ((0, 3), (1, 3)), ((2, 1, 3, 1), (1, 4, 1, 5))])
def test_binary_broadcast(op, shapes):
    torch.manual_seed(8)
    a, b = (torch.randn(s) + 2 for s in shapes)
    kwargs = {"alpha": -2.5} if op == torch.sub else {}
    torch.testing.assert_close(op(a.to("vulkan"), b.to("vulkan"), **kwargs).cpu(),
                               op(a, b, **kwargs), rtol=3e-6, atol=2e-6)


@pytest.mark.parametrize("op", [torch.sub, torch.div])
def test_binary_strided_nonfinite_scalar_and_dtype(op):
    a = torch.arange(48, dtype=torch.float32).reshape(6, 8) - 20
    b = torch.tensor([[-float("inf"), -1., -0., 0., 1., float("inf"), float("nan"), 3.]])
    av, bv = a.to("vulkan"), b.to("vulkan")
    torch.testing.assert_close(op(av[1::2, ::2], bv[:, ::2]).cpu(),
                               op(a[1::2, ::2], b[:, ::2]), equal_nan=True)
    for scalar in [torch.tensor(2.), torch.tensor(2., dtype=torch.float64)]:
        torch.testing.assert_close(op(av, scalar).cpu(), op(a, scalar))
        torch.testing.assert_close(op(scalar, av).cpu(), op(scalar, a), equal_nan=True)
    for dtype in [torch.float64, torch.int64]:
        a = torch.arange(6, dtype=dtype).reshape(2, 3)
        b = torch.full((1, 3), 2, dtype=dtype)
        torch.testing.assert_close(op(a.to("vulkan"), b.to("vulkan")).cpu(), op(a, b))


@pytest.mark.parametrize("layout", ["contiguous", "transpose", "offset", "stepped"])
@pytest.mark.parametrize("op", ["fill", "div"])
def test_scalar_mutation_preserves_storage(op, layout):
    a = torch.arange(77, dtype=torch.float32).reshape(7, 11)
    b = a.to("vulkan")
    x, y = view(a, layout), view(b, layout)
    address = y.data_ptr()
    if op == "fill":
        x.fill_(-1.25)
        result = y.fill_(-1.25)
    else:
        x.div_(-3)
        result = y.div_(-3)
    assert result.data_ptr() == address
    torch.testing.assert_close(b.cpu(), a)


def test_fill_expanded_and_overlapping_views():
    for maker in [lambda x: x[:1].expand(5, -1),
                  lambda x: x.as_strided((4, 4), (1, 1), 2)]:
        a = torch.arange(40, dtype=torch.float32).reshape(5, 8)
        b = a.to("vulkan")
        maker(a).fill_(7)
        maker(b).fill_(7)
        torch.testing.assert_close(b.cpu(), a)


def test_div_expanded_rejects_and_fill_dtype_fallback():
    for device in ["cpu", "vulkan"]:
        x = torch.ones(1).to(device).expand(4)
        with pytest.raises(RuntimeError):
            x.div_(2)
    for dtype in [torch.int64, torch.float64]:
        a = torch.arange(12, dtype=dtype).reshape(3, 4)
        b = a.to("vulkan")
        a[:, ::2].fill_(3)
        b[:, ::2].fill_(3)
        torch.testing.assert_close(b.cpu(), a)


@pytest.mark.parametrize("shape", [(), (0, 5), (1,), (127,), (1025,), (3, 17, 19), (1025, 1025)])
@pytest.mark.parametrize("keepdim", [False, True])
def test_global_mean(shape, keepdim):
    torch.manual_seed(9)
    a = torch.randn(shape)
    b = a.to("vulkan")
    dims = tuple(range(a.ndim))
    torch.testing.assert_close(b.mean(dim=dims, keepdim=keepdim).cpu(),
                               a.mean(dim=dims, keepdim=keepdim),
                               rtol=3e-5, atol=2e-7, equal_nan=True)
    torch.testing.assert_close(b.mean().cpu(), a.mean(),
                               rtol=3e-5, atol=2e-7, equal_nan=True)


def test_mean_views_fallbacks_and_dim_validation():
    a = torch.randn(5, 7, 9)
    b = a.to("vulkan")
    torch.testing.assert_close(b[1:, ::2].transpose(0, 2).mean().cpu(),
                               a[1:, ::2].transpose(0, 2).mean())
    torch.testing.assert_close(b.mean(dim=(0, 2), keepdim=True).cpu(),
                               a.mean(dim=(0, 2), keepdim=True))
    torch.testing.assert_close(b.mean(dtype=torch.float64).cpu(),
                               a.mean(dtype=torch.float64))
    for dim in [(0, 0), (4,), (-4,)]:
        for x in [a, b]:
            with pytest.raises((RuntimeError, IndexError)):
                x.mean(dim=dim)


def test_log_sigmoid_buffer_contract():
    a = torch.tensor([-float("inf"), -20., -2., -0., 2., 20., float("inf"), float("nan")])
    expected = torch.ops.aten.log_sigmoid_forward(a)
    actual = torch.ops.aten.log_sigmoid_forward(a.to("vulkan"))
    for x, y in zip(actual, expected):
        torch.testing.assert_close(x.cpu(), y, rtol=3e-6, atol=2e-7, equal_nan=True)
    g = torch.arange(a.numel(), dtype=torch.float32)
    dx = torch.ops.aten.log_sigmoid_backward(g.to("vulkan"), a.to("vulkan"), actual[1])
    ref = torch.ops.aten.log_sigmoid_backward(g, a, expected[1])
    torch.testing.assert_close(dx.cpu(), ref, rtol=3e-6, atol=2e-7, equal_nan=True)


@pytest.mark.parametrize("op", [torch.sigmoid, torch.abs, F.logsigmoid,
                                lambda x: x.mean(), lambda x: x / 3.5])
def test_unary_autograd(op):
    torch.manual_seed(11)
    a = torch.randn(3, 5, requires_grad=True)
    b = a.detach().to("vulkan").requires_grad_()
    x, y = op(a), op(b)
    g = torch.randn_like(x)
    x.backward(g)
    y.backward(g.to("vulkan"))
    torch.testing.assert_close(y.cpu(), x, rtol=3e-6, atol=2e-7)
    torch.testing.assert_close(b.grad.cpu(), a.grad, rtol=3e-6, atol=2e-7)


@pytest.mark.parametrize("op", [torch.sub, torch.div])
def test_binary_broadcast_autograd(op):
    torch.manual_seed(12)
    a = torch.randn(3, 5, requires_grad=True)
    z = (torch.randn(1, 5) + 4).requires_grad_()
    b, w = (x.detach().to("vulkan").requires_grad_() for x in (a, z))
    x, y = op(a, z), op(b, w)
    g = torch.randn_like(x)
    x.backward(g)
    y.backward(g.to("vulkan"))
    torch.testing.assert_close(b.grad.cpu(), a.grad)
    torch.testing.assert_close(w.grad.cpu(), z.grad)


def test_primitive_chain_does_not_flush_deferred_work():
    if not all(torch_vulkan.performance_options()[k] for k in ("gpu_resident", "deferred_submissions")):
        return
    a = torch.randn(7, 13)
    b = a.to("vulkan")
    divisor = torch.full((1, 13), 2.).to("vulkan")
    torch_vulkan.synchronize()
    before = torch_vulkan.queue_stats()["submissions"]
    y = torch.sigmoid(b)
    y = torch.abs(y - b)
    y = torch.div(y, divisor).mean()
    assert torch_vulkan.queue_stats()["submissions"] == before
    expected = (torch.sigmoid(a) - a).abs().div(2).mean()
    torch.testing.assert_close(y.cpu(), expected)


@pytest.mark.parametrize('values', [[float('inf'), 1.], [-float('inf'), 1.],
                                    [float('inf'), -float('inf')], [float('nan'), 1.]])
def test_mean_nonfinite(values):
    a = torch.tensor(values)
    torch.testing.assert_close(a.to('vulkan').mean().cpu(), a.mean(), equal_nan=True)


def test_rank_nine_fallbacks_and_scalar_division_nonfinite():
    a = torch.arange(4, dtype=torch.float32).reshape(2, 1, 1, 1, 1, 1, 1, 1, 2)
    b = a.to('vulkan')
    for op in [torch.sigmoid, torch.abs, torch.sgn, F.logsigmoid, torch.mean]:
        torch.testing.assert_close(op(b).cpu(), op(a))
    b.div_(2)
    a.div_(2)
    torch.testing.assert_close(b.cpu(), a)
    a = torch.tensor([-float('inf'), -1., -0., 0., 1., float('inf'), float('nan')])
    for scalar in [0., -0., float('inf'), float('nan')]:
        b = a.to('vulkan')
        torch.testing.assert_close((b / scalar).cpu(), a / scalar, equal_nan=True)
        b.div_(scalar)
        torch.testing.assert_close(b.cpu(), a / scalar, equal_nan=True)


@pytest.mark.parametrize('layout', ['contiguous', 'transpose', 'offset', 'stepped'])
@pytest.mark.parametrize('rhs_kind', ['cpu_scalar', 'device_scalar', 'broadcast', 'self'])
def test_div_tensor_mutation_layout_and_broadcast(layout, rhs_kind):
    a = torch.arange(77, dtype=torch.float32).reshape(7, 11) + 1
    b = a.to('vulkan')
    x, y = view(a, layout), view(b, layout)
    if rhs_kind == 'self':
        rhs, vrhs = x, y
    elif rhs_kind == 'broadcast':
        rhs = torch.arange(x.shape[-1], dtype=torch.float32).reshape(1, -1) + 2
        vrhs = rhs.to('vulkan')
    else:
        rhs = torch.tensor(-3.)
        vrhs = rhs if rhs_kind == 'cpu_scalar' else rhs.to('vulkan')
    address = y.data_ptr()
    x.div_(rhs)
    result = torch.ops.aten.div_.Tensor(y, vrhs)
    assert result.data_ptr() == address
    torch.testing.assert_close(b.cpu(), a)


def test_div_tensor_overlap_and_broadcast_errors():
    for device in ['cpu', 'vulkan']:
        x = torch.arange(12, dtype=torch.float32).to(device) + 1
        with pytest.raises(RuntimeError):
            x[1:].div_(x[:-1])
        x = torch.ones(1, 3).to(device)
        with pytest.raises(RuntimeError):
            x.div_(torch.ones(2, 3).to(device))
        x = torch.ones(1).to(device).expand(3)
        with pytest.raises(RuntimeError):
            x.div_(torch.ones(3).to(device))


def test_div_tensor_mutation_fallback_preserves_storage():
    for dtype in [torch.float32, torch.float64]:
        a = torch.arange(48, dtype=dtype).reshape(6, 8) + 1
        b = a.to('vulkan')
        rhs = torch.tensor([[2., 3., 4., 5.]], dtype=torch.float64)
        a[1::2, ::2].div_(rhs)
        b[1::2, ::2].div_(rhs.to('vulkan'))
        torch.testing.assert_close(b.cpu(), a)


@pytest.mark.parametrize('op', ['sub_', 'mul_'])
@pytest.mark.parametrize('layout', ['contiguous', 'transpose', 'offset', 'stepped'])
@pytest.mark.parametrize('rhs_kind', ['scalar', 'cpu_scalar', 'device_scalar', 'broadcast', 'self'])
def test_mutation_arithmetic_views(op, layout, rhs_kind):
    a = torch.arange(77, dtype=torch.float32).reshape(7, 11) / 9 + 1
    b = a.to('vulkan')
    x, y = view(a, layout), view(b, layout)
    if rhs_kind == 'self':
        rhs, vrhs = x, y
    elif rhs_kind == 'broadcast':
        rhs = torch.arange(x.shape[-1], dtype=torch.float32).reshape(1, -1) / 7 + 2
        vrhs = rhs.to('vulkan')
    elif rhs_kind == 'scalar':
        rhs = vrhs = -2.5
    else:
        rhs = torch.tensor(-2.5)
        vrhs = rhs if rhs_kind == 'cpu_scalar' else rhs.to('vulkan')
    kwargs = {'alpha': -1.75} if op == 'sub_' else {}
    address = y.data_ptr()
    getattr(x, op)(rhs, **kwargs)
    result = getattr(y, op)(vrhs, **kwargs)
    assert result.data_ptr() == address
    torch.testing.assert_close(b.cpu(), a, rtol=3e-6, atol=2e-6)


@pytest.mark.parametrize('op', ['sub_', 'mul_'])
def test_mutation_arithmetic_errors_fallbacks_and_nonfinite(op):
    for device in ['cpu', 'vulkan']:
        x = torch.arange(12, dtype=torch.float32).to(device) + 1
        with pytest.raises(RuntimeError):
            getattr(x[1:], op)(x[:-1])
        x = torch.ones(1, 3).to(device)
        with pytest.raises(RuntimeError):
            getattr(x, op)(torch.ones(2, 3).to(device))
        x = torch.ones(1).to(device).expand(3)
        with pytest.raises(RuntimeError):
            getattr(x, op)(2.)
    for dtype in [torch.float32, torch.float64, torch.int64]:
        a = torch.arange(48, dtype=dtype).reshape(6, 8) + 1
        b = a.to('vulkan')
        rhs = torch.tensor([[2, 3, 4, 5]], dtype=dtype)
        getattr(a[1::2, ::2], op)(rhs)
        getattr(b[1::2, ::2], op)(rhs.to('vulkan'))
        torch.testing.assert_close(b.cpu(), a)
    a = torch.tensor([-float('inf'), -1., -0., 0., 1., float('inf'), float('nan')])
    for scalar in [0., float('inf'), float('nan')]:
        x, y = a.clone(), a.to('vulkan')
        getattr(x, op)(scalar)
        getattr(y, op)(scalar)
        torch.testing.assert_close(y.cpu(), x, equal_nan=True)


def test_subtract_bool_and_alpha_errors():
    for device in ['cpu', 'vulkan']:
        for rhs in [True, torch.tensor(True), torch.tensor([True]).to(device)]:
            x = torch.ones(1).to(device)
            with pytest.raises((RuntimeError, NotImplementedError)):
                x.sub_(rhs)
            with pytest.raises((RuntimeError, NotImplementedError)):
                torch.sub(x, rhs)
        for alpha in [True, 1j]:
            x = torch.ones(2).to(device)
            with pytest.raises(RuntimeError):
                x.sub_(2, alpha=alpha)


@pytest.mark.parametrize('op', ['sub_', 'mul_'])
def test_mutation_arithmetic_autograd(op):
    torch.manual_seed(14)
    a = torch.randn(3, 5, requires_grad=True)
    z = torch.randn(1, 5, requires_grad=True)
    b, w = (v.detach().to('vulkan').requires_grad_() for v in (a, z))
    x, y = a * 2, b * 2
    kwargs = {'alpha': 1.75} if op == 'sub_' else {}
    getattr(x, op)(z, **kwargs)
    getattr(y, op)(w, **kwargs)
    g = torch.randn_like(x)
    x.backward(g)
    y.backward(g.to('vulkan'))
    torch.testing.assert_close(y.cpu(), x)
    torch.testing.assert_close(b.grad.cpu(), a.grad)
    torch.testing.assert_close(w.grad.cpu(), z.grad)
