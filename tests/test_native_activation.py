"""FP32 LeakyReLU preserves runtime slopes, gradients, views and mutation."""
import os
import pytest
import torch
import torch.nn.functional as F
import torch_vulkan


def view(x, layout):
    if layout == "transpose":
        return x.t()
    if layout == "offset":
        return x[1:, 1:]
    if layout == "stepped":
        return x[:, ::2]
    return x


def exact(actual, expected):
    actual = actual.cpu()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    zeros = expected == 0
    assert torch.equal(torch.signbit(actual[zeros]), torch.signbit(expected[zeros]))


@pytest.mark.parametrize("slope", [0.01, 0., -0., -.2, 2., float("inf"), -float("inf"), float("nan")])
def test_leaky_special_values_and_backward(slope):
    x = torch.tensor([-float("inf"), -3., -0., 0., 2., float("inf"), float("nan")])
    g = torch.tensor([-2., 3., -0., 0., -4., float("inf"), float("nan")])
    vx, vg = x.to("vulkan"), g.to("vulkan")
    exact(F.leaky_relu(vx, slope), F.leaky_relu(x, slope))
    exact(torch.ops.aten.leaky_relu_backward(vg, vx, slope, False),
          torch.ops.aten.leaky_relu_backward(g, x, slope, False))
    ci, vi = x.clone(), vx.clone()
    F.leaky_relu(ci, slope, inplace=True)
    F.leaky_relu(vi, slope, inplace=True)
    exact(vi, ci)


@pytest.mark.parametrize("layout", ["contiguous", "transpose", "offset", "stepped"])
@pytest.mark.parametrize("inplace", [False, True])
@pytest.mark.parametrize("slope", [.2, -.3])
def test_leaky_layouts_and_alias(layout, inplace, slope):
    cpu = torch.arange(63, dtype=torch.float32).reshape(7, 9) - 30
    gpu = cpu.to("vulkan")
    a, b = view(cpu, layout), view(gpu, layout)
    ptr = b.data_ptr()
    expected = F.leaky_relu(a, slope, inplace=inplace)
    actual = F.leaky_relu(b, slope, inplace=inplace)
    exact(actual, expected)
    if inplace:
        assert actual.data_ptr() == ptr
    exact(gpu, cpu)
    gbase = torch.arange(63, dtype=torch.float32).reshape(7, 9) / 16
    g = view(gbase, layout)
    vg = view(gbase.to("vulkan"), layout)
    exact(torch.ops.aten.leaky_relu_backward(vg, b, slope, False),
          torch.ops.aten.leaky_relu_backward(g, a, slope, False))


@pytest.mark.parametrize("shape", [(), (0, 3), (2, 1, 1, 1, 1, 1, 2, 3),
                                    (2, 1, 1, 1, 1, 1, 1, 2, 3)])
def test_leaky_rank_and_empty(shape):
    x = torch.full(shape, -2.)
    g = torch.full(shape, 3.)
    vx, vg = x.to("vulkan"), g.to("vulkan")
    exact(F.leaky_relu(vx, .25), F.leaky_relu(x, .25))
    exact(torch.ops.aten.leaky_relu_backward(vg, vx, .25, False),
          torch.ops.aten.leaky_relu_backward(g, x, .25, False))


@pytest.mark.parametrize("layout", ["contiguous", "transpose", "offset", "stepped"])
def test_leaky_out_views_preserve_storage(layout):
    dest = torch.full((7, 9), 77.)
    vdest = dest.to("vulkan")
    out, vout = view(dest, layout), view(vdest, layout)
    x = torch.arange(out.numel(), dtype=torch.float32).reshape(out.shape) - 17
    vx = x.to("vulkan")
    ptr = vout.data_ptr()
    torch.ops.aten.leaky_relu.out(x, .2, out=out)
    returned = torch.ops.aten.leaky_relu.out(vx, .2, out=vout)
    assert returned.data_ptr() == ptr
    exact(vdest, dest)
    g = torch.full_like(x, -2.)
    torch.ops.aten.leaky_relu_backward.grad_input(g, x, -.3, False, grad_input=out)
    torch.ops.aten.leaky_relu_backward.grad_input(g.to("vulkan"), vx, -.3, False, grad_input=vout)
    exact(vdest, dest)


def test_leaky_out_resize_and_dtype_fallback():
    for dtype in [torch.float32, torch.float64]:
        x = torch.tensor([-2., 0., 3.], dtype=dtype)
        a = torch.empty(0, dtype=dtype)
        b = a.to("vulkan")
        torch.ops.aten.leaky_relu.out(x, .3, out=a)
        torch.ops.aten.leaky_relu.out(x.to("vulkan"), .3, out=b)
        exact(b, a)
    for device in ["cpu", "vulkan"]:
        with pytest.raises(RuntimeError):
            torch.ops.aten.leaky_relu.out(torch.ones(3).to(device), .3,
                                         out=torch.empty(3, dtype=torch.float64).to(device))


@pytest.mark.parametrize("slope", [-.3, float("nan")])
def test_leaky_result_backward_rejects_invalid_slope(slope):
    for device in ["cpu", "vulkan"]:
        x = torch.ones(3).to(device)
        with pytest.raises(RuntimeError, match="negative slope"):
            torch.ops.aten.leaky_relu_backward(x, x, slope, True)


@pytest.mark.parametrize("inplace", [False, True])
def test_leaky_autograd(inplace):
    torch.manual_seed(122)
    cpu = torch.randn(5, 7, requires_grad=True)
    vk = cpu.detach().to("vulkan").requires_grad_()
    a, b = cpu * 2, vk * 2
    # Positive slope supports both saved-input and saved-result differentiation.
    ya = F.leaky_relu(a, .13, inplace=inplace)
    yb = F.leaky_relu(b, .13, inplace=inplace)
    g = torch.randn_like(cpu)
    ya.backward(g)
    yb.backward(g.to("vulkan"))
    exact(yb, ya.detach())
    exact(vk.grad, cpu.grad)


def test_leaky_overlap_errors_and_expanded_read():
    for device in ["cpu", "vulkan"]:
        x = torch.tensor([-2.]).to(device).expand(7)
        with pytest.raises(RuntimeError):
            F.leaky_relu(x, .2, inplace=True)
        base = torch.arange(10, dtype=torch.float32).to(device)
        with pytest.raises(RuntimeError):
            torch.ops.aten.leaky_relu.out(base[:-1], .2, out=base[1:])
    x = torch.tensor([-2., 3.]).reshape(1, 2)
    exact(F.leaky_relu(x.to("vulkan").expand(5, 2), .2), F.leaky_relu(x.expand(5, 2), .2))


def test_leaky_out_exact_alias():
    x = torch.linspace(-3., 3., 129)
    v = x.to("vulkan")
    torch.ops.aten.leaky_relu.out(x, -.5, out=x)
    torch.ops.aten.leaky_relu.out(v, -.5, out=v)
    exact(v, x)
    g = torch.linspace(-2., 2., 129)
    vg = g.to("vulkan")
    torch.ops.aten.leaky_relu_backward.grad_input(g, x, .3, False, grad_input=g)
    torch.ops.aten.leaky_relu_backward.grad_input(vg, v, .3, False, grad_input=vg)
    exact(vg, g)


def test_leaky_queued_chain_stays_on_gpu():
    if not all(torch_vulkan.performance_options()[k] for k in ("gpu_resident", "deferred_submissions")):
        pytest.skip("requires resident deferred mode")
    x = torch.linspace(-3., 3., 257)
    v = x.to("vulkan")
    torch_vulkan.synchronize()
    before = torch_vulkan.queue_stats()["submissions"]
    for _ in range(12):
        x = F.leaky_relu(x, -.25)
        v = F.leaky_relu(v, -.25)
    assert torch_vulkan.queue_stats()["submissions"] == before
    exact(v, x)


def test_leaky_backward_broadcast_and_resize_fallback():
    x = torch.arange(6, dtype=torch.float32).reshape(2, 3) - 2
    g = torch.ones(1, 3)
    expected = torch.ops.aten.leaky_relu_backward(g, x, .2, False)
    vx, vg = x.to('vulkan'), g.to('vulkan')
    exact(torch.ops.aten.leaky_relu_backward(vg, vx, .2, False), expected)
    out = torch.empty(0).to('vulkan')
    torch.ops.aten.leaky_relu_backward.grad_input(vg, vx, .2, False, grad_input=out)
    exact(out, expected)


@pytest.mark.parametrize('slope', [0., -0., .2, float('inf')])
def test_leaky_saved_result_backward(slope):
    result = torch.tensor([-3., -0., 0., 2., float('nan')])
    grad = torch.tensor([2., -1., 1., -2., 3.])
    exact(torch.ops.aten.leaky_relu_backward(grad.to('vulkan'), result.to('vulkan'), slope, True),
          torch.ops.aten.leaky_relu_backward(grad, result, slope, True))
