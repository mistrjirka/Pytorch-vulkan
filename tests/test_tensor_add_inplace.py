"""Tensor accumulation keeps storage, views, autograd versions and queued work."""
import os

import pytest
import torch
import torch_vulkan


def view(x, layout):
    if layout == "transpose":
        return x.t()
    if layout == "offset":
        return x[1:, 1:]
    if layout == "stepped":
        return x[:, ::2]
    return x


@pytest.mark.parametrize("layout", ["contiguous", "transpose", "offset", "stepped"])
@pytest.mark.parametrize("rhs_kind", ["same_shape", "row", "scalar", "strided"])
def test_tensor_add_inplace_views_and_broadcast(layout, rhs_kind):
    torch.manual_seed(147)
    cpu = torch.randn(7, 9)
    gpu = cpu.to("vulkan")
    a, b = view(cpu, layout), view(gpu, layout)
    if rhs_kind == "same_shape":
        rhs = torch.randn(a.shape)
    elif rhs_kind == "row":
        rhs = torch.randn(1, a.shape[1])
    elif rhs_kind == "scalar":
        rhs = torch.randn(())
    else:
        source = torch.randn(a.shape[1] * 2, a.shape[0] + 1)
        rhs = source[::2, 1:].t()
    # Transfer the base before slicing so the Vulkan RHS has real strides/offset.
    if rhs_kind == "strided":
        vg = source.to("vulkan")[::2, 1:].t()
    else:
        vg = rhs.to("vulkan")
    pointer, version, strides, offset = b.data_ptr(), b._version, b.stride(), b.storage_offset()
    expected = a.add_(rhs, alpha=-0.375)
    result = b.add_(vg, alpha=-0.375)
    assert result is b
    assert b.data_ptr() == pointer and b.stride() == strides and b.storage_offset() == offset
    assert b._version == version + 1
    torch.testing.assert_close(result.cpu(), expected)
    torch.testing.assert_close(gpu.cpu(), cpu)


@pytest.mark.parametrize("shape,rhs_shape", [
    ((), ()),
    ((0, 3), (1, 3)),
    ((2, 3, 4), (3, 1)),
    ((2, 1, 1, 3, 1, 1, 2, 4), (1, 1, 3, 1, 1, 1, 4)),
    ((2, 1, 1, 1, 3, 1, 1, 2, 4), (4,)),  # rank-nine fallback
])
def test_tensor_add_inplace_ranks(shape, rhs_shape):
    a, rhs = torch.randn(shape), torch.randn(rhs_shape)
    b, vg = a.to("vulkan"), rhs.to("vulkan")
    a.add_(rhs, alpha=2.25)
    b.add_(vg, alpha=2.25)
    torch.testing.assert_close(b.cpu(), a)


@pytest.mark.parametrize("layout", ["contiguous", "transpose", "offset", "stepped"])
@pytest.mark.parametrize("separate_view", [False, True])
def test_tensor_add_inplace_exact_alias(layout, separate_view):
    a = torch.arange(63, dtype=torch.float32).reshape(7, 9)
    b = a.to("vulkan")
    av, bv = view(a, layout), view(b, layout)
    ar = av.as_strided(av.shape, av.stride(), av.storage_offset()) if separate_view else av
    br = bv.as_strided(bv.shape, bv.stride(), bv.storage_offset()) if separate_view else bv
    av.add_(ar, alpha=-0.25)
    bv.add_(br, alpha=-0.25)
    torch.testing.assert_close(b.cpu(), a)


@pytest.mark.parametrize("kind", ["partial", "transpose", "internal"])
def test_tensor_add_inplace_overlap_rejection(kind):
    for device in ("cpu", "vulkan"):
        x = torch.arange(16, dtype=torch.float32).reshape(4, 4).to(device)
        if kind == "partial":
            a, b = x.flatten()[1:], x.flatten()[:-1]
        elif kind == "transpose":
            a, b = x, x.t()
        else:
            a, b = x[:1, :].expand(4, 4), torch.ones(4, 4).to(device)
        with pytest.raises(RuntimeError, match="single memory location"):
            a.add_(b)


@pytest.mark.parametrize("dtype", [torch.float64, torch.int64])
def test_tensor_add_inplace_dtype_fallback(dtype):
    a = torch.arange(15, dtype=dtype).reshape(3, 5)
    rhs = torch.arange(5, dtype=dtype)
    b = a.to("vulkan")
    a.add_(rhs, alpha=-2)
    b.add_(rhs.to("vulkan"), alpha=-2)
    torch.testing.assert_close(b.cpu(), a, rtol=0, atol=0)


@pytest.mark.parametrize("shape,rhs_shape,alpha", [
    ((2, 3), (4,), 1),
    ((1, 3), (2, 3), 1),
    ((2, 3), (2, 3), True),
    ((2, 3), (2, 3), 1j),
])
def test_tensor_add_inplace_invalid_inputs(shape, rhs_shape, alpha):
    for device in ("cpu", "vulkan"):
        a, b = torch.ones(shape).to(device), torch.ones(rhs_shape).to(device)
        with pytest.raises(RuntimeError):
            a.add_(b, alpha=alpha)


def test_tensor_add_inplace_autograd_accumulation_and_version():
    torch.manual_seed(13)
    a = torch.randn(4, 7, requires_grad=True)
    b = a.detach().to("vulkan").requires_grad_()
    rhs = torch.randn(1, 7, requires_grad=True)
    vrhs = rhs.detach().to("vulkan").requires_grad_()
    x, y = a * 2, b * 2
    x.add_(rhs, alpha=-0.25)
    y.add_(vrhs, alpha=-0.25)
    # Shared leaf participates twice, requiring native tensor gradient accumulation.
    (x.sum() + a.sum()).backward()
    (y.sum() + b.sum()).backward()
    torch.testing.assert_close(b.grad.cpu(), a.grad)
    torch.testing.assert_close(vrhs.grad.cpu(), rhs.grad)
    for device in ("cpu", "vulkan"):
        leaf = torch.ones(4).to(device).requires_grad_()
        with pytest.raises(RuntimeError, match="leaf Variable"):
            leaf.add_(torch.ones(4).to(device))


def test_tensor_add_inplace_queued_rhs_lifetime():
    if not all(torch_vulkan.performance_options()[k] for k in ("gpu_resident", "deferred_submissions")):
        return
    torch.manual_seed(17)
    cpu = torch.randn(17, 19)
    gpu = cpu.to("vulkan")
    rhs_cpu = [torch.randn(17, 19) for _ in range(12)]
    rhs_gpu = [x.to("vulkan") for x in rhs_cpu]
    torch_vulkan.synchronize()
    before = torch_vulkan.queue_stats()["submissions"]
    for c, g in zip(rhs_cpu, rhs_gpu):
        # Produced RHS can be destroyed before deferred commands execute.
        gpu.add_(torch.relu(g), alpha=0.5)
        cpu.add_(torch.relu(c), alpha=0.5)
    del rhs_gpu
    assert torch_vulkan.queue_stats()["submissions"] == before
    torch.testing.assert_close(gpu.cpu(), cpu)
