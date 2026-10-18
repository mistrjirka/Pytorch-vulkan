"""Native FP32 Vulkan single-axis reductions and autograd."""
import pytest
import torch
import torch_vulkan


@pytest.mark.parametrize("shape,axis,keepdim", [
    ((128, 1024), 0, False),
    ((128, 1024), 0, True),
    ((128, 1024), -1, False),
    ((3, 7, 19), 1, True),
    ((3, 7, 19), 2, False),
    ((513,), 0, False),
    ((2, 3, 5, 7), -3, False),
])
def test_native_single_axis_sum_matches_cpu(shape, axis, keepdim):
    assert torch._C._dispatch_has_kernel_for_dispatch_key(
        "aten::sum.dim_IntList", "PrivateUse1")
    torch.manual_seed(910 + len(shape))
    x_cpu = torch.randn(*shape, requires_grad=True)
    x_vk = x_cpu.detach().to("vulkan").requires_grad_()
    reference = x_cpu.sum(dim=axis, keepdim=keepdim)
    actual = x_vk.sum(dim=axis, keepdim=keepdim)
    assert actual.device.type == "vulkan"
    assert actual.shape == reference.shape
    torch.testing.assert_close(actual.cpu(), reference, atol=7e-5, rtol=3e-4)
    grad_cpu = torch.randn_like(reference)
    expected = torch.autograd.grad(reference, x_cpu, grad_cpu)[0]
    got = torch.autograd.grad(actual, x_vk, grad_cpu.to("vulkan"))[0]
    torch.testing.assert_close(got.cpu(), expected, atol=2e-5, rtol=3e-5)


def test_multiple_axis_sum_preserves_cpu_semantics():
    x_cpu = torch.randn(2, 3, 4, requires_grad=True)
    x_vk = x_cpu.detach().to("vulkan").requires_grad_()
    expected = x_cpu.sum(dim=(0, 2), keepdim=True)
    got = x_vk.sum(dim=(0, 2), keepdim=True)
    torch.testing.assert_close(got.cpu(), expected)
    grad_cpu = torch.randn_like(expected)
    expected_grad = torch.autograd.grad(expected, x_cpu, grad_cpu)[0]
    actual_grad = torch.autograd.grad(got, x_vk, grad_cpu.to("vulkan"))[0]
    torch.testing.assert_close(actual_grad.cpu(), expected_grad)


def test_noncontiguous_sum_preserves_semantics():
    x = torch.randn(6, 9)
    vk = x.to("vulkan")
    expected = x.t().sum(dim=1)
    got = vk.t().sum(dim=1)
    torch.testing.assert_close(got.cpu(), expected)
