"""Conv2D universality checks inspired by PyTorch/cuDNN and ggml Vulkan dispatch.

The correctness-first FP32 NCHW implementation should not accidentally become
specialized to square kernels, stride=1, groups=1, or channel multiplier=1.
All tests run CPU and Vulkan forward plus dInput/dWeight/dBias in autograd.
"""
import pytest
import torch
import torch.nn.functional as F
import torch_vulkan


@pytest.mark.parametrize("xshape,cout,kernel,stride,padding,dilation,groups", [
    ((1, 2, 14, 15), 3, (3, 5), (2, 1), (1, 2), (1, 2), 1),
    ((1, 1, 13, 14), 2, (7, 7), (1, 1), (3, 3), (1, 1), 1),
    ((1, 4, 11, 12), 12, (3, 3), (1, 1), (1, 1), (1, 1), 4),
    ((2, 6, 9, 10), 9, (5, 1), (1, 2), (2, 0), (1, 1), 3),
])
def test_conv2d_shape_semantics_and_backward(xshape, cout, kernel, stride,
                                               padding, dilation, groups):
    torch.manual_seed(sum(xshape) + cout + sum(kernel))
    x = torch.randn(*xshape, requires_grad=True)
    w = torch.randn(cout, xshape[1] // groups, *kernel, requires_grad=True)
    b = torch.randn(cout, requires_grad=True)
    x_vk = x.detach().to("vulkan").requires_grad_()
    w_vk = w.detach().to("vulkan").requires_grad_()
    b_vk = b.detach().to("vulkan").requires_grad_()

    cpu = F.conv2d(x, w, b, stride, padding, dilation, groups)
    vk = F.conv2d(x_vk, w_vk, b_vk, stride, padding, dilation, groups)
    assert vk.shape == cpu.shape
    assert vk.device.type == "vulkan"
    torch.testing.assert_close(vk.cpu(), cpu, atol=4e-4, rtol=3e-4)
    grad = torch.randn_like(cpu)
    expected = torch.autograd.grad(cpu, (x, w, b), grad_outputs=grad)
    actual = torch.autograd.grad(vk, (x_vk, w_vk, b_vk),
                                 grad_outputs=grad.to("vulkan"))
    for o, ref in zip(actual, expected):
        assert o.device.type == "vulkan"
        torch.testing.assert_close(o.cpu(), ref, atol=5e-4, rtol=4e-4)


def test_conv2d_unsupported_strided_view_fails_loudly():
    base = torch.randn(1, 2, 9, 9).to("vulkan")
    weight = torch.randn(3, 2, 3, 3).to("vulkan")
    x = base.transpose(2, 3)
    assert not x.is_contiguous()
    with pytest.raises(RuntimeError, match="contiguous"):
        F.conv2d(x, weight, None, padding=1)
