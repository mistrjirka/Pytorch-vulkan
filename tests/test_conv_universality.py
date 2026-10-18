"""Conv2D universality checks inspired by PyTorch/cuDNN and ggml Vulkan dispatch.

The correctness-first FP32 NCHW implementation should not accidentally become
specialized to square kernels, stride=1, groups=1, or channel multiplier=1.
All tests run CPU and Vulkan forward plus dInput/dWeight/dBias in autograd.
"""
import pytest
import torch
import torch.nn.functional as F
import torch_vulkan


@pytest.mark.parametrize("algorithm", ["direct", "optimized", "auto"])
@pytest.mark.parametrize("xshape,cout,kernel,stride,padding,dilation,groups", [
    ((2, 9, 17, 19), 19, (1, 1), (1, 1), (0, 0), (1, 1), 1),
    ((2, 9, 17, 19), 19, (5, 5), (2, 1), (3, 2), (2, 1), 1),
    ((1, 2, 14, 15), 3, (3, 5), (2, 1), (1, 2), (1, 2), 1),
    ((1, 1, 13, 14), 2, (7, 7), (1, 1), (3, 3), (1, 1), 1),
    ((1, 4, 11, 12), 12, (3, 3), (1, 1), (1, 1), (1, 1), 4),
    ((2, 6, 9, 10), 9, (5, 1), (1, 2), (2, 0), (1, 1), 3),
])
def test_conv2d_shape_semantics_and_backward(xshape, cout, kernel, stride,
                                               padding, dilation, groups, algorithm, monkeypatch):
    monkeypatch.setenv("TORCH_VULKAN_CONV_BACKWARD_ALGO", algorithm)
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


@pytest.mark.parametrize("layout", ["transpose", "channels_last", "slice", "offset", "broadcast"])
@pytest.mark.parametrize("algorithm", ["direct", "optimized"])
def test_conv2d_strided_views_forward_and_all_gradients(layout, algorithm, monkeypatch):
    monkeypatch.setenv("TORCH_VULKAN_CONV_BACKWARD_ALGO", algorithm)
    torch.manual_seed(700)
    def view(t):
        if layout == "transpose": return t.transpose(2, 3)
        if layout == "channels_last": return t.contiguous(memory_format=torch.channels_last)
        if layout == "slice": return t[:, :, 1::2, ::2]
        if layout == "offset": return t[1:]
        return t[:1].expand(3, -1, -1, -1)
    base = torch.randn(3, 6, 13, 15)
    weight_base = torch.randn(10, 3, 3, 5)
    bias_base = torch.randn(20)
    # Apply views after device transfer so actual GPU strides/offsets survive.
    x, xv = view(base), view(base.to("vulkan"))
    w, wv = weight_base.transpose(2, 3), weight_base.to("vulkan").transpose(2, 3)
    b, bv = bias_base[1::2], bias_base.to("vulkan")[1::2]
    args = ([1, 2], [2, 1], [1, 1], 2)
    ref = F.conv2d(x, w, b, *args)
    out = F.conv2d(xv, wv, bv, *args)
    torch.testing.assert_close(out.cpu(), ref, atol=4e-4, rtol=3e-4)
    grad_base = torch.randn(*ref.shape[:-1], ref.shape[-1] * 2)
    grad, gradv = grad_base[..., 1::2], grad_base.to("vulkan")[..., 1::2]
    def backward(go, inp, wt, mask):
        return torch.ops.aten.convolution_backward.default(
            go, inp, wt, [10], args[0], args[1], args[2], False, [0, 0], 2, mask)
    for mask in ([True, True, True], [True, False, False], [False, True, False], [False, False, True]):
        expected = backward(grad, x, w, mask)
        actual = backward(gradv, xv, wv, mask)
        for enabled, a, e in zip(mask, actual, expected):
            if enabled:
                torch.testing.assert_close(a.cpu(), e, atol=5e-4, rtol=4e-4)
            else:
                assert a is None


def test_conv2d_sum_backward_broadcast_gradient():
    torch.manual_seed(19)
    x = torch.randn(2, 3, 9, 11, requires_grad=True)
    w = torch.randn(8, 3, 3, 3, requires_grad=True)
    xv = x.detach().to("vulkan").requires_grad_()
    wv = w.detach().to("vulkan").requires_grad_()
    F.conv2d(x, w, padding=1).sum().backward()
    F.conv2d(xv, wv, padding=1).sum().backward()
    for a, e in ((xv.grad, x.grad), (wv.grad, w.grad)):
        torch.testing.assert_close(a.cpu(), e, atol=5e-4, rtol=4e-4)


def test_conv2d_effective_kernel_larger_than_input_rejected():
    # A -1 numerator must not truncate to zero under C++ signed division.
    x = torch.randn(1, 2, 2, 2).to("vulkan")
    w = torch.randn(3, 2, 3, 3).to("vulkan")
    with pytest.raises(RuntimeError):
        F.conv2d(x, w, stride=2)
