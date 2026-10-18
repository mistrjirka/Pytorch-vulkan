"""Read-only constants must survive queued reuse and remain separate from user data."""
import pytest
import torch
import torch.nn.functional as F
import torch_vulkan


@pytest.mark.parametrize("algorithm", ["direct", "optimized", "auto"])
@pytest.mark.parametrize("cout,groups,kernel,stride,dilation,layout", [
    (1, 1, (1, 1), (1, 1), (1, 1), "plain"),
    (10, 2, (3, 5), (2, 1), (1, 2), "transpose"),
    (11, 1, (5, 3), (1, 2), (2, 1), "offset"),
    (16, 4, (3, 3), (1, 1), (1, 1), "channels_last"),
    (17, 1, (1, 3), (1, 1), (1, 1), "slice"),
    (33, 1, (3, 1), (1, 1), (1, 1), "plain"),
])
def test_biasless_conv_repeated_forward_and_gradients(
        algorithm, cout, groups, kernel, stride, dilation, layout, monkeypatch):
    monkeypatch.setenv("TORCH_VULKAN_CONV_BACKWARD_ALGO", algorithm)
    torch.manual_seed(601 + cout)
    base = torch.randn(2, 4, 13, 15)
    def view(x):
        if layout == "transpose": return x.transpose(2, 3)
        if layout == "offset": return x[1:]
        if layout == "slice": return x[:, :, 1::2, ::2]
        if layout == "channels_last": return x.contiguous(memory_format=torch.channels_last)
        return x
    x = view(base).detach().requires_grad_()
    xv = view(base.to("vulkan")).detach().requires_grad_()
    w = torch.randn(cout, 4 // groups, *kernel, requires_grad=True)
    wv = w.detach().to("vulkan").requires_grad_()
    b = torch.randn(cout, requires_grad=True)
    bv = b.detach().to("vulkan").requires_grad_()
    args = (stride, (4, 4), dilation, groups)
    refs = [F.conv2d(x, w, bias, *args) for bias in (None, b, None)]
    grads = [torch.randn_like(y) for y in refs]
    gradsv = [g.to("vulkan") for g in grads]
    expected = [torch.autograd.grad(y, (x, w) if i != 1 else (x, w, b), g)
                for i, (y, g) in enumerate(zip(refs, grads))]
    # Enqueue both uses of the default bias, interleaved with an explicit bias.
    outs = [F.conv2d(xv, wv, bias, *args) for bias in (None, bv, None)]
    actual = [torch.autograd.grad(y, (xv, wv) if i != 1 else (xv, wv, bv), g)
              for i, (y, g) in enumerate(zip(outs, gradsv))]
    for out, ref, got_grads, ref_grads in zip(outs, refs, actual, expected):
        torch.testing.assert_close(out.cpu(), ref, rtol=4e-4, atol=5e-4)
        for got, wanted in zip(got_grads, ref_grads):
            torch.testing.assert_close(got.cpu(), wanted, rtol=5e-4, atol=6e-4)


def test_zero_and_one_constants_queued_across_operators_and_capacities():
    torch.manual_seed(621)
    cases = []
    for count in (10, 11, 16, 17, 31, 33, 10):
        x = torch.randn(1, 2, 3, 5, requires_grad=True)
        w = torch.randn(count, 2, 1, 1, requires_grad=True)
        xv = x.detach().to("vulkan").requires_grad_()
        wv = w.detach().to("vulkan").requires_grad_()
        # LayerNorm of the channel dimension reuses the convolution's zero
        # capacity bucket while also requiring a distinct buffer of ones.
        ref = F.conv2d(x, w).permute(0, 2, 3, 1)
        ref = F.layer_norm(ref, (count,), eps=0.003)
        grad = torch.randn_like(ref)
        gradv = grad.to("vulkan")
        expected = torch.autograd.grad(ref, (x, w), grad)
        cases.append((count, xv, wv, gradv, ref, expected))
    results = []
    for count, xv, wv, gradv, ref, expected in cases:
        out = F.conv2d(xv, wv).permute(0, 2, 3, 1)
        out = F.layer_norm(out, (count,), eps=0.003)
        got = torch.autograd.grad(out, (xv, wv), gradv)
        results.append((out, got, ref, expected))
    for out, got, ref, expected in results:
        torch.testing.assert_close(out.cpu(), ref, rtol=4e-4, atol=5e-4)
        for actual, wanted in zip(got, expected):
            torch.testing.assert_close(actual.cpu(), wanted, rtol=5e-4, atol=7e-4)


@pytest.mark.parametrize("count", [65535, 65536, 65537])
def test_layer_norm_constant_cache_boundary_forward(count):
    # Above the bounded cache, the existing temporary path must stay correct.
    # LayerNorm backward currently supports at most 65535 elements per row.
    torch.manual_seed(622)
    x = torch.randn(2, count)
    xv = x.to("vulkan")
    wanted = F.layer_norm(x, (count,), eps=0.003)
    outs = [F.layer_norm(xv, (count,), eps=0.003) for _ in range(3)]
    for out in outs:
        torch.testing.assert_close(out.cpu(), wanted, rtol=4e-4, atol=5e-4)
