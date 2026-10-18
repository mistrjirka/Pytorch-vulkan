"""Exercise batched buffer dependencies with aliases, fan-out and queued gradients."""
import pytest
import torch
import torch_vulkan


@pytest.mark.parametrize('width', [7, 257, 2049])
def test_queued_fanin_fanout_and_gradients(width):
    torch.manual_seed(940 + width)
    x = torch.randn(3, width, requires_grad=True)
    y = torch.randn(3, width, requires_grad=True)
    xv = x.detach().to('vulkan').requires_grad_()
    yv = y.detach().to('vulkan').requires_grad_()
    grad = torch.randn_like(x)
    gradv = grad.to('vulkan')
    def graph(a, b):
        for _ in range(24):
            tmp = a * 0.5 + b * 0.125
            b = torch.relu(b + tmp * 0.01)
            a = tmp + tmp  # Repeated VkBuffer binding in a single dispatch.
        return a + b * 0.25
    ref = graph(x, y)
    expected = torch.autograd.grad(ref, (x, y), grad)
    out = graph(xv, yv)
    got = torch.autograd.grad(out, (xv, yv), gradv)
    torch.testing.assert_close(out.cpu(), ref, rtol=3e-4, atol=3e-5)
    for actual, wanted in zip(got, expected):
        torch.testing.assert_close(actual.cpu(), wanted, rtol=3e-4, atol=3e-5)


@pytest.mark.parametrize('width', [7, 257, 2049])
def test_queued_inplace_alias_and_host_read_boundary(width):
    torch.manual_seed(950 + width)
    state = torch.randn(3, width)
    other = torch.randn(3, width)
    statev, otherv = state.to('vulkan'), other.to('vulkan')
    refs, outs = [], []
    for _ in range(48):
        # Both read-only and writable references alias the same storage.
        refs.append(torch.relu(state + state))
        outs.append(torch.relu(statev + statev))
        state.mul_(0.5).add_(other)
        statev.mul_(0.5).add_(otherv)
    # Reading one early result flushes all queued users, including later writes.
    torch.testing.assert_close(outs[0].cpu(), refs[0], rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(statev.cpu(), state, rtol=2e-5, atol=2e-5)
    for index in (1, 23, 47):
        torch.testing.assert_close(outs[index].cpu(), refs[index], rtol=2e-5, atol=2e-5)
