"""Numerical and autograd tests for native Vulkan log-softmax."""
import pytest
import torch
import torch.nn.functional as F
import torch_vulkan


@pytest.mark.parametrize("shape", [
    (1, 10),
    (128, 10),
    (16, 257),
    (7, 1024),
    (2, 3, 4096),
])
def test_last_axis_log_softmax_forward_backward(shape, monkeypatch):
    # Small tensors default to faster CPU fallback; force GPU execution so
    # the actual shader is tested across all input sizes and layouts.
    monkeypatch.setenv("TORCH_VULKAN_NATIVE_LOGSOFTMAX", "1")
    assert torch._C._dispatch_has_kernel_for_dispatch_key(
        "aten::_log_softmax", "PrivateUse1")
    assert torch._C._dispatch_has_kernel_for_dispatch_key(
        "aten::_log_softmax_backward_data", "PrivateUse1")
    torch.manual_seed(200+shape[-1])
    cpu = torch.randn(*shape, requires_grad=True)
    vulkan = cpu.detach().to("vulkan").requires_grad_()
    ref = F.log_softmax(cpu, dim=-1)
    actual = F.log_softmax(vulkan, dim=-1)
    torch.testing.assert_close(actual.cpu(), ref, atol=2e-5, rtol=3e-5)
    grad = torch.randn_like(cpu)
    reference_grad = torch.autograd.grad(ref,cpu,grad_outputs=grad)[0]
    vulkan_grad = torch.autograd.grad(actual,vulkan,grad_outputs=grad.to("vulkan"))[0]
    torch.testing.assert_close(vulkan_grad.cpu(),reference_grad,atol=5e-5,rtol=5e-5)


def test_non_last_dim_routes_through_correct_cpu_fallback():
    torch.manual_seed(909)
    cpu = torch.randn(4, 6, 3, requires_grad=True)
    vulkan = cpu.detach().to("vulkan").requires_grad_()
    expected = F.log_softmax(cpu,dim=1)
    got = F.log_softmax(vulkan,dim=1)
    torch.testing.assert_close(got.cpu(),expected)
    grad = torch.randn_like(expected)
    expected_dx = torch.autograd.grad(expected,cpu,grad_outputs=grad)[0]
    got_dx = torch.autograd.grad(got,vulkan,grad_outputs=grad.to("vulkan"))[0]
    torch.testing.assert_close(got_dx.cpu(),expected_dx)


def test_cross_entropy_backward_vulkan():
    torch.manual_seed(811)
    x_cpu = torch.randn(32, 10, requires_grad=True)
    x_vk = x_cpu.detach().to("vulkan").requires_grad_()
    labels = torch.randint(0,10,(32,))
    expected=F.cross_entropy(x_cpu,labels)
    actual=F.cross_entropy(x_vk,labels.to("vulkan"))
    torch.testing.assert_close(actual.cpu(),expected,atol=1e-5,rtol=1e-5)
    dx_cpu=torch.autograd.grad(expected,x_cpu)[0]
    dx_vk=torch.autograd.grad(actual,x_vk)[0]
    torch.testing.assert_close(dx_vk.cpu(),dx_cpu,atol=3e-5,rtol=3e-5)


def test_auto_logsoftmax_large_uses_correct_gradient():
    # Larger than the 32768-element native crossover threshold.
    torch.manual_seed(3333)
    x = torch.randn(32, 2048, requires_grad=True)
    v = x.detach().to("vulkan").requires_grad_()
    cpu_out = torch.nn.functional.log_softmax(x, -1)
    vk_out = torch.nn.functional.log_softmax(v, -1)
    torch.testing.assert_close(vk_out.cpu(), cpu_out, atol=3e-5, rtol=3e-5)
    g = torch.randn_like(cpu_out)
    expected = torch.autograd.grad(cpu_out, x, g)[0]
    actual = torch.autograd.grad(vk_out, v, g.to("vulkan"))[0]
    torch.testing.assert_close(actual.cpu(), expected, atol=5e-5, rtol=5e-5)


def test_large_cross_entropy_gpu_logsoftmax_and_cpu_loss_boundary(monkeypatch):
    # Exercises true native Vulkan log-softmax, CPU NLL fallback, and the
    # gradient crossing that boundary while GPU residency is enabled.
    monkeypatch.setenv("TORCH_VULKAN_NATIVE_LOGSOFTMAX", "1")
    torch.manual_seed(913)
    x_cpu = torch.randn(64, 2048, requires_grad=True)
    x_vk = x_cpu.detach().to("vulkan").requires_grad_()
    targets = torch.randint(0, 2048, (64,), dtype=torch.long)
    expected = F.cross_entropy(x_cpu, targets)
    actual = F.cross_entropy(x_vk, targets.to("vulkan"))
    torch.testing.assert_close(actual.cpu(), expected, atol=2e-5, rtol=2e-5)
    grad_cpu = torch.autograd.grad(expected, x_cpu)[0]
    grad_vk = torch.autograd.grad(actual, x_vk)[0]
    torch.testing.assert_close(grad_vk.cpu(), grad_cpu, atol=5e-5, rtol=5e-5)
