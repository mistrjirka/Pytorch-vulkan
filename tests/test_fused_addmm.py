"""Fused GEMM+bias correctness and autograd regression for the Vulkan backend."""
import pytest
import torch
import torch_vulkan


@pytest.mark.parametrize("shape", [
    (5, 7, 11),
    (32, 64, 19),
    (128, 256, 128),
    (256, 512, 512),
])
@pytest.mark.parametrize("coefficients", [(1.0, 1.0), (0.25, 0.8), (-0.5, 1.5)])
def test_addmm_1d_bias_forward_backward(shape, coefficients):
    m, k, n = shape
    beta, alpha = coefficients
    torch.manual_seed(1234 + m + k + n)
    lhs_cpu = torch.randn(m, k, requires_grad=True)
    rhs_cpu = torch.randn(k, n, requires_grad=True)
    bias_cpu = torch.randn(n, requires_grad=True)
    lhs_gpu = lhs_cpu.detach().to("vulkan").requires_grad_()
    rhs_gpu = rhs_cpu.detach().to("vulkan").requires_grad_()
    bias_gpu = bias_cpu.detach().to("vulkan").requires_grad_()
    ref = torch.addmm(bias_cpu, lhs_cpu, rhs_cpu, beta=beta, alpha=alpha)
    got = torch.addmm(bias_gpu, lhs_gpu, rhs_gpu, beta=beta, alpha=alpha)
    assert got.device.type == "vulkan"
    torch.testing.assert_close(got.cpu(), ref, atol=3e-4, rtol=4e-4)
    grad = torch.randn_like(ref)
    expected = torch.autograd.grad(ref, (lhs_cpu, rhs_cpu, bias_cpu), grad_outputs=grad)
    actual = torch.autograd.grad(got, (lhs_gpu, rhs_gpu, bias_gpu),
                                 grad_outputs=grad.to("vulkan"))
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a.cpu(), b, atol=5e-4, rtol=5e-4)


def test_linear_with_bias_training_step():
    torch.manual_seed(706)
    import copy
    cpu = torch.nn.Linear(128, 64)
    gpu = copy.deepcopy(cpu).to("vulkan")
    x = torch.randn(16, 128)
    labels = torch.randn(16, 64)
    cpu_loss = torch.nn.functional.mse_loss(cpu(x), labels)
    cpu_loss.backward()
    gpu_loss = torch.nn.functional.mse_loss(gpu(x.to("vulkan")), labels.to("vulkan"))
    gpu_loss.backward()
    torch.testing.assert_close(gpu_loss.cpu(), cpu_loss.detach(), atol=1e-5, rtol=1e-5)
    for a, b in zip(cpu.parameters(), gpu.parameters()):
        torch.testing.assert_close(b.grad.cpu(), a.grad, atol=6e-5, rtol=6e-5)
