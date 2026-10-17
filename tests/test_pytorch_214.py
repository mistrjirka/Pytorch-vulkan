"""PyTorch 2.14.1 Vulkan bring-up and training regression tests.

Run against a Vulkan ICD; Lavapipe validates functional behavior only, not GPU speed.
"""
import torch
import torch_vulkan


def check_close(actual, expected, *, tol=1e-4):
    assert actual.device.type == "vulkan"
    torch.testing.assert_close(actual.cpu(), expected, rtol=tol, atol=tol)


def test_pytorch_version():
    assert torch.__version__.split("+")[0] == "2.14.1"


def test_cpu_to_device_roundtrip():
    a = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    b = a.to("vulkan")
    assert b.device.type == "vulkan"
    torch.testing.assert_close(b.cpu(), a)


def test_elementwise_gpu_ops():
    a = torch.tensor([1.5, 2.5, 3.5, 4.5])
    b = torch.tensor([2.0, -1.0, 4.0, 0.5])
    x = a.to("vulkan")
    y = b.to("vulkan")
    check_close(torch.add(x, y, alpha=2), torch.add(a, b, alpha=2))
    check_close(x * y, a * b)


def test_matmul_gpu_op():
    a = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
    b = torch.tensor([[1., 0.], [2., -1.], [3., 4.]])
    check_close(a.to("vulkan") @ b.to("vulkan"), a @ b)


def test_mul_backward():
    a = torch.tensor([2., 3., 4.])
    b = torch.tensor([5., 6., 7.])
    go = torch.tensor([1., -2., 3.])
    x = a.to("vulkan").detach().requires_grad_()
    y = b.to("vulkan").detach().requires_grad_()
    out = x * y
    gx, gy = torch.autograd.grad(out, (x, y), grad_outputs=go.to("vulkan"))
    check_close(gx, go * b)
    check_close(gy, go * a)


def test_matmul_backward():
    a = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
    b = torch.tensor([[1., 0.], [2., -1.], [3., 4.]])
    go = torch.tensor([[2., -1.], [3., 4.]])
    a_cpu, b_cpu = a.clone().requires_grad_(), b.clone().requires_grad_()
    ref_a, ref_b = torch.autograd.grad(a_cpu @ b_cpu, (a_cpu, b_cpu), grad_outputs=go)
    x = a.to("vulkan").detach().requires_grad_()
    y = b.to("vulkan").detach().requires_grad_()
    gx, gy = torch.autograd.grad(x @ y, (x, y), grad_outputs=go.to("vulkan"))
    check_close(gx, ref_a)
    check_close(gy, ref_b)


def test_native_relu_backward():
    assert torch._C._dispatch_has_kernel_for_dispatch_key(
        "aten::threshold_backward", "PrivateUse1"
    ), "ReLU backward must be Vulkan-native, not CPU fallback"
    cpu = torch.tensor([-2., -1., 0., 1., 2.], requires_grad=True)
    grad = torch.tensor([1., 2., 3., 4., 5.])
    reference = torch.autograd.grad(torch.relu(cpu), cpu, grad_outputs=grad)[0]
    x = cpu.detach().to("vulkan").requires_grad_()
    (got,) = torch.autograd.grad(
        torch.relu(x), x, grad_outputs=grad.to("vulkan")
    )
    check_close(got, reference)


def test_selected_vulkan_driver_is_reported():
    assert torch_vulkan.is_available()
    assert torch_vulkan.device_name()
    assert torch_vulkan.device_name() != "Vulkan Device 0"
    assert isinstance(torch_vulkan.is_hardware_accelerated(), bool)


def test_device_to_cpu_cast_dtype():
    cpu = torch.tensor([1.25, -2.75], dtype=torch.float32)
    out = cpu.to("vulkan").to(device="cpu", dtype=torch.float64)
    assert out.dtype == torch.float64
    torch.testing.assert_close(out, cpu.double())


def test_reject_incompatible_dtype_in_arithmetic():
    import pytest
    x = torch.tensor([1, 2], dtype=torch.int32).to("vulkan")
    with pytest.raises(RuntimeError, match="float32 inputs required"):
        _ = x + x
