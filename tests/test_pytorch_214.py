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
    x = torch.tensor([1, 2], dtype=torch.float64).to("vulkan")
    with pytest.raises(RuntimeError, match="float32 inputs required"):
        _ = x + x


def test_layer_norm_forward_saved_statistics_and_backward():
    for shape, normalized in [((4, 7), (7,)), ((2, 3, 5), (5,)),
                               ((2, 3, 5), (3, 5))]:
        torch.manual_seed(19)
        cpu_x = torch.randn(*shape, requires_grad=True)
        cpu_w = torch.randn(*normalized, requires_grad=True)
        cpu_b = torch.randn(*normalized, requires_grad=True)
        grads = torch.randn(*shape)
        eps = 0.003
        out, mean, rstd = torch.ops.aten.native_layer_norm(
            cpu_x, normalized, cpu_w, cpu_b, eps
        )
        x = cpu_x.detach().to("vulkan").requires_grad_()
        w = cpu_w.detach().to("vulkan").requires_grad_()
        b = cpu_b.detach().to("vulkan").requires_grad_()
        vkout, vkmean, vkrstd = torch.ops.aten.native_layer_norm(
            x, normalized, w, b, eps
        )
        check_close(vkout, out, tol=1e-4)
        assert vkmean.shape == mean.shape
        assert vkrstd.shape == rstd.shape
        check_close(vkmean, mean, tol=1e-5)
        check_close(vkrstd, rstd, tol=1e-5)
        ref = torch.autograd.grad(out, (cpu_x, cpu_w, cpu_b), grad_outputs=grads)
        got = torch.autograd.grad(vkout, (x, w, b), grad_outputs=grads.to('vulkan'))
        for actual, expected in zip(got, ref):
            check_close(actual, expected, tol=2e-4)


def test_real_gpu_dispatch_or_explicit_skip():
    # Hardware CI/benchmark test: Lavapipe must not masquerade as a GPU.
    import pytest
    if not torch_vulkan.is_hardware_accelerated():
        pytest.skip("Only software Vulkan found")
    assert "llvmpipe" not in torch_vulkan.device_name().lower()
    a = torch.arange(1024, dtype=torch.float32).view(32, 32)
    x = a.to("vulkan")
    y = (x @ x).cpu()
    torch.testing.assert_close(y, a @ a, rtol=1e-4, atol=1e-4)


def test_view_storage_alias_and_backward():
    cpu = torch.randn(2, 3, 4, requires_grad=True)
    gpu = cpu.detach().to("vulkan").requires_grad_()
    view = gpu.view(6, 4)
    assert view.untyped_storage().data_ptr() == gpu.untyped_storage().data_ptr()
    check_close(view, cpu.detach().view(6, 4))
    grad = torch.randn(6, 4)
    expected = torch.autograd.grad(cpu.view(6, 4), cpu, grad_outputs=grad)[0]
    actual = torch.autograd.grad(view, gpu, grad_outputs=grad.to("vulkan"))[0]
    check_close(actual, expected)


def test_as_strided_noncontiguous():
    cpu = torch.arange(20, dtype=torch.float32).view(4, 5)
    gpu = cpu.to("vulkan")
    view = torch.as_strided(gpu, size=(3, 3), stride=(5, 1), storage_offset=1)
    assert view.stride() == (5, 1)
    torch.testing.assert_close(view.cpu(),
        torch.as_strided(cpu, size=(3, 3), stride=(5, 1), storage_offset=1))


def test_as_strided_gradient_is_strided_and_correct():
    cpu = torch.randn(4, 5, requires_grad=True)
    gpu = cpu.detach().to("vulkan").requires_grad_()
    def view(x):
        return torch.as_strided(x, (3, 3), (5, 1), storage_offset=1)
    grad = torch.randn(3, 3)
    expected = torch.autograd.grad(view(cpu), cpu, grad_outputs=grad)[0]
    actual = torch.autograd.grad(view(gpu), gpu, grad_outputs=grad.to('vulkan'))[0]
    check_close(actual, expected)


def test_broadcast_requires_compatible_shape():
    import pytest
    a = torch.arange(6).float().reshape(2, 3).to('vulkan')
    b = torch.arange(6).float().reshape(3, 2).to('vulkan')
    with pytest.raises(RuntimeError):
        _ = a + b


def test_transpose_is_view_and_matmul_uses_its_strides():
    torch.manual_seed(23)
    for m, k, n in [(4, 3, 7), (13, 33, 29), (32, 128, 64)]:
        cpu_a = torch.randn(m, k, requires_grad=True)
        cpu_b = torch.randn(n, k, requires_grad=True)
        a = cpu_a.detach().to('vulkan').requires_grad_()
        b = cpu_b.detach().to('vulkan').requires_grad_()
        bt = b.t()
        assert bt.stride() == (1, k)
        assert bt.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()
        check_close(bt, cpu_b.detach().t())
        ref = cpu_a @ cpu_b.t()
        got = a @ bt
        check_close(got, ref, tol=6e-4)
        grad = torch.randn_like(ref)
        cpu_grad = torch.autograd.grad(ref, (cpu_a, cpu_b), grad_outputs=grad)
        vk_grad = torch.autograd.grad(got, (a, b), grad_outputs=grad.to('vulkan'))
        for actual, expected in zip(vk_grad, cpu_grad):
            check_close(actual, expected, tol=8e-4)


def test_gelu_forward_backward_matches_exact_cpu_reference():
    torch.manual_seed(251)
    x_cpu = torch.cat((torch.linspace(-8, 8, steps=327), torch.randn(1000))).requires_grad_()
    x_gpu = x_cpu.detach().to('vulkan').requires_grad_()
    grad = torch.randn_like(x_cpu)
    ref = torch.nn.functional.gelu(x_cpu)
    got = torch.nn.functional.gelu(x_gpu)
    check_close(got, ref, tol=3e-6)
    ref_grad = torch.autograd.grad(ref, x_cpu, grad_outputs=grad)[0]
    got_grad = torch.autograd.grad(got, x_gpu, grad_outputs=grad.to('vulkan'))[0]
    check_close(got_grad, ref_grad, tol=4e-6)
    assert torch._C._dispatch_has_kernel_for_dispatch_key(
        'aten::gelu_backward', 'PrivateUse1'
    )


def test_broadcast_add_mul_matches_cpu_with_autograd():
    torch.manual_seed(700)
    x_cpu = torch.randn(2, 3, 5, requires_grad=True)
    y_cpu = torch.randn(1, 3, 1, requires_grad=True)
    x = x_cpu.detach().to('vulkan').requires_grad_()
    y = y_cpu.detach().to('vulkan').requires_grad_()
    for operator in ('add', 'mul'):
        if operator == 'add':
            ref = x_cpu + 2.0 * y_cpu
            actual = torch.add(x, y, alpha=2)
        else:
            ref = x_cpu * y_cpu
            actual = x * y
        check_close(actual, ref.detach(), tol=2e-5)
        grad = torch.randn_like(ref)
        wanted = torch.autograd.grad(ref, (x_cpu, y_cpu), grad, retain_graph=True)
        got = torch.autograd.grad(actual, (x, y), grad.to('vulkan'), retain_graph=True)
        for calc, expected in zip(got, wanted):
            check_close(calc, expected, tol=3e-5)


def test_reshape_alias_backprop_matches_cpu():
    torch.manual_seed(717)
    cpu = torch.randn(2, 3, 4, requires_grad=True)
    vk = cpu.detach().to('vulkan').requires_grad_()
    # Permuted tensor has nontrivial strides. Some reshapes can be views.
    ref = cpu.transpose(1, 2).reshape(2, 4, 3)
    actual = vk.transpose(1, 2).reshape(2, 4, 3)
    check_close(actual, ref, tol=1e-6)
    grad = torch.randn_like(ref)
    ref_grad = torch.autograd.grad(ref, cpu, grad)[0]
    got_grad = torch.autograd.grad(actual, vk, grad.to('vulkan'))[0]
    check_close(got_grad, ref_grad, tol=1e-6)


def test_native_layer_norm_backward_no_affine_and_partial_masks():
    import torch.nn.functional as F
    assert torch._C._dispatch_has_kernel_for_dispatch_key(
        'aten::native_layer_norm_backward', 'PrivateUse1')
    torch.manual_seed(38)
    for shape, normalized in [((3, 7), (7,)), ((2, 3, 5), (3, 5)), ((2, 3, 1), (1,))]:
        x_cpu = torch.randn(*shape, requires_grad=True)
        x_vk = x_cpu.detach().to('vulkan').requires_grad_()
        grad = torch.randn(*shape)
        out_cpu = F.layer_norm(x_cpu, normalized, eps=2e-3)
        out_vk = F.layer_norm(x_vk, normalized, eps=2e-3)
        check_close(out_vk, out_cpu.detach(), tol=2e-4)
        wanted = torch.autograd.grad(out_cpu, x_cpu, grad)[0]
        got = torch.autograd.grad(out_vk, x_vk, grad.to('vulkan'))[0]
        check_close(got, wanted, tol=5e-4)


def test_layer_norm_queued_strided_affine_lifetimes():
    """Packing affine views must survive later allocations and deferred work."""
    torch.manual_seed(83)
    x = torch.randn(3, 4, 7, requires_grad=True)
    w = torch.randn(7, 4).t().detach().requires_grad_()
    b = torch.randn(4, 14)[:, 1::2].detach().requires_grad_()
    xv = x.detach().to('vulkan').requires_grad_()
    wv = w.detach().t().contiguous().to('vulkan').t().requires_grad_()
    rawb = torch.zeros(4, 14)
    rawb[:, 1::2] = b.detach()
    bv = rawb.to('vulkan')[:, 1::2].requires_grad_()
    ref, got = x, xv
    for _ in range(5):
        ref = torch.nn.functional.layer_norm(ref, (4, 7), w, b, 0.003)
        got = torch.nn.functional.layer_norm(got, (4, 7), wv, bv, 0.003)
    grad = torch.randn_like(ref)
    ref_grads = torch.autograd.grad(ref, (x, w, b), grad)
    got_grads = torch.autograd.grad(got, (xv, wv, bv), grad.to('vulkan'))
    torch.testing.assert_close(got.cpu(), ref, rtol=3e-4, atol=2e-5)
    for a, e in zip(got_grads, ref_grads):
        torch.testing.assert_close(a.cpu(), e, rtol=3e-4, atol=2e-5)


def test_layer_norm_rank_nine_view_keeps_correctness_fallback():
    torch.manual_seed(94)
    data = torch.randn(2, 2, 2, 2, 2, 2, 2, 2, 3)
    x = data.transpose(0, 1).detach().requires_grad_()
    xv = data.to('vulkan').transpose(0, 1).detach().requires_grad_()
    grad = torch.randn_like(data).transpose(0, 1)
    y = torch.nn.functional.layer_norm(x, (3,))
    yv = torch.nn.functional.layer_norm(xv, (3,))
    g = torch.autograd.grad(y, x, grad)[0]
    gv = torch.autograd.grad(yv, xv, grad.contiguous().to('vulkan'))[0]
    torch.testing.assert_close(yv.cpu(), y, rtol=3e-4, atol=2e-5)
    torch.testing.assert_close(gv.cpu(), g, rtol=3e-4, atol=2e-5)
