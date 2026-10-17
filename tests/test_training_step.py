"""Small SGD-style training regression, once backward paths are implemented."""
import torch
import torch_vulkan


def test_linear_weight_update():
    x_cpu = torch.tensor([[1., 2.], [3., -2.], [2., 1.]])
    w_cpu = torch.tensor([[0.5], [-1.0]])
    target_cpu = torch.tensor([[1.], [2.], [3.]])
    x = x_cpu.to('vulkan')
    w = w_cpu.to('vulkan').detach().requires_grad_()
    # Explicit gradient avoids hiding missing sum/reduction kernels in fallback.
    pred = x @ w
    grad_pred = 2 * (pred.cpu() - target_cpu) / target_cpu.numel()
    (dw,) = torch.autograd.grad(pred, w, grad_outputs=grad_pred.to('vulkan'))
    ref_dw = x_cpu.t() @ grad_pred
    torch.testing.assert_close(dw.cpu(), ref_dw, atol=1e-4, rtol=1e-4)


def test_mse_sgd_decreases_loss():
    # Exercises CPU fallback for unsupported loss/reduction/optimizer ops,
    # while GEMM and its autograd gradient dispatch on Vulkan.
    x = torch.tensor([[1., 2.], [2., 1.], [3., 4.]]).to("vulkan")
    y = torch.tensor([[1.], [2.], [3.]]).to("vulkan")
    w = torch.nn.Parameter(torch.zeros(2, 1, device="vulkan"))
    optimizer = torch.optim.SGD([w], lr=0.01)
    losses = []
    for _ in range(5):
        optimizer.zero_grad()
        loss = torch.nn.functional.mse_loss(x @ w, y)
        losses.append(loss.detach().cpu().item())
        loss.backward()
        assert w.grad is not None and w.grad.device.type == "vulkan"
        optimizer.step()
    assert losses[-1] < losses[0] * 0.2


def test_adamw_registration_and_parameter_update_matches_cpu():
    # PyTorch 2.14 optimizers require torch.vulkan to be a registered module.
    assert torch.vulkan is torch_vulkan
    cpu = torch.nn.Parameter(torch.tensor([1., 2., 3.]))
    gpu = torch.nn.Parameter(cpu.detach().to("vulkan"))
    cpu_opt = torch.optim.AdamW([cpu], lr=0.0001)
    gpu_opt = torch.optim.AdamW([gpu], lr=0.0001)
    (cpu * cpu).sum().backward()
    (gpu * gpu).sum().backward()
    cpu_opt.step()
    gpu_opt.step()
    torch.testing.assert_close(gpu.cpu(), cpu, atol=1e-6, rtol=1e-6)
