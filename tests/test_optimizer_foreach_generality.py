"""Regression checks for *standard* PyTorch Adam/AdamW on Vulkan.

Vertical fusion and foreach dispatch optimizations must preserve the exact
optimizer selected by the user: Adam is not interchangeable with AdamW.
Test heterogeneous parameter groups, missing gradients, maximize, AMSGrad,
and the standard foreach path against the regular CPU implementation.
"""
import pytest
import torch
import torch_vulkan


@pytest.mark.parametrize("cls", [torch.optim.Adam, torch.optim.AdamW],
                         ids=["adam", "adamw"])
@pytest.mark.parametrize("foreach", [False, True])
@pytest.mark.parametrize("amsgrad", [False, True])
@pytest.mark.parametrize("maximize", [False, True])
def test_heterogeneous_groups_and_state_match_cpu(cls, foreach, amsgrad, maximize):
    gen = torch.Generator().manual_seed(1227)
    shapes = [(7,), (3, 9), (4, 2, 13)]
    cpu = [torch.nn.Parameter(torch.randn(s, generator=gen)) for s in shapes]
    gpu = [torch.nn.Parameter(p.detach().to("vulkan")) for p in cpu]
    def groups(tensors):
        return [
            {"params": tensors[:2], "lr": 0.003, "weight_decay": 0.041,
             "betas": (0.83, 0.92), "eps": 1e-6},
            {"params": tensors[2:], "lr": 0.0007, "weight_decay": 0.025,
             "betas": (0.76, 0.96), "eps": 1e-7},
        ]
    cpu_opt = cls(groups(cpu), foreach=foreach, amsgrad=amsgrad, maximize=maximize)
    gpu_opt = cls(groups(gpu), foreach=foreach, amsgrad=amsgrad, maximize=maximize)
    initial_fallback = torch_vulkan.execution_stats()["cpu_fallback_calls"]

    # Keep model-side gradient creation identical across the two backends.
    for step in range(7):
        for i, (x, y) in enumerate(zip(cpu, gpu)):
            if (step + i) % 4 == 0:
                x.grad = y.grad = None
                continue
            g = torch.randn(x.shape, generator=gen) * (0.03 + step * 0.02)
            x.grad = g
            y.grad = g.to("vulkan")
        cpu_opt.step()
        gpu_opt.step()
    torch_vulkan.synchronize()

    for x, y in zip(cpu, gpu):
        torch.testing.assert_close(y.detach().cpu(), x.detach(), rtol=6e-4, atol=1e-5)
        cpu_state, gpu_state = cpu_opt.state[x], gpu_opt.state[y]
        assert cpu_state.keys() == gpu_state.keys()
        assert int(cpu_state["step"].item()) == int(gpu_state["step"].cpu().item())
        for key in ["exp_avg", "exp_avg_sq"] + (["max_exp_avg_sq"] if amsgrad else []):
            torch.testing.assert_close(gpu_state[key].cpu(), cpu_state[key],
                                       rtol=7e-4, atol=1e-5)
    assert torch_vulkan.execution_stats()["cpu_fallback_calls"] == initial_fallback
