"""Validate native Vulkan AdamW weights and moment tensors against CPU PyTorch."""
import copy
import pytest
import torch
import torch_vulkan
from torch_vulkan.optim import AdamW as VulkanAdamW


@pytest.mark.parametrize('shape', [(3,), (4, 5), (137, 351), (256, 1024)])
def test_fused_adamw_matches_pytorch_cpu(shape):
    torch.manual_seed(2026)
    param_cpu = torch.nn.Parameter(torch.randn(*shape))
    param_vk = torch.nn.Parameter(param_cpu.detach().to('vulkan'))
    optim_cpu = torch.optim.AdamW([param_cpu], lr=0.005, betas=(0.8, 0.91),
                                   eps=1e-6, weight_decay=0.031, foreach=False)
    optim_vk = VulkanAdamW([param_vk], lr=0.005, betas=(0.8, 0.91),
                            eps=1e-6, weight_decay=0.031)
    for i in range(8):
        grad = torch.randn(*shape) * (0.1 + i)
        param_cpu.grad = grad.clone()
        param_vk.grad = grad.to('vulkan')
        optim_cpu.step()
        optim_vk.step()
        torch.testing.assert_close(param_vk.detach().cpu(), param_cpu.detach(),
                                   atol=2e-5, rtol=2e-5)
        state_cpu = optim_cpu.state[param_cpu]
        state_vk = optim_vk.state[param_vk]
        for key in ('exp_avg','exp_avg_sq'):
            torch.testing.assert_close(state_vk[key].cpu(), state_cpu[key],
                                       atol=2e-5, rtol=2e-5)


def test_fused_adamw_trains_real_mlp():
    from pathlib import Path
    import sys
    zoo = Path('/workspace/pytorch-XDNA2-vulkan-zoo-reference/benchmarks')
    if not (zoo / 'model_zoo.py').exists():
        pytest.skip('Model zoo not present')
    sys.path.insert(0,str(zoo))
    import model_zoo
    torch.set_num_threads(6)
    torch.manual_seed(8)
    ref,_,_ = model_zoo.MODELS['mlp']()
    vk = copy.deepcopy(ref).to('vulkan')
    cpu_optim = torch.optim.AdamW(ref.parameters(),lr=0.0001,foreach=False)
    vk_optim = VulkanAdamW(vk.parameters(),lr=0.0001)
    x=torch.randn(16,784)
    target=torch.randint(10,(16,))
    for i in range(3):
        cpu_optim.zero_grad(set_to_none=True)
        a=torch.nn.functional.cross_entropy(ref(x),target)
        a.backward();cpu_optim.step()
        vk_optim.zero_grad(set_to_none=True)
        b=torch.nn.functional.cross_entropy(vk(x.to('vulkan')),target.to('vulkan'))
        b.backward();vk_optim.step()
        assert abs(float(a.detach())-float(b.detach().cpu())) < 1.e-3
    for pa,pb in zip(ref.parameters(),vk.parameters()):
        torch.testing.assert_close(pb.detach().cpu(),pa.detach(),atol=1e-3,rtol=1e-3)


def test_adamw_queued_gradients_updates_and_parameter_reads():
    """Several optimizer steps may remain queued without stale gradients/state."""
    torch.manual_seed(719)
    a=torch.nn.Parameter(torch.randn(7,13));b=torch.nn.Parameter(a.detach().to('vulkan'))
    oa=torch.optim.AdamW([a],lr=.001,foreach=False);ob=VulkanAdamW([b],lr=.001)
    for _ in range(6):
        oa.zero_grad(set_to_none=True);ob.zero_grad(set_to_none=True)
        # Native multiply backward generates new GPU gradients, then the next
        # iteration immediately reads the asynchronously updated parameter.
        (a*a).sum().backward();(b*b).sum().backward()
        oa.step();ob.step()
    torch.testing.assert_close(b.detach().cpu(),a.detach(),atol=2e-5,rtol=2e-5)
    for key in ['exp_avg','exp_avg_sq']:
        torch.testing.assert_close(ob.state[b][key].cpu(),oa.state[a][key],atol=2e-5,rtol=2e-5)
