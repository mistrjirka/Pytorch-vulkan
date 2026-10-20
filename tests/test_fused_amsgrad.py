"""Opt-in fused Vulkan AMSGrad must reproduce standard AdamW state and weights."""
import copy

import pytest
import torch
import torch_vulkan
from torch_vulkan.optim import AdamW as VulkanAdamW


@pytest.mark.parametrize("batched", [True, False])
@pytest.mark.parametrize("shapes", [
    [(37,)],
    [(3, 7), (8,), (2, 3, 5)],
    [(8193,), (0,), (1, 16)],
    [(9, 12, 7), (7, 4)],
])
def test_fused_amsgrad_matches_standard_cpu_state(batched, shapes):
    g=torch.Generator().manual_seed(88219)
    initial=[torch.randn(shape,generator=g) for shape in shapes]
    cpu=[torch.nn.Parameter(p.clone()) for p in initial]
    gpu=[torch.nn.Parameter(p.to("vulkan")) for p in initial]
    opts=dict(lr=.005,betas=(.8,.91),eps=1e-6,weight_decay=.031,amsgrad=True)
    std=torch.optim.AdamW(cpu,foreach=False,**opts)
    fused=VulkanAdamW(gpu,batched=batched,**opts)
    for step in range(1,12):
        for i,(c,v) in enumerate(zip(cpu,gpu)):
            # Exercise missing gradients after states have been populated.
            if (step+i)%5==0:
                c.grad=None
                v.grad=None
            else:
                grad=torch.randn(c.shape,generator=g)*(0.1+.01*step)
                c.grad=grad
                v.grad=grad.to("vulkan")
        std.step()
        fused.step()
        for c,v in zip(cpu,gpu):
            torch.testing.assert_close(v.detach().cpu(),c.detach(),rtol=6e-5,atol=3e-5)
            if c in std.state:
                for key in ["exp_avg","exp_avg_sq","max_exp_avg_sq"]:
                    torch.testing.assert_close(fused.state[v][key].cpu(),std.state[c][key],
                                               rtol=6e-5,atol=3e-5)
                assert fused.state[v]["step"]==int(std.state[c]["step"].item())
        assert torch_vulkan.execution_stats()["cpu_fallback_calls"]==0


def test_fused_amsgrad_restored_state_and_parameter_groups():
    torch.manual_seed(405)
    layers=[torch.nn.Parameter(torch.randn(7,11)),torch.nn.Parameter(torch.randn(4,3))]
    cuda_params=[torch.nn.Parameter(p.detach().to("vulkan")) for p in layers]
    # Distinct betas/lr and no moment mixing across parameter groups.
    groups_cpu=[{"params":[layers[0]],"lr":.001,"betas":(.75,.88)},
                {"params":[layers[1]],"lr":.005,"betas":(.91,.975)}]
    groups_vk=[{"params":[cuda_params[0]],"lr":.001,"betas":(.75,.88)},
               {"params":[cuda_params[1]],"lr":.005,"betas":(.91,.975)}]
    a=torch.optim.AdamW(groups_cpu,amsgrad=True,foreach=False)
    b=VulkanAdamW(groups_vk,amsgrad=True,batched=True)
    for step in range(9):
        for x,y in zip(layers,cuda_params):
            grad=torch.randn_like(x)*(.001+step*.003)
            x.grad=grad
            y.grad=grad.to("vulkan")
        a.step();b.step()
    checkpoint=copy.deepcopy(b.state_dict())
    restored_params=[torch.nn.Parameter(p.detach().clone()) for p in cuda_params]
    restored=VulkanAdamW([
        {"params":[restored_params[0]],"lr":.001,"betas":(.75,.88)},
        {"params":[restored_params[1]],"lr":.005,"betas":(.91,.975)},
    ],amsgrad=True,batched=True)
    restored.load_state_dict(checkpoint)
    # Preserve parameters, moments and step counts across checkpoint reload.
    for x,y in zip(cuda_params,restored_params):
        torch.testing.assert_close(x.detach().cpu(),y.detach().cpu())
    for x,y in zip(cuda_params,restored_params):
        for state in ("exp_avg","exp_avg_sq","max_exp_avg_sq"):
            torch.testing.assert_close(b.state[x][state].cpu(),restored.state[y][state].cpu())
        assert b.state[x]["step"]==restored.state[y]["step"]
    for step in range(3):
        for x,y in zip(cuda_params,restored_params):
            grad=torch.randn(x.shape)
            x.grad=grad.to("vulkan")
            y.grad=grad.to("vulkan")
        b.step();restored.step()
    for x,y in zip(cuda_params,restored_params):
        torch.testing.assert_close(x.detach().cpu(),y.detach().cpu(),atol=3e-5,rtol=3e-5)


@pytest.mark.parametrize("amsgrad", [False, True])
def test_fused_adamw_multiple_queued_optimizer_steps(amsgrad):
    cpu=torch.nn.Parameter(torch.linspace(-.5,.6,97))
    gpu=torch.nn.Parameter(cpu.detach().to("vulkan"))
    ref=torch.optim.AdamW([cpu],amsgrad=amsgrad,foreach=False)
    opt=VulkanAdamW([gpu],amsgrad=amsgrad)
    for _ in range(8):
        ref.zero_grad(set_to_none=True);opt.zero_grad(set_to_none=True)
        cpu.square().sum().backward()
        gpu.square().sum().backward()
        ref.step();opt.step()
    torch.testing.assert_close(gpu.detach().cpu(),cpu.detach(),atol=5e-5,rtol=5e-5)
    for name in ("exp_avg","exp_avg_sq")+ (("max_exp_avg_sq",) if amsgrad else ()):
        torch.testing.assert_close(opt.state[gpu][name].cpu(),ref.state[cpu][name],atol=5e-5,rtol=5e-5)
    assert torch_vulkan.execution_stats()["cpu_fallback_calls"]==0


def test_fused_amsgrad_imports_standard_pytorch_optimizer_state():
    """A standard AdamW checkpoint with scalar-Tensor 'step' must resume."""
    original=torch.nn.Parameter(torch.linspace(-1,1,17))
    ref=torch.optim.AdamW([original],lr=.002,amsgrad=True,foreach=False)
    for _ in range(4):
        original.grad=torch.full_like(original,.04)
        ref.step()
    state=copy.deepcopy(ref.state_dict())
    vulkan=torch.nn.Parameter(original.detach().to('vulkan'))
    fused=VulkanAdamW([vulkan],lr=.002,amsgrad=True)
    fused.load_state_dict(state)
    assert isinstance(fused.state[vulkan]['step'],torch.Tensor)
    original.grad=torch.full_like(original,.05)
    vulkan.grad=original.grad.to('vulkan')
    ref.step();fused.step()
    torch.testing.assert_close(vulkan.detach().cpu(),original.detach(),rtol=5e-5,atol=3e-5)
    for name in ('exp_avg','exp_avg_sq','max_exp_avg_sq'):
        torch.testing.assert_close(fused.state[vulkan][name].cpu(),ref.state[original][name],rtol=5e-5,atol=3e-5)



def test_fused_amsgrad_propagates_nan_in_maximum_moment():
    # Standard torch.maximum propagates NaNs even if the old maximum is finite.
    cpu=torch.nn.Parameter(torch.tensor([.25,.5,.75]))
    gpu=torch.nn.Parameter(cpu.detach().to('vulkan'))
    a=torch.optim.AdamW([cpu],amsgrad=True,foreach=False)
    b=VulkanAdamW([gpu],amsgrad=True,batched=True)
    cpu.grad=torch.tensor([.1,.2,.3])
    gpu.grad=cpu.grad.to('vulkan')
    a.step();b.step()
    cpu.grad=torch.tensor([float('nan'),.1,.15])
    gpu.grad=cpu.grad.to('vulkan')
    a.step();b.step()
    for name in ("exp_avg","exp_avg_sq","max_exp_avg_sq"):
        torch.testing.assert_close(b.state[gpu][name].cpu(),
                                   a.state[cpu][name],equal_nan=True)
    torch.testing.assert_close(gpu.detach().cpu(),cpu.detach(),equal_nan=True)
