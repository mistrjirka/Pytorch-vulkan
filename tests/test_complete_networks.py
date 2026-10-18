import copy
import sys
from pathlib import Path
import pytest
import torch
import torch_vulkan
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'benchmarks'))
from feature_networks import COMPLETION_CASES,make_case

@pytest.mark.parametrize('name',COMPLETION_CASES)
def test_composed_network_training(name):
    model,x=make_case(name);gpu=copy.deepcopy(model).to('vulkan')
    a=x.requires_grad_();b=x.detach().to('vulkan').requires_grad_()
    reference=torch.optim.AdamW(model.parameters(),lr=.001,foreach=False,amsgrad=True)
    actual=torch.optim.AdamW(gpu.parameters(),lr=.001,foreach=False,amsgrad=True)
    for step in range(2):
        reference.zero_grad(set_to_none=True);actual.zero_grad(set_to_none=True)
        a.grad=None;b.grad=None
        expected=model(a);expected.square().mean().backward();reference.step()
        torch_vulkan.reset_execution_stats()
        result=gpu(b);result.square().mean().backward();actual.step()
        stats=torch_vulkan.execution_stats()
        assert stats['cpu_fallback_calls']==0 and stats['host_read_max_bytes']<=4,stats
        torch.testing.assert_close(result.cpu(),expected,rtol=1e-3,atol=6e-5)
        torch.testing.assert_close(b.grad.cpu(),a.grad,rtol=2e-3,atol=8e-5)
        for p,q in zip(gpu.parameters(),model.parameters()):
            torch.testing.assert_close(p.grad.cpu(),q.grad,rtol=2e-3,atol=8e-5)
            torch.testing.assert_close(p.cpu(),q,rtol=1e-3,atol=6e-5)
            for key in ('exp_avg','exp_avg_sq','max_exp_avg_sq'):
                torch.testing.assert_close(actual.state[p][key].cpu(),reference.state[q][key],rtol=2e-3,atol=8e-6)
