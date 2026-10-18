import sys
from pathlib import Path
import pytest
import torch
import torch_vulkan
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'benchmarks'))
from feature_networks import CASES,make_case

@pytest.mark.parametrize('name',CASES)
def test_network_training_native_and_correct(name,monkeypatch,capfd):
    import copy
    model,x=make_case(name)
    gpu=copy.deepcopy(model).to('vulkan')
    a=x.requires_grad_();b=x.detach().to('vulkan').requires_grad_()
    monkeypatch.setenv('TORCH_VULKAN_TRACE_FALLBACK','1')
    expected=model(a);actual=gpu(b)
    grad=torch.randn_like(expected)
    expected.backward(grad);actual.backward(grad.to('vulkan'))
    torch.testing.assert_close(actual.cpu(),expected,rtol=6e-4,atol=3e-5)
    torch.testing.assert_close(b.grad.cpu(),a.grad,rtol=8e-4,atol=5e-5)
    for p,q in zip(gpu.parameters(),model.parameters()):
        assert p.grad is not None and q.grad is not None
        torch.testing.assert_close(p.grad.cpu(),q.grad,rtol=8e-4,atol=8e-5)
    assert 'PV_FALLBACK' not in capfd.readouterr().err
