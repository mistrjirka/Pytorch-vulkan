"""FP32 log-softmax must stay on device across execution modes and layouts."""
import pytest
import torch
import torch.nn.functional as F
import torch_vulkan as vk

LAYOUTS={
 "contiguous":((2,3,7),lambda x:x),
 "transpose":((2,3,7),lambda x:x.transpose(0,2)),
 "offset":((4,5,11),lambda x:x[1:3,1:4,2:9]),
 "slice":((4,6,13),lambda x:x[::2,::2,1::2]),
 "expanded":((1,3,1),lambda x:x.expand(2,3,7)),
 "negative":((2,3,7),torch._neg_view),
 "scalar":((),lambda x:x),
 "empty":((2,0,7),lambda x:x),
 "rank9":((2,1,2,1,2,1,2,1,3),lambda x:x.transpose(0,8)),
}
RANKS={k:len(shape) for k,(shape,_) in LAYOUTS.items()}
CASES=[(name,axis) for name,rank in RANKS.items() for axis in range(-max(1,rank),max(1,rank))]

def pair(name,seed):
    shape,view=LAYOUTS[name]
    base=torch.randn(shape,generator=torch.Generator().manual_seed(seed))
    gpu=base.to("vulkan")
    return view(base),view(gpu)

@pytest.mark.parametrize("layout,axis",CASES)
def test_axes_layouts_forward_backward_no_host_reads(layout,axis,monkeypatch):
    monkeypatch.delenv("TORCH_VULKAN_NATIVE_LOGSOFTMAX",raising=False)
    monkeypatch.setenv("TORCH_VULKAN_FORBID_CPU_FALLBACK","1")
    cpu,gpu=pair(layout,311);grad,ggrad=pair(layout,312)
    cpu=cpu.detach().requires_grad_();gpu=gpu.detach().requires_grad_()
    expected=F.log_softmax(cpu,axis)
    reference_grad=torch.autograd.grad(expected,cpu,grad)[0]
    vk.reset_execution_stats()
    actual=F.log_softmax(gpu,axis)
    actual_grad=torch.autograd.grad(actual,gpu,ggrad)[0]
    stats=vk.execution_stats()
    assert stats["cpu_fallback_calls"]==stats["host_read_calls"]==0,stats
    torch.testing.assert_close(actual.cpu(),expected,rtol=3e-5,atol=3e-5)
    torch.testing.assert_close(actual_grad.cpu(),reference_grad,rtol=5e-5,atol=5e-5)
    assert actual.is_contiguous()

@pytest.mark.parametrize("axis",[0,1,-1])
def test_channels_last_input_and_gradient(axis,monkeypatch):
    monkeypatch.setenv("TORCH_VULKAN_FORBID_CPU_FALLBACK","1")
    cpu=torch.randn(2,3,5,7).to(memory_format=torch.channels_last).requires_grad_()
    gpu=cpu.detach().to("vulkan",memory_format=torch.channels_last).requires_grad_()
    grad=torch.randn_like(cpu)
    expected=F.log_softmax(cpu,axis);dx=torch.autograd.grad(expected,cpu,grad)[0]
    vk.reset_execution_stats()
    actual=F.log_softmax(gpu,axis);dv=torch.autograd.grad(actual,gpu,grad.to("vulkan",memory_format=torch.channels_last))[0]
    assert vk.execution_stats()["host_read_calls"]==vk.execution_stats()["cpu_fallback_calls"]==0
    torch.testing.assert_close(actual.cpu(),expected,rtol=3e-5,atol=3e-5)
    torch.testing.assert_close(dv.cpu(),dx,rtol=5e-5,atol=5e-5)

@pytest.mark.parametrize("axis",[0,1])
def test_nonfinite_classification(axis,monkeypatch):
    monkeypatch.setenv("TORCH_VULKAN_FORBID_CPU_FALLBACK","1")
    x=torch.tensor([[1.,2.,3.],[-float("inf"),-float("inf"),-float("inf")],
        [float("inf"),2.,3.],[float("nan"),2.,3.],[-1e38,1e38,0.]])
    expected=F.log_softmax(x,axis)
    actual=F.log_softmax(x.to("vulkan"),axis).cpu()
    torch.testing.assert_close(actual,expected,rtol=3e-5,atol=3e-5,equal_nan=True)
    torch.testing.assert_close(torch.isnan(actual),torch.isnan(expected))

def test_more_than_65535_rows(monkeypatch):
    monkeypatch.setenv("TORCH_VULKAN_FORBID_CPU_FALLBACK","1")
    x=torch.randn(65536,1);gpu=x.to("vulkan").requires_grad_()
    vk.reset_execution_stats()
    out=F.log_softmax(gpu,-1);out.backward(torch.ones_like(out))
    assert vk.execution_stats()["host_read_calls"]==vk.execution_stats()["cpu_fallback_calls"]==0
    torch.testing.assert_close(out.cpu(),torch.zeros_like(x))
    torch.testing.assert_close(gpu.grad.cpu(),torch.zeros_like(x))

def test_explicit_cpu_diagnostic_is_audited_and_forbidden(monkeypatch):
    x=torch.randn(3,7);gpu=x.to("vulkan")
    monkeypatch.setenv("TORCH_VULKAN_NATIVE_LOGSOFTMAX","0")
    monkeypatch.delenv("TORCH_VULKAN_FORBID_CPU_FALLBACK",raising=False)
    vk.reset_execution_stats();out=F.log_softmax(gpu,-1)
    assert vk.execution_stats()["cpu_fallback_calls"]>0
    torch.testing.assert_close(out.cpu(),F.log_softmax(x,-1))
    monkeypatch.setenv("TORCH_VULKAN_FORBID_CPU_FALLBACK","1")
    vk.reset_execution_stats()
    with pytest.raises(RuntimeError,match="CPU fallback forbidden"):
        F.log_softmax(gpu,-1)
    assert vk.execution_stats()["cpu_fallback_calls"]>0
    assert vk.execution_stats()["host_read_calls"]==0

def test_unsupported_precision_obeys_strict_audit(monkeypatch):
    gpu=torch.randn(2,3,dtype=torch.float64).to("vulkan")
    monkeypatch.setenv("TORCH_VULKAN_FORBID_CPU_FALLBACK","1")
    vk.reset_execution_stats()
    with pytest.raises(RuntimeError,match="CPU fallback forbidden"):
        F.log_softmax(gpu,-1)
    assert vk.execution_stats()["cpu_fallback_calls"]>0
    assert vk.execution_stats()["host_read_calls"]==0
