"""Scalar updates preserve integer bits, view aliases, and queued GPU work."""
import pytest
import torch
import torch_vulkan


def view(x,layout):
    if layout=='transpose':return x.t()
    if layout=='offset':return x[1:,1:]
    if layout=='stepped':return x[:,::2]
    return x


@pytest.mark.parametrize('dtype',[torch.float32,torch.int64])
@pytest.mark.parametrize('layout',['contiguous','transpose','offset','stepped'])
@pytest.mark.parametrize('inplace',[False,True])
def test_scalar_add_layouts(dtype,layout,inplace):
    cpu=torch.arange(63,dtype=dtype).reshape(7,9)-30
    gpu=cpu.to('vulkan');a=view(cpu,layout);b=view(gpu,layout)
    pointer=b.data_ptr()
    if inplace:
        a.add_(-7,alpha=-3);out=b.add_(-7,alpha=-3)
        assert out.data_ptr()==pointer
        torch.testing.assert_close(gpu.cpu(),cpu)
    else:
        out=torch.ops.aten.add.Scalar(b,-7,alpha=-3)
        torch.testing.assert_close(out.cpu(),torch.ops.aten.add.Scalar(a,-7,alpha=-3))
        torch.testing.assert_close(gpu.cpu(),cpu)


@pytest.mark.parametrize('value,alpha',[(1,1),(-1,1),(2**40+1,-7),(-2**63,-1)])
def test_int64_add_carries_and_wraparound(value,alpha):
    a=torch.tensor([-2**63,-2**63+1,-2**40,-1,0,2**32-1,2**40,2**63-1])
    b=a.to('vulkan')
    for _ in range(3):
        a.add_(value,alpha=alpha);b.add_(value,alpha=alpha)
    torch.testing.assert_close(b.cpu(),a,rtol=0,atol=0)


@pytest.mark.parametrize('shape',[(),(0,3),(2,1,1,1,1,1,2,3),(2,1,1,1,1,1,1,2,3)])
@pytest.mark.parametrize('dtype',[torch.float32,torch.int64])
def test_scalar_add_ranks_and_empty(shape,dtype):
    cpu=torch.zeros(shape,dtype=dtype);gpu=cpu.to('vulkan')
    cpu.add_(3);gpu.add_(3)
    torch.testing.assert_close(gpu.cpu(),cpu)


def test_scalar_add_cpu_tensor_and_fallbacks():
    for dtype,scalar in [(torch.float32,torch.tensor(.25,dtype=torch.float64)),
                         (torch.int64,torch.tensor(2)),(torch.float64,torch.tensor(.25))]:
        a=torch.ones(3,5,dtype=dtype);b=a.to('vulkan')
        a.add_(scalar,alpha=3);b.add_(scalar,alpha=3)
        torch.testing.assert_close(b.cpu(),a)
    a=torch.randn(3,5);b=a.to('vulkan');other=torch.randn(1,5)
    a.add_(other);b.add_(other.to('vulkan'))
    torch.testing.assert_close(b.cpu(),a)


@pytest.mark.parametrize('dtype,value,alpha',[(torch.int64,.5,1),(torch.int64,1,0.5),
                                            (torch.float32,1,True),(torch.float32,1,1j)])
def test_scalar_add_invalid_cast_or_alpha(dtype,value,alpha):
    for device in ['cpu','vulkan']:
        x=torch.ones(4,dtype=dtype).to(device)
        with pytest.raises(RuntimeError):x.add_(value,alpha=alpha)


def test_scalar_add_expanded_overlap_rejected():
    for device in ['cpu','vulkan']:
        x=torch.ones(1).to(device).expand(4)
        with pytest.raises(RuntimeError):x.add_(2)


def test_scalar_add_autograd():
    a=torch.randn(3,5,requires_grad=True);b=a.detach().to('vulkan').requires_grad_()
    ac=a*2;bc=b*2
    ac.add_(.25);bc.add_(.25)
    ac.sum().backward();bc.sum().backward()
    torch.testing.assert_close(b.grad.cpu(),a.grad)


def test_scalar_add_counter_does_not_flush_unrelated_work():
    import os
    if not all(torch_vulkan.performance_options()[k] for k in ("gpu_resident", "deferred_submissions")):
        return
    x=torch.randn(32,32);v=x.to('vulkan');counter=torch.tensor(0,dtype=torch.int64).to('vulkan')
    torch_vulkan.synchronize();before=torch_vulkan.queue_stats()['submissions']
    for _ in range(12):
        v=torch.relu(v);counter.add_(1)
    assert torch_vulkan.queue_stats()['submissions']==before
    torch.testing.assert_close(v.cpu(),torch.relu(x));assert counter.cpu().item()==12
