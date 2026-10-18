import pytest
import torch
import torch.nn.functional as F
import torch_vulkan
from test_batch_norm import layout

@pytest.mark.parametrize('shape,size,scales',[
    ((2,3,5,7),(10,14),None),((1,7,3,5),(7,13),None),
    ((2,3,9,11),(4,3),None),((1,2,1,7),(3,1),None),
    ((1,2,7,9),(7,9),None),((2,3,5,7),None,(1.7,2.3)),
    ((2,3,13,11),None,(.6,.7)),((2,3,7,9),None,(1.001,1.5)),
])
@pytest.mark.parametrize('kind',['contiguous','channels_last','slice','transpose','broadcast'])
def test_nearest2d_forward_backward(shape,size,scales,kind):
    torch.manual_seed(45)
    x=torch.randn(shape)
    xc=layout(x,kind,'cpu').detach().requires_grad_();xv=layout(x,kind,'vulkan').detach().requires_grad_()
    yc=F.interpolate(xc,size=size,scale_factor=scales,mode='nearest')
    yv=F.interpolate(xv,size=size,scale_factor=scales,mode='nearest')
    g=torch.randn(yc.shape)
    yc.backward(layout(g,kind,'cpu'));yv.backward(layout(g,kind,'vulkan'))
    torch.testing.assert_close(yv.cpu(),yc,atol=0,rtol=0)
    torch.testing.assert_close(xv.grad.cpu(),xc.grad,atol=2e-6,rtol=2e-6)


def test_nearest_broadcast_grad_and_deferred_chain():
    x=torch.randn(2,3,5,7)
    def run(device):
        v=x.to(device).detach().requires_grad_()
        y=F.interpolate(F.interpolate(v,scale_factor=1.7,mode='nearest'),size=(7,11),mode='nearest')
        y.sum().backward()
        return y,v.grad
    ref=run('cpu');got=run('vulkan')
    for a,b in zip(got,ref):torch.testing.assert_close(a.cpu(),b,rtol=0,atol=0)


def test_nearest_no_fallback():
    x=torch.randn(1,2,5,7).to('vulkan')
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
        y=torch.ops.aten.upsample_nearest2d(x,[7,11],None,None)
        torch.ops.aten.upsample_nearest2d_backward(y,[7,11],list(x.shape),None,None)
        torch_vulkan.synchronize()
    assert not any(e.key=='aten::_to_copy' for e in prof.key_averages())


def test_nearest_empty_batch():
    x=torch.empty(0,3,5,7,device='vulkan',requires_grad=True)
    y=F.interpolate(x,size=(7,11),mode='nearest')
    assert y.shape==(0,3,7,11)
    y.backward(torch.empty_like(y))
    assert x.grad.shape==x.shape
