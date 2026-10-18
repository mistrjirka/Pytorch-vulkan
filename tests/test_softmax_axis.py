"""Native Vulkan softmax forward/backward on non-last axes and strided views."""
import pytest
import torch
import torch.nn.functional as F
import torch_vulkan


@pytest.mark.parametrize("shape,dim",[
    ((2,3,5),0),((2,3,5),1),((2,3,5),2),((2,3,5),-2),
    ((2,4,3,7),-3),((3,13,2),1),((1,7,32),-1),
    ((2,2,2,2,2),0),((1,),0),((),0),
])
@pytest.mark.parametrize("layout",["contiguous","transpose","offset"])
def test_softmax_arbitrary_axis_forward_backward(shape,dim,layout):
    torch.manual_seed(62)
    x=torch.randn(shape)
    if layout=="transpose" and x.dim()>=2:x=x.transpose(0,1)
    elif layout=="offset" and x.dim()>=1:
        raw=torch.randn((x.shape[0]+1,*x.shape[1:]))
        x=raw[1:]
    if x.dim()==0 and dim in (0,-1):
        pass
    elif dim>=x.dim() or dim< -x.dim():return
    # The Vulkan inputs preserve actual stride+offset instead of turning
    # all layouts into easy contiguous test cases.
    original=torch.randn(shape).to("vulkan")
    if layout=="transpose" and original.dim()>=2:
        gpu=original.transpose(0,1)
        cpu=original.cpu().transpose(0,1)
    elif layout=="offset" and original.dim()>=1:
        value=torch.randn((shape[0]+1,*shape[1:]))
        cpu=value[1:]
        gpu=value.to("vulkan")[1:]
    else:
        cpu=x
        gpu=x.to("vulkan")
    cpu=cpu.detach().requires_grad_()
    gpu=gpu.detach().requires_grad_()
    expected=F.softmax(cpu,dim=dim)
    actual=F.softmax(gpu,dim=dim)
    torch.testing.assert_close(actual.cpu(),expected,atol=5e-6,rtol=4e-5)
    grad=torch.randn_like(expected)
    ref=torch.autograd.grad(expected,cpu,grad)[0]
    got=torch.autograd.grad(actual,gpu,grad.to("vulkan"))[0]
    torch.testing.assert_close(got.cpu(),ref,atol=6e-6,rtol=6e-5)


def test_softmax_non_last_axis_regression():
    x=torch.tensor([[[1.,2.],[20.,40.]],[[8.,9.],[3.,4.]]])
    for dim in range(x.dim()):
        expected=F.softmax(x,dim=dim)
        actual=F.softmax(x.to("vulkan"),dim=dim).cpu()
        torch.testing.assert_close(actual,expected,atol=4e-6,rtol=5e-5)


def test_softmax_zero_size_and_infinite_values():
    for shape in ((0,4),(2,0,3)):
        x=torch.empty(shape)
        for dim in range(x.dim()):
            torch.testing.assert_close(F.softmax(x.to("vulkan"),dim).cpu(),
                                       F.softmax(x,dim))
    x=torch.tensor([[float("-inf"),float("-inf"),0.],
                    [1.,4.,20.]])
    torch.testing.assert_close(F.softmax(x.to("vulkan"),1).cpu(),F.softmax(x,1),
                               equal_nan=True,atol=3e-6,rtol=3e-5)
