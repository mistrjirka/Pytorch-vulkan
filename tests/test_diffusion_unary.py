"""FP32 native Vulkan diffusion unary math, strides, nonfinite and autograd."""
import os
import pytest
import torch
import torch.nn.functional as F
import torch_vulkan


OPS = [
    ("silu",F.silu),
    ("sin",torch.sin),
    ("cos",torch.cos),
    ("exp",torch.exp),
    ("sqrt",torch.sqrt),
    ("rsqrt",torch.rsqrt),
    ("tanh",torch.tanh),
    ("neg",torch.neg),
]


@pytest.mark.parametrize("opname,op",OPS)
@pytest.mark.parametrize("layout",["contiguous","transpose","offset","expanded"])
def test_unary_forward_and_gradient(opname,op,layout):
    torch.manual_seed(24)
    values=torch.randn(4,5,6)*0.7+0.9
    if layout=="transpose":
        raw=values.transpose(0,2)
    elif layout=="offset":
        raw=values[:,1:,2:]
    elif layout=="expanded":
        raw=values[:1].expand(4,-1,-1)
    else:
        raw=values
    cpu=raw.detach().requires_grad_()
    vk=values.to("vulkan")
    if layout=="transpose":vk=vk.transpose(0,2)
    elif layout=="offset":vk=vk[:,1:,2:]
    elif layout=="expanded":vk=vk[:1].expand(4,-1,-1)
    vk=vk.detach().requires_grad_()
    assert torch._C._dispatch_has_kernel_for_dispatch_key("aten::"+opname,"PrivateUse1")
    a=op(cpu)
    b=op(vk)
    torch.testing.assert_close(b.cpu(),a,rtol=3e-5,atol=3e-6,equal_nan=True)
    grads=torch.randn_like(a)
    ga=torch.autograd.grad(a,cpu,grads)[0]
    gb=torch.autograd.grad(b,vk,grads.to("vulkan"))[0]
    torch.testing.assert_close(gb.cpu(),ga,rtol=3e-4,atol=3e-5,equal_nan=True)


@pytest.mark.parametrize("opname,op",OPS)
def test_unary_special_values(opname,op):
    values=torch.tensor([-float("inf"),-50.,-20.,-3.,-0.,0.,3.,20.,50.,
                         float("inf"),float("nan")])
    cpu=op(values)
    actual=op(values.to("vulkan")).cpu()
    torch.testing.assert_close(actual,cpu,rtol=2e-4,atol=2e-6,equal_nan=True)
    empty=torch.empty(0,5)
    torch.testing.assert_close(op(empty.to("vulkan")).cpu(),op(empty))


@pytest.mark.parametrize("opname,op",OPS)
def test_unary_scalar_tensor(opname,op):
    x=torch.tensor(0.65,requires_grad=True)
    y=x.detach().to("vulkan").requires_grad_()
    a=op(x);b=op(y)
    torch.testing.assert_close(b.cpu(),a)
    ga=torch.autograd.grad(a,x)[0]
    gb=torch.autograd.grad(b,y)[0]
    torch.testing.assert_close(gb.cpu(),ga,rtol=3e-5,atol=3e-6,equal_nan=True)


def test_silu_backward_broadcast_gradient():
    torch.manual_seed(73)
    raw=torch.randn(2,3,5)
    x=raw.detach().requires_grad_()
    y=raw.to("vulkan").detach().requires_grad_()
    gy=torch.randn(1,3,5).expand(2,-1,-1)
    gradcpu=torch.autograd.grad(F.silu(x),x,gy)[0]
    gradvk=torch.autograd.grad(F.silu(y),y,gy.to("vulkan"))[0]
    torch.testing.assert_close(gradvk.cpu(),gradcpu,atol=5e-5,rtol=4e-5)
