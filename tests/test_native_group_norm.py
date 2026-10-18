"""Native Vulkan GroupNorm: general shape/layout, forward, saved stats and backward."""
import os
import pytest
import torch
import torch.nn.functional as F
import torch_vulkan


@pytest.mark.parametrize("shape,groups", [
    ((2, 8, 4, 5),4),
    ((1, 8, 8, 8),1),
    ((1, 12, 3, 7),3),
    ((2, 16, 8),8),
    ((2, 8, 2, 3, 4),2),
    ((1, 8, 2, 2),8),
    ((2, 4, 1, 1),2),
])
@pytest.mark.parametrize("affine", [False, True])
def test_group_norm_matches_cpu_forward_backward(shape,groups,affine):
    torch.manual_seed(1000+sum(shape)+groups)
    x=torch.randn(*shape,requires_grad=True)
    xv=x.detach().to("vulkan").requires_grad_()
    channels=shape[1]
    if affine:
        w=torch.randn(channels,requires_grad=True)
        b=torch.randn(channels,requires_grad=True)
        wv=w.detach().to("vulkan").requires_grad_()
        bv=b.detach().to("vulkan").requires_grad_()
    else:
        w=b=wv=bv=None
    out=F.group_norm(x,groups,w,b,1e-5)
    actual=F.group_norm(xv,groups,wv,bv,1e-5)
    assert torch._C._dispatch_has_kernel_for_dispatch_key("aten::native_group_norm","PrivateUse1")
    assert torch._C._dispatch_has_kernel_for_dispatch_key("aten::native_group_norm_backward","PrivateUse1")
    torch.testing.assert_close(actual.cpu(),out,rtol=5e-4,atol=4e-5)
    grad=torch.randn_like(out)
    expected=torch.autograd.grad(out,tuple(v for v in (x,w,b) if v is not None),grad)
    grads=torch.autograd.grad(actual,tuple(v for v in (xv,wv,bv) if v is not None),
                              grad.to("vulkan"))
    for got,want in zip(grads,expected):
        torch.testing.assert_close(got.cpu(),want,rtol=8e-4,atol=7e-5)


@pytest.mark.parametrize("kind", ["transpose_spatial","channels_last","offset","expanded"])
def test_group_norm_packed_views(kind):
    torch.manual_seed(901)
    raw=torch.randn(2,8,6,7)
    if kind=="transpose_spatial":
        x=raw.transpose(-1,-2)
    elif kind=="channels_last":
        x=raw.contiguous(memory_format=torch.channels_last)
    elif kind=="offset":
        x=raw[:,:,:5,1:]
    else:
        x=raw[:1].expand(2,-1,-1,-1)
    source=raw.to("vulkan")
    if kind=="transpose_spatial":
        xv=source.transpose(-1,-2)
    elif kind=="channels_last":
        xv=source.contiguous(memory_format=torch.channels_last)
    elif kind=="offset":
        xv=source[:,:,:5,1:]
    else:
        xv=source[:1].expand(2,-1,-1,-1)
    x=x.detach().requires_grad_()
    xv=xv.detach().requires_grad_()
    out=F.group_norm(x,4)
    ov=F.group_norm(xv,4)
    torch.testing.assert_close(ov.cpu(),out,atol=2e-4,rtol=5e-4)
    grad=torch.randn_like(out)
    expected=torch.autograd.grad(out,x,grad)[0]
    actual=torch.autograd.grad(ov,xv,grad.to("vulkan"))[0]
    torch.testing.assert_close(actual.cpu(),expected,atol=3e-4,rtol=5e-4)


@pytest.mark.parametrize("mask", [(True,False,False),(False,True,False),(False,False,True),
                                   (True,True,True)])
def test_group_norm_output_mask(mask):
    torch.manual_seed(123)
    x=torch.randn(2,8,5,6)
    w=torch.randn(8)
    b=torch.randn(8)
    y,m,r=torch.ops.aten.native_group_norm(x,w,b,2,8,30,4,1e-5)
    xv,wv,bv=(t.to("vulkan") for t in (x,w,b))
    yv,mv,rv=torch.ops.aten.native_group_norm(xv,wv,bv,2,8,30,4,1e-5)
    torch.testing.assert_close(mv.cpu(),m,atol=2e-5,rtol=2e-4)
    torch.testing.assert_close(rv.cpu(),r,atol=2e-5,rtol=2e-4)
    go=torch.randn_like(y)
    expected=torch.ops.aten.native_group_norm_backward(go,x,m,r,w,2,8,30,4,mask)
    result=torch.ops.aten.native_group_norm_backward(
        go.to("vulkan"),xv,mv,rv,wv,2,8,30,4,mask)
    for want,got,needed in zip(expected,result,mask):
        if not needed:
            assert got is None
        else:
            assert got is not None
            torch.testing.assert_close(got.cpu(),want,atol=2e-4,rtol=5e-4)


def test_group_norm_nonzero_centering():
    torch.manual_seed(4)
    raw=torch.randn(2,8,8,8)*0.02+1000.
    xv=raw.to("vulkan")
    expected=F.group_norm(raw,4,eps=1e-3)
    actual=F.group_norm(xv,4,eps=1e-3)
    torch.testing.assert_close(actual.cpu(),expected,rtol=1e-2,atol=1e-2)
