"""General geometry, boundary conventions, strided views and training parity."""
import itertools
import pytest
import torch
import torch.nn.functional as F
import torch_vulkan


def pair(shape,layout='plain'):
    torch.manual_seed(sum(shape))
    x=torch.randn(shape)
    v=x.to('vulkan')
    if layout=='transpose':x=x.transpose(-1,-2);v=v.transpose(-1,-2)
    if layout=='offset':x=x[...,1:];v=v[...,1:]
    if layout=='channels_last':x=x.contiguous(memory_format=torch.channels_last);v=v.contiguous(memory_format=torch.channels_last)
    if layout=='expand':x=x[:1].expand(3,*x.shape[1:]);v=v[:1].expand(3,*v.shape[1:])
    return x.detach().requires_grad_(),v.detach().requires_grad_()


def check(x,v,fn,atol=2e-5):
    a,b=fn(x),fn(v)
    torch.testing.assert_close(b.cpu(),a,rtol=4e-4,atol=atol,equal_nan=True)
    g=torch.randn_like(a)
    ga=torch.autograd.grad(a,x,g)[0]
    gb=torch.autograd.grad(b,v,g.to('vulkan'))[0]
    torch.testing.assert_close(gb.cpu(),ga,rtol=6e-4,atol=atol,equal_nan=True)


@pytest.mark.parametrize('mode,align',list(itertools.product(['bilinear','bicubic'],[False,True])))
@pytest.mark.parametrize('shape,size,scale',[
 ((2,3,4,7),(9,3),None),((1,2,1,5),(1,1),None),((1,1,1,1),(5,7),None),
 ((1,2,8,9),(2,3),None),((1,3,3,2),None,(2.3,1.7)),
 ((1,2,7,11),None,(.6,.7)),((0,2,4,5),(9,8),None)])
def test_resize(mode,align,shape,size,scale):
    x,v=pair(shape)
    check(x,v,lambda a:F.interpolate(a,size=size,scale_factor=scale,mode=mode,align_corners=align),8e-5)


@pytest.mark.parametrize('mode',['bilinear','bicubic'])
@pytest.mark.parametrize('layout',['transpose','offset','channels_last','expand'])
def test_resize_layout(mode,layout):
    x,v=pair((2,3,5,7),layout)
    check(x,v,lambda a:F.interpolate(a,size=(8,11),mode=mode,align_corners=False),8e-5)


@pytest.mark.parametrize('shape,kernel,stride,pad',[
 ((2,3,5,7),(3,2),(2,1),(1,1)),((3,4,6),3,2,1),
 ((1,2,2,3),2,2,0),((1,2,11,13),9,3,4),((0,2,5,7),3,2,1)])
@pytest.mark.parametrize('ceil,include,divisor',list(itertools.product([False,True],[False,True],[None,3,-7])))
def test_average_pool(shape,kernel,stride,pad,ceil,include,divisor):
    x,v=pair(shape)
    check(x,v,lambda a:F.avg_pool2d(a,kernel,stride,pad,ceil,include,divisor))


@pytest.mark.parametrize('size',[(1,1),(3,4),(9,11)])
@pytest.mark.parametrize('layout',['plain','transpose','offset','channels_last','expand'])
def test_adaptive_pool(size,layout):
    x,v=pair((2,3,5,7),layout)
    check(x,v,lambda a:F.adaptive_avg_pool2d(a,size))


@pytest.mark.parametrize('op',['sum','mean'])
@pytest.mark.parametrize('shape,dims',[
 ((2,3,5),(0,2)),((2,3,65),(1,2)),((2,3,5),(-1,)),((2,3,5),()),
 ((2,0,3),(1,)),((0,2,3),(1,)),((),(0,)),((2,3,4,5),(0,2,3))])
@pytest.mark.parametrize('keep',[False,True])
def test_reductions(op,shape,dims,keep):
    x,v=pair(shape)
    check(x,v,lambda a:getattr(a,op)(dim=dims,keepdim=keep))


@pytest.mark.parametrize('op',['sum','mean'])
@pytest.mark.parametrize('layout',['transpose','offset','channels_last','expand'])
def test_reduce_layout(op,layout):
    x,v=pair((2,3,5,7),layout)
    check(x,v,lambda a:getattr(a,op)(dim=(0,2)))


def test_duplicate_axes_rejected():
    x=torch.randn(2,3).to('vulkan')
    with pytest.raises(RuntimeError):x.sum((1,1))
    with pytest.raises(RuntimeError):x.mean((1,1))
