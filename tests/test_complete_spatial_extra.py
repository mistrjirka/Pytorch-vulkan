import pytest
import torch
import torch.nn.functional as F
import torch_vulkan

@pytest.mark.parametrize('kind',['avg','max','adaptive_avg','adaptive_max'])
@pytest.mark.parametrize('layout',['contiguous','channels_last','offset'])
@pytest.mark.parametrize('ceil',[False,True])
def test_pool3d(kind,layout,ceil):
    torch.manual_seed(132)
    x=torch.randn(2,3,6,7,8)
    if layout=='channels_last':x=x.contiguous(memory_format=torch.channels_last_3d)
    if layout=='offset':x=x[...,1:7:2]
    x=x.detach().requires_grad_();v=x.detach().to('vulkan').requires_grad_()
    if kind=='avg':fn=lambda a:F.avg_pool3d(a,(2,3,2),(2,2,1),(1,1,0),ceil_mode=ceil,count_include_pad=ceil)
    elif kind=='max':fn=lambda a:F.max_pool3d(a,(2,3,2),(2,2,1),(1,1,0),dilation=(2,1,1),ceil_mode=ceil)
    elif kind=='adaptive_avg':fn=lambda a:F.adaptive_avg_pool3d(a,(3,4,2))
    else:fn=lambda a:F.adaptive_max_pool3d(a,(3,4,2))
    a=fn(x);b=fn(v);g=torch.randn_like(a);a.backward(g);b.backward(g.to('vulkan'))
    torch.testing.assert_close(b.cpu(),a,rtol=2e-5,atol=2e-6)
    torch.testing.assert_close(v.grad.cpu(),x.grad,rtol=2e-5,atol=2e-6)

@pytest.mark.parametrize('kind',['trilinear','bilinear','bicubic'])
@pytest.mark.parametrize('align',[False,True])
@pytest.mark.parametrize('factor',[.6,1.7])
def test_resample(kind,align,factor):
    torch.manual_seed(91)
    x=torch.randn((2,3,5,7,9) if kind=='trilinear' else (2,3,7,9),requires_grad=True);v=x.detach().to('vulkan').requires_grad_()
    fn=lambda a:F.interpolate(a,scale_factor=factor,mode=kind,align_corners=align,antialias=kind!='trilinear')
    a=fn(x);b=fn(v);g=torch.randn_like(a);a.backward(g);b.backward(g.to('vulkan'))
    torch.testing.assert_close(b.cpu(),a,rtol=5e-5,atol=7e-6)
    torch.testing.assert_close(v.grad.cpu(),x.grad,rtol=6e-5,atol=1e-5)

@pytest.mark.parametrize('spatial',[1,2,3])
@pytest.mark.parametrize('mode',['reflect','replicate'])
@pytest.mark.parametrize('negative',[False,True])
def test_padding(spatial,mode,negative):
    torch.manual_seed(81)
    x=torch.randn((2,3)+tuple(range(5,5+spatial)),requires_grad=True);v=x.detach().to('vulkan').requires_grad_()
    pad=(1,-1 if negative else 2)*spatial
    a=F.pad(x,pad,mode);b=F.pad(v,pad,mode);g=torch.randn_like(a);a.backward(g);b.backward(g.to('vulkan'))
    torch.testing.assert_close(b.cpu(),a)
    torch.testing.assert_close(v.grad.cpu(),x.grad,rtol=2e-5,atol=2e-6)

@pytest.mark.parametrize('spatial,mode',[(2,'bilinear'),(2,'nearest'),(2,'bicubic'),(3,'bilinear'),(3,'nearest')])
@pytest.mark.parametrize('padding',['zeros','border','reflection'])
@pytest.mark.parametrize('align',[False,True])
def test_grid_sample(spatial,mode,padding,align):
    torch.manual_seed(32)
    x=torch.randn((2,3,5,6) if spatial==2 else (2,2,4,5,6),requires_grad=True)
    grid=(torch.rand((2,4,7,2) if spatial==2 else (2,3,4,5,3))*3-1.5).requires_grad_()
    xv,gv=[t.detach().to('vulkan').requires_grad_() for t in (x,grid)]
    a=F.grid_sample(x,grid,mode=mode,padding_mode=padding,align_corners=align);b=F.grid_sample(xv,gv,mode=mode,padding_mode=padding,align_corners=align)
    grad=torch.randn_like(a);a.backward(grad);b.backward(grad.to('vulkan'))
    torch.testing.assert_close(b.cpu(),a,rtol=5e-5,atol=1e-5)
    for v,t in ((xv,x),(gv,grid)):torch.testing.assert_close(v.grad.cpu(),t.grad,rtol=8e-5,atol=3e-5)
