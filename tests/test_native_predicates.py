import pytest
import torch
import torch.nn.functional as F
import torch_vulkan

@pytest.mark.parametrize('shape',[(1,),(3,),(5,),(2,3,7),(),(0,)])
@pytest.mark.parametrize('kind',['isneginf','gt','eq','ge','le','lt'])
def test_predicates(shape,kind):
    x=torch.randn(shape)
    if x.numel():x.flatten()[0]=-float('inf')
    if x.numel()>1:x.flatten()[1]=float('nan')
    fn=lambda a:torch.isneginf(a) if kind=='isneginf' else getattr(torch,kind)(a,0.)
    torch.testing.assert_close(fn(x.to('vulkan')).cpu(),fn(x))

@pytest.mark.parametrize('shape',[(2,3,7),(2,0,5),(),(0,)])
@pytest.mark.parametrize('axis',[None,0,-1])
@pytest.mark.parametrize('keep',[False,True])
def test_all(shape,axis,keep):
    x=torch.rand(shape)>.3
    fn=lambda a:a.all() if axis is None else a.all(axis,keepdim=keep)
    torch.testing.assert_close(fn(x.to('vulkan')).cpu(),fn(x))

@pytest.mark.parametrize('offset',[0,1,2,3,4])
def test_byte_offsets_and_broadcast(offset):
    raw=torch.randn(3,11)
    c=raw>0;cv=c.to('vulkan')
    c=c[:,offset:offset+7].t();cv=cv[:,offset:offset+7].t()
    a=torch.randn(1,3,requires_grad=True);b=torch.randn(7,1,requires_grad=True)
    av=a.detach().to('vulkan').requires_grad_();bv=b.detach().to('vulkan').requires_grad_()
    y=torch.where(c,a,b);yv=torch.where(cv,av,bv)
    torch.testing.assert_close(yv.cpu(),y)
    grad=torch.randn_like(y)
    expected=torch.autograd.grad(y,(a,b),grad);actual=torch.autograd.grad(yv,(av,bv),grad.to('vulkan'))
    for x,z in zip(actual,expected):torch.testing.assert_close(x.cpu(),z,atol=1e-5,rtol=1e-5)
    torch.testing.assert_close(cv.all(1).cpu(),c.all(1))
    torch.testing.assert_close((cv&cv[:,0:1]).cpu(),c&c[:,0:1])
    torch.testing.assert_close((~cv).cpu(),~c)

@pytest.mark.parametrize('exponent',[0.,1.,2.,3.,.5,-.5,-1.,-2.,-3.,2.5,float('nan'),float('inf'),-float('inf')])
def test_power_special_values(exponent):
    x=torch.tensor([-float('inf'),-3.,-1.,-0.,0.,.1,1.,2.,float('inf'),float('nan')])
    torch.testing.assert_close(x.to('vulkan').pow(exponent).cpu(),x.pow(exponent),rtol=2e-5,atol=2e-6,equal_nan=True)

@pytest.mark.parametrize('low,high',[(-.5,.7),(None,.3),(-.4,None),(2.,-1.),(float('nan'),1.)])
def test_clamp_forward_backward(low,high):
    x=torch.randn(2,3,5,requires_grad=True);v=x.detach().to('vulkan').requires_grad_()
    a=x.clamp(low,high);b=v.clamp(low,high)
    torch.testing.assert_close(b.cpu(),a,equal_nan=True)
    g=torch.randn_like(a)
    torch.testing.assert_close(torch.autograd.grad(b,v,g.to('vulkan'))[0].cpu(),torch.autograd.grad(a,x,g)[0],equal_nan=True)


def test_attention_no_generic_fallback(monkeypatch,capfd):
    monkeypatch.setenv('TORCH_VULKAN_TRACE_FALLBACK','1')
    torch.manual_seed(31)
    x=torch.randn(2,3,5,7,requires_grad=True);v=x.detach().to('vulkan').requires_grad_()
    a=F.scaled_dot_product_attention(x,x,x);b=F.scaled_dot_product_attention(v,v,v)
    g=torch.randn_like(a)
    grad=torch.autograd.grad(b,v,g.to('vulkan'))[0]
    torch.testing.assert_close(b.cpu(),a,atol=2e-5,rtol=5e-4)
    torch.testing.assert_close(grad.cpu(),torch.autograd.grad(a,x,g)[0],atol=3e-5,rtol=5e-4)
    assert 'PV_FALLBACK' not in capfd.readouterr().err

@pytest.mark.parametrize('layout',['alias','offset','transpose','empty'])
def test_where_out(layout):
    torch.manual_seed(8)
    x=torch.randn(3,5);c=x>0;y=torch.randn_like(x)
    xv=x.to('vulkan');cv=c.to('vulkan');yv=y.to('vulkan')
    if layout=='alias':out=xv
    elif layout=='offset':out=torch.full((3,7),42.).to('vulkan')[:,1:6]
    elif layout=='transpose':out=torch.empty(5,3,device='vulkan').t()
    else:out=torch.empty(0,device='vulkan')
    actual=torch.where(cv,xv,yv,out=out)
    assert actual is out
    torch.testing.assert_close(actual.cpu(),torch.where(c,x,y))
    if layout=='offset':
        storage=out.as_strided((3,7),(7,1),0).cpu()
        assert (storage[:,0]==42).all() and (storage[:,-1]==42).all()


def test_where_partial_overlap_rejected():
    x=torch.randn(8).to('vulkan');c=(x>0)[:7]
    with pytest.raises(RuntimeError):torch.where(c,x[:7],x[:7],out=x[1:])

@pytest.mark.parametrize('offset',[False,True])
def test_logical_and_out_alias_and_tail(offset):
    x=torch.rand(3,5)>.3;y=torch.rand(3,5)>.4
    xv=x.to('vulkan');yv=y.to('vulkan')
    if offset:out=torch.ones(3,7,dtype=torch.bool).to('vulkan')[:,1:6]
    else:out=xv
    torch.logical_and(xv,yv,out=out)
    torch.testing.assert_close(out.cpu(),x&y)
    if offset:
        storage=out.as_strided((3,7),(7,1),0).cpu()
        assert storage[:,0].all() and storage[:,-1].all()
