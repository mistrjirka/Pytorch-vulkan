import pytest
import torch
import torch.nn.functional as F
import torch_vulkan

@pytest.mark.parametrize('shape',[(2,3,5,7,4,6),(1,2,137,19,8,3),(2,1,4,141,7,129)])
@pytest.mark.parametrize('kind',['none','causal','bool','float'])
def test_attention_forward_backward(shape,kind):
    torch.manual_seed(432)
    b,h,l,s,d,e=shape
    q,k,v=[torch.randn(size,requires_grad=True) for size in [(b,h,l,d),(b,h,s,d),(b,h,s,e)]]
    mask=None
    if kind=='bool':
        mask=torch.rand(l,s)>.3;mask[0]=False
    if kind=='float':mask=torch.randn(1,1,l,s,requires_grad=True)
    inputs=[x.detach().to('vulkan').requires_grad_() for x in (q,k,v)]
    mv=mask.detach().to('vulkan').requires_grad_(mask.requires_grad) if mask is not None else None
    a=F.scaled_dot_product_attention(q,k,v,mask,is_causal=kind=='causal',scale=.37)
    z=F.scaled_dot_product_attention(*inputs,mv,is_causal=kind=='causal',scale=.37)
    g=torch.randn_like(a);a.backward(g);z.backward(g.to('vulkan'))
    torch.testing.assert_close(z.cpu(),a,rtol=4e-4,atol=4e-5)
    for x,y in zip(inputs,(q,k,v)):torch.testing.assert_close(x.grad.cpu(),y.grad,rtol=6e-4,atol=7e-5)
    if mask is not None and mask.requires_grad:torch.testing.assert_close(mv.grad.cpu(),mask.grad,rtol=6e-4,atol=7e-5)

@pytest.mark.parametrize('gqa',[False,True])
def test_attention_broadcast_and_gqa(gqa):
    q=torch.randn(2,4,7,5,requires_grad=True);k=torch.randn(1,2 if gqa else 4,9,5,requires_grad=True);v=torch.randn(2,2 if gqa else 4,9,3,requires_grad=True)
    vs=[x.detach().to('vulkan').requires_grad_() for x in (q,k,v)]
    a=F.scaled_dot_product_attention(q,k,v,enable_gqa=gqa);b=F.scaled_dot_product_attention(*vs,enable_gqa=gqa)
    a.sum().backward();b.sum().backward();torch.testing.assert_close(b.cpu(),a,rtol=3e-4,atol=3e-5)
    for x,y in zip(vs,(q,k,v)):torch.testing.assert_close(x.grad.cpu(),y.grad,rtol=5e-4,atol=5e-5)

@pytest.mark.parametrize('p',[0.,.35,1.])
def test_attention_dropout_grad_finite_difference(p):
    q,k,v=[torch.randn(1,2,3,4,device='vulkan').requires_grad_() for _ in range(3)]
    torch_vulkan.manual_seed(42);out=F.scaled_dot_product_attention(q,k,v,dropout_p=p);out.sum().backward()
    # Use the identical Philox mask for both finite-difference evaluations.
    cpu=q.detach().cpu();eps=.005;cpu[0,0,1,2]+=eps
    torch_vulkan.manual_seed(42);plus=F.scaled_dot_product_attention(cpu.to('vulkan'),k,v,dropout_p=p).cpu().sum()
    cpu[0,0,1,2]-=2*eps
    torch_vulkan.manual_seed(42);minus=F.scaled_dot_product_attention(cpu.to('vulkan'),k,v,dropout_p=p).cpu().sum()
    torch.testing.assert_close(q.grad.cpu()[0,0,1,2],(plus-minus)/(2*eps),rtol=.02,atol=.001)

@pytest.mark.parametrize('batch,qlen,klen,dv',[(0,3,4,5),(2,0,4,5),(2,3,0,5),(2,3,4,0)])
def test_attention_empty_dimensions(batch,qlen,klen,dv):
    q=torch.randn(batch,qlen,7,requires_grad=True)
    k=torch.randn(batch,klen,7,requires_grad=True)
    v=torch.randn(batch,klen,dv,requires_grad=True)
    inputs=[x.detach().to('vulkan').requires_grad_() for x in (q,k,v)]
    expected=F.scaled_dot_product_attention(q,k,v)
    actual=F.scaled_dot_product_attention(*inputs)
    expected.sum().backward();actual.sum().backward()
    torch.testing.assert_close(actual.cpu(),expected)
    for a,b in zip(inputs,(q,k,v)):torch.testing.assert_close(a.grad.cpu(),b.grad)
