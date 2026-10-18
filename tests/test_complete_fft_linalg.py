import pytest
import torch
import torch_vulkan

@pytest.mark.parametrize('n',[1,7,8,11,16,64])
@pytest.mark.parametrize('norm',['backward','forward','ortho'])
@pytest.mark.parametrize('kind',['fft','ifft','rfft','irfft'])
def test_fft_forward_backward(n,norm,kind):
    torch.manual_seed(321)
    shape=(3,n//2+1 if kind=='irfft' else n)
    dtype=torch.complex64 if kind in ('ifft','irfft') else torch.float32
    x=torch.randn(shape,dtype=dtype,requires_grad=True)
    v=x.detach().to('vulkan').requires_grad_();fn=getattr(torch.fft,kind)
    a=fn(x,n=n,norm=norm);b=fn(v,n=n,norm=norm)
    g=torch.randn_like(a);a.backward(g);b.backward(g.to('vulkan'))
    torch.testing.assert_close(b.cpu(),a,rtol=3e-4,atol=4e-5)
    torch.testing.assert_close(v.grad.cpu(),x.grad,rtol=4e-4,atol=8e-5)

@pytest.mark.parametrize('kind',['fftn','ifftn','rfftn','irfftn'])
@pytest.mark.parametrize('dims',[(0,2),(2,0),(1,2)])
def test_multidimensional_fft(kind,dims):
    torch.manual_seed(42)
    shape=[3,4,8];sizes=[shape[d] for d in dims]
    if kind=='irfftn':shape[dims[-1]]=shape[dims[-1]]//2+1
    x=torch.randn(shape,dtype=torch.complex64 if kind in ('ifftn','irfftn') else torch.float32,requires_grad=True)
    v=x.detach().to('vulkan').requires_grad_();fn=getattr(torch.fft,kind)
    a=fn(x,s=sizes,dim=dims,norm='ortho');b=fn(v,s=sizes,dim=dims,norm='ortho')
    g=torch.randn_like(a);a.backward(g);b.backward(g.to('vulkan'))
    torch.testing.assert_close(b.cpu(),a,rtol=4e-4,atol=5e-5)
    torch.testing.assert_close(v.grad.cpu(),x.grad,rtol=5e-4,atol=8e-5)

@pytest.mark.parametrize('batch',[(),(2,),(2,1)])
@pytest.mark.parametrize('left',[False,True])
def test_solve_forward_backward(batch,left):
    torch.manual_seed(123)
    a=torch.randn((*batch,5,5));a=(a@a.mT+torch.eye(5)*2).requires_grad_()
    b=torch.randn((*batch,5,3) if left else (*batch,3,5),requires_grad=True)
    av,bv=[x.detach().to('vulkan').requires_grad_() for x in (a,b)]
    e=torch.linalg.solve(a,b,left=left);v=torch.linalg.solve(av,bv,left=left)
    g=torch.randn_like(e);e.backward(g);v.backward(g.to('vulkan'))
    torch.testing.assert_close(v.cpu(),e,rtol=2e-4,atol=2e-5)
    for x,y in [(av,a),(bv,b)]:torch.testing.assert_close(x.grad.cpu(),y.grad,rtol=3e-4,atol=3e-5)

@pytest.mark.parametrize('adjoint',[False,True])
def test_lu_pivot_broadcast_solve(adjoint):
    a=torch.tensor([[0.,1.,2.],[2.,3.,4.],[1.,7.,9.]])
    b=torch.randn(2,3,4)
    lu,piv,info=torch.linalg.lu_factor_ex(a)
    lv,pv,iv=torch.linalg.lu_factor_ex(a.to('vulkan'))
    torch.testing.assert_close(lv.cpu(),lu,rtol=1e-5,atol=1e-6);torch.testing.assert_close(pv.cpu(),piv);torch.testing.assert_close(iv.cpu(),info)
    expected=torch.linalg.lu_solve(lu,piv,b,adjoint=adjoint)
    actual=torch.linalg.lu_solve(lv,pv,b.to('vulkan'),adjoint=adjoint)
    torch.testing.assert_close(actual.cpu(),expected,rtol=3e-4,atol=2e-5)


def test_solve_vector_and_singular():
    a=torch.eye(4)*2;b=torch.arange(4.)
    torch.testing.assert_close(torch.linalg.solve(a.to('vulkan'),b.to('vulkan')).cpu(),torch.linalg.solve(a,b))
    with pytest.raises(torch.linalg.LinAlgError):torch.linalg.solve(torch.zeros(3,3,device='vulkan'),torch.ones(3,device='vulkan'))


def test_spectral_filter_training():
    torch.manual_seed(123)
    x=torch.randn(3,16,requires_grad=True);w=torch.randn(9,requires_grad=True)
    xv,wv=[a.detach().to('vulkan').requires_grad_() for a in (x,w)]
    def fn(a,b):return torch.fft.irfft(torch.fft.rfft(a)*b,n=16)
    expected=fn(x,w);torch_vulkan.reset_execution_stats();actual=fn(xv,wv);actual.square().sum().backward();stats=torch_vulkan.execution_stats();expected.square().sum().backward()
    assert stats['cpu_fallback_calls']==0 and stats['host_read_max_bytes']==0,stats
    torch.testing.assert_close(actual.cpu(),expected,rtol=3e-4,atol=3e-5)
    for a,b in ((xv,x),(wv,w)):torch.testing.assert_close(a.grad.cpu(),b.grad,rtol=4e-4,atol=5e-5)


@pytest.mark.parametrize('transpose',[False,True])
def test_complex_conjugate_copy_and_arithmetic(transpose):
    x=torch.randn(3,5,dtype=torch.complex64);y=torch.randn(3,5,dtype=torch.complex64)
    if transpose:x=x.t();y=y.t()
    a=x.to('vulkan');b=y.to('vulkan')
    torch_vulkan.reset_execution_stats()
    result=a.conj()*b;clone=a.conj().clone();out=torch.empty_like(a);out.conj().copy_(b)
    stats=torch_vulkan.execution_stats();assert stats['cpu_fallback_calls']==0 and stats['host_read_max_bytes']==0,stats
    torch.testing.assert_close(result.cpu(),x.conj()*y)
    torch.testing.assert_close(clone.cpu(),x.conj())
    torch.testing.assert_close(out.cpu(),y.conj())


def test_lu_empty_and_invalid_pivots():
    a=torch.empty(2,0,0,device='vulkan')
    lu,pivots,info=torch.linalg.lu_factor_ex(a)
    assert lu.shape==(2,0,0) and pivots.shape==(2,0)
    torch.testing.assert_close(info.cpu(),torch.zeros(2,dtype=torch.int32))
    a=torch.eye(3,device='vulkan');lu,p=torch.linalg.lu_factor(a)
    with pytest.raises(RuntimeError):
        torch.linalg.lu_solve(lu,torch.tensor([1,4,3],dtype=torch.int32,device='vulkan'),torch.ones(3,2,device='vulkan'))


@pytest.mark.parametrize('dtype',[torch.float32,torch.int32,torch.int64,torch.complex64])
def test_eye_factory_and_strided_output(dtype):
    torch.testing.assert_close(torch.eye(3,5,device='vulkan',dtype=dtype).cpu(),torch.eye(3,5,dtype=dtype))
    base=torch.full((7,11),17,dtype=dtype).to('vulkan');out=base[1:7:2,1:11:2]
    torch.eye(3,5,out=out)
    expected=torch.full((7,11),17,dtype=dtype);expected[1:7:2,1:11:2]=torch.eye(3,5,dtype=dtype)
    torch.testing.assert_close(base.cpu(),expected)


@pytest.mark.parametrize('dtype',[torch.float32,torch.complex64])
def test_negative_view_copy(dtype):
    x=torch.randn(3,5,dtype=dtype);v=x.to('vulkan')
    a=torch._neg_view(x).t().clone();b=torch._neg_view(v).t().clone()
    torch.testing.assert_close(b.cpu(),a)
