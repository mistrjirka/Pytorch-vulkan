"""BatchNorm parity across dimensionality, layouts, statistics and autograd."""
import itertools
import pytest
import torch
import torch_vulkan


def layout(x,kind,device):
    if kind=='channels_last': return x.to(device).contiguous(memory_format=torch.channels_last)
    if kind=='transpose': return x.transpose(-1,-2).contiguous().to(device).transpose(-1,-2)
    if kind=='slice':
        storage=torch.zeros((*x.shape[:-1],x.shape[-1]*2+3))
        storage[...,1:1+2*x.shape[-1]:2]=x
        return storage.to(device)[...,1:1+2*x.shape[-1]:2]
    if kind=='broadcast': return x[:1].to(device).expand(x.shape)
    return x.to(device)


@pytest.mark.parametrize('shape,kind',[
    ((7,5),'contiguous'),((2,3,11),'slice'),((2,5,7,9),'channels_last'),
    ((2,5,7,9),'transpose'),((2,3,4,5,7),'slice'),((3,2,4,7),'broadcast'),
    ((2,3,2,3,2,2),'transpose'),
])
@pytest.mark.parametrize('training,affine,stats',[(True,True,True),(True,False,False),(False,True,True),(False,False,True)])
def test_native_batch_norm(shape,kind,training,affine,stats):
    torch.manual_seed(90)
    x=torch.randn(shape);g=torch.randn(shape);c=shape[1]
    w=torch.randn(c) if affine else None;b=torch.randn(c) if affine else None
    rm=torch.randn(c) if stats else None;rv=torch.rand(c)+.3 if stats else None
    def execute(device):
        xx=layout(x,kind,device);gg=layout(g,kind,device)
        # Strided offset affine and running statistics exercise both reads and mutation.
        # The CPU optimized backward assumes contiguous channel vectors.
        # Use the same logical values contiguously for the reference.
        def vec(v):
            if v is None:return None
            return v.clone() if device=='cpu' else layout(v,'slice',device)
        ww,bb,mm,vv=map(vec,(w,b,rm,rv))
        out,mean,inv=torch.ops.aten.native_batch_norm(xx,ww,bb,mm,vv,training,.23,1e-4)
        grad=torch.ops.aten.native_batch_norm_backward(gg,xx,ww,mm,vv,mean,inv,training,1e-4,[True]*3)
        return (out,mean,inv,*grad,*([mm,vv] if stats else []))
    ref=execute('cpu');got=execute('vulkan')
    for a,b in zip(got,ref):torch.testing.assert_close(a.cpu(),b,rtol=3e-4,atol=3e-5)


@pytest.mark.parametrize('mask',list(itertools.product([False,True],repeat=3)))
@pytest.mark.parametrize('training',[False,True])
def test_batch_norm_masks(mask,training):
    torch.manual_seed(8)
    x=torch.randn(2,5,7);g=torch.randn_like(x);w=torch.randn(5);m=torch.randn(5);v=torch.rand(5)+.5
    def run(device):
        xx,gg,ww,mm,vv=[t.to(device) for t in (x,g,w,m,v)]
        _,sm,si=torch.ops.aten.native_batch_norm(xx,ww,None,mm,vv,training,.1,1e-5)
        return torch.ops.aten.native_batch_norm_backward(gg,xx,ww,mm,vv,sm,si,training,1e-5,mask)
    for yes,a,b in zip(mask,run('vulkan'),run('cpu')):
        if yes:torch.testing.assert_close(a.cpu(),b,rtol=3e-4,atol=3e-5)
        else:assert a is None


@pytest.mark.parametrize('offset,noise,eps',[(10000.,.1,1e-5),(3.,0.,0.),(3.,0.,1e-5)])
def test_batch_norm_stable_variance(offset,noise,eps):
    torch.manual_seed(4)
    x=(torch.randn(4,3,31,19)*noise+offset)
    # Double reference exposes cancellation rather than reproducing float CPU reduction error.
    ref=torch.ops.aten.native_batch_norm(x.double(),None,None,None,None,True,.1,eps)
    got=torch.ops.aten.native_batch_norm(x.to('vulkan'),None,None,None,None,True,.1,eps)
    torch.testing.assert_close(got[1].cpu(),ref[1].float(),rtol=0,atol=.001)
    torch.testing.assert_close(got[2].cpu(),ref[2].float(),rtol=3e-5,atol=1e-5)
    torch.testing.assert_close(got[0].cpu(),ref[0].float(),rtol=3e-4,atol=.008)


@pytest.mark.parametrize('ndim',[1,2,3])
@pytest.mark.parametrize('training',[True,False])
def test_batch_norm_module_chain(ndim,training):
    torch.manual_seed(77)
    cls=getattr(torch.nn,f'BatchNorm{ndim}d')
    cpu=torch.nn.Sequential(cls(3),torch.nn.ReLU(),cls(3)).train(training)
    vk=torch.nn.Sequential(cls(3),torch.nn.ReLU(),cls(3)).train(training).to('vulkan')
    vk.load_state_dict(cpu.state_dict())
    x=torch.randn((2,3)+((5,)*ndim));xc=x.clone().requires_grad_();xv=x.to('vulkan').requires_grad_()
    yc=cpu(xc);yv=vk(xv)
    yc.sum().backward();yv.sum().backward() # broadcast, zero-stride output gradient
    torch.testing.assert_close(yv.cpu(),yc,atol=3e-5,rtol=3e-4)
    torch.testing.assert_close(xv.grad.cpu(),xc.grad,atol=4e-5,rtol=4e-4)
    # The mathematically zero scale gradient sums 250 centered FP32 values.
    # Lavapipe and NVIDIA differ in reduction/FMA rounding (observed 5.6e-5).
    for pc,pv in zip(cpu.parameters(),vk.parameters()):torch.testing.assert_close(pv.grad.cpu(),pc.grad,atol=1e-4,rtol=4e-4)
    for bc,bv in zip(cpu.buffers(),vk.buffers()):torch.testing.assert_close(bv.cpu(),bc,atol=3e-5,rtol=3e-4)


def test_batch_norm_single_sample_and_no_cpu_fallback():
    x=torch.ones(1,3).to('vulkan');m=torch.zeros(3).to('vulkan');v=torch.ones(3).to('vulkan')
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
        y,mean,inv=torch.ops.aten.native_batch_norm(x,None,None,m,v,True,.1,1e-5)
        torch.ops.aten.native_batch_norm_backward(x,x,None,m,v,mean,inv,True,1e-5,[True]*3)
        torch_vulkan.synchronize()
    assert not any(e.key=='aten::_to_copy' for e in prof.key_averages())
    torch.testing.assert_close(y.cpu(),torch.zeros(1,3))
    assert v.cpu().isnan().all()
