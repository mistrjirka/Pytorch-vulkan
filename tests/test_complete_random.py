import pytest
import torch
import torch.nn.functional as F
import torch_vulkan as vk

@pytest.mark.parametrize('factory',[torch.rand,torch.randn])
def test_device_rng_independence_and_reproducibility(factory):
    torch.manual_seed(751)
    cpu_state=torch.get_rng_state().clone()
    vk.manual_seed_all(111)
    a=factory(1003,device='vulkan').cpu()
    assert torch.equal(cpu_state,torch.get_rng_state())
    b=factory(1003,device='vulkan').cpu()
    assert not torch.equal(a,b)
    vk.manual_seed_all(111)
    assert torch.equal(a,factory(1003,device='vulkan').cpu())
    torch.manual_seed(712)
    c=factory(1003,device='vulkan').cpu()
    torch.manual_seed(712)
    assert torch.equal(c,factory(1003,device='vulkan').cpu())

@pytest.mark.parametrize('factory',[torch.rand,torch.randn])
def test_explicit_generator_state_clone(factory):
    a=torch.Generator(device='vulkan').manual_seed(2**63+47)
    b=a.clone_state()
    assert a.device==torch.device('vulkan:0')
    cpu=torch.get_rng_state().clone();default=vk.get_rng_state().clone()
    x=factory(39,device='vulkan',generator=a)
    y=factory(39,device='vulkan',generator=b)
    torch.testing.assert_close(x.cpu(),y.cpu(),rtol=0,atol=0)
    state=a.get_state()
    x=factory(97,device='vulkan',generator=a).cpu()
    a.set_state(state)
    torch.testing.assert_close(x,factory(97,device='vulkan',generator=a).cpu(),rtol=0,atol=0)
    assert torch.equal(cpu,torch.get_rng_state())
    assert torch.equal(default,vk.get_rng_state())
    with pytest.raises(RuntimeError):factory(3,device='vulkan',generator=torch.Generator())


def test_rng_statistics():
    vk.manual_seed(19)
    a=torch.rand(131071,device='vulkan').cpu()
    b=torch.randn(131071,device='vulkan').cpu()
    assert a.min()>=0 and a.max()<1
    assert abs(a.mean()-.5)<.004 and abs(a.var()-1/12)<.004
    assert abs(b.mean())<.015 and abs(b.var()-1)<.025


def test_philox_known_zero_vector():
    g=torch.Generator(device='vulkan').manual_seed(0)
    # Philox4x32-10 zero counter/key reference, checked independently of floats.
    expected=torch.tensor([0x6627e8d5>>8,0xe169c58d>>8,0xbc57ac4c>>8,0x9b00dbd8>>8],dtype=torch.float32)/(2**24)
    torch.testing.assert_close(torch.rand(4,device='vulkan',generator=g).cpu(),expected,rtol=0,atol=0)

@pytest.mark.parametrize('method',['uniform_','normal_','bernoulli_'])
def test_random_fill_strided_aliases(method):
    x=torch.full((5,13),17.,device='vulkan')
    y=x[1:4,1::2].t()
    vk.manual_seed(241)
    getattr(y,method)()
    host=x.cpu();mask=torch.zeros_like(host,dtype=torch.bool);mask[1:4,1::2]=True
    assert (host[~mask]==17).all()
    vk.manual_seed(241)
    ref=torch.empty(y.shape,device='vulkan');getattr(ref,method)()
    torch.testing.assert_close(y.cpu(),ref.cpu(),rtol=0,atol=0)

@pytest.mark.parametrize('p',[0.,.25,1.])
@pytest.mark.parametrize('training',[False,True])
def test_dropout_output_gradient_and_rng(p,training):
    x=torch.arange(103,dtype=torch.float32).to('vulkan').requires_grad_()
    vk.manual_seed(91);cpu=torch.get_rng_state().clone()
    y=F.dropout(x,p,training)
    grad=torch.linspace(.5,1.5,103)
    y.backward(grad.to('vulkan'))
    assert torch.equal(cpu,torch.get_rng_state())
    if not training or p==0:scale=torch.ones(103)
    elif p==1:scale=torch.zeros(103)
    else:scale=x.grad.cpu()/grad
    torch.testing.assert_close(y.cpu(),x.detach().cpu()*scale)
    torch.testing.assert_close(x.grad.cpu(),grad*scale)
    vk.manual_seed(91)
    torch.testing.assert_close(y.cpu(),F.dropout(x.detach(),p,training).cpu(),rtol=0,atol=0)


def test_rng_fork_state_restoration():
    before=vk.get_rng_state()
    with torch.random.fork_rng(devices=[0],device_type='vulkan'):
        vk.manual_seed(991);torch.randn(63,device='vulkan')
    assert torch.equal(vk.get_rng_state(),before)


@pytest.mark.parametrize('method',['uniform_','normal_','bernoulli_'])
def test_random_partial_write_preserves_uploaded_storage(method):
    # Start from a host-written allocation, not a preceding GPU fill.
    x=torch.full((5,13),17.).to('vulkan');view=x[1:4,1::2].t()
    getattr(view,method)()
    host=x.cpu();mask=torch.zeros_like(host,dtype=torch.bool);mask[1:4,1::2]=True
    assert (host[~mask]==17).all()


@pytest.mark.parametrize('dtype,low,high',[(torch.int32,-9,13),(torch.int32,-2**31,2**31),(torch.int64,-9,13),(torch.int64,-2**62,2**62+17),(torch.int64,2**60,2**60+3)])
def test_integer_rng_ranges_and_state(dtype,low,high):
    cpu=torch.get_rng_state().clone();g=torch.Generator(device='vulkan').manual_seed(71);state=g.get_state()
    a=torch.randint(low,high,(4097,),dtype=dtype,device='vulkan',generator=g).cpu()
    g.set_state(state);b=torch.randint(low,high,(4097,),dtype=dtype,device='vulkan',generator=g).cpu()
    assert torch.equal(a,b) and torch.equal(cpu,torch.get_rng_state())
    assert bool((a.long()>=low).all()) and bool((a.long()<high).all())
    if high-low==3:
        counts=torch.bincount(a-low,minlength=3).float()/len(a)
        assert bool((counts-1/3).abs().lt(.035).all())


def test_integer_rng_full_signed_range_and_partial_storage():
    x=torch.full((9,11),23,dtype=torch.int64).to('vulkan');view=x[1::2,1::2]
    view.random_(-2**63,None)
    actual=x.cpu();mask=torch.zeros_like(actual,dtype=torch.bool);mask[1::2,1::2]=True
    assert (actual[~mask]==23).all() and (actual[mask]<0).any() and (actual[mask]>=0).any()
    with pytest.raises(RuntimeError):torch.randint(4,4,(3,),device='vulkan')
