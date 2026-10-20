"""Numerical, layout, alias and residency contracts for fused pointwise primitives."""
import os
import pytest
import torch
import torch_vulkan as vk

OPS=["lerp_scalar","lerp_tensor","addcmul","addcdiv","maximum","minimum"]

def apply(name,a,b,c,variant):
    if name=="lerp_scalar":
        if variant=="inplace":return a.lerp_(b,.31)
        return torch.lerp(a,b,.31,**({"out":a} if variant=="out" else {}))
    if name=="lerp_tensor":
        if variant=="inplace":return a.lerp_(b,c)
        return torch.lerp(a,b,c,**({"out":a} if variant=="out" else {}))
    if name in ["maximum","minimum"]:
        return getattr(torch,name)(a,b,**({"out":a} if variant!="functional" else {}))
    if variant=="inplace":return getattr(a,name+"_")(b,c,value=.37)
    return getattr(torch,name)(a,b,c,value=.37,**({"out":a} if variant=="out" else {}))

def pair(layout,seed):
    g=torch.Generator().manual_seed(seed)
    if layout=="contiguous":base=torch.randn(2,3,5,generator=g);view=lambda x:x
    elif layout=="transpose":base=torch.randn(2,3,5,generator=g);view=lambda x:x.transpose(1,2)
    elif layout=="offset":base=torch.randn(4,5,7,generator=g);view=lambda x:x[1:3,1:4,1:6]
    elif layout=="sliced":base=torch.randn(4,7,13,generator=g);view=lambda x:x[1::2,::2,1::3]
    elif layout=="rank8":base=torch.randn(2,1,2,1,2,1,2,3,generator=g);view=lambda x:x.transpose(0,7)
    elif layout=="scalar":base=torch.randn((),generator=g);view=lambda x:x
    elif layout=="empty":base=torch.randn(2,0,5,generator=g);view=lambda x:x
    else:raise ValueError(layout)
    base=base+1.7
    gpu=base.to("vulkan")
    return base,view(base),gpu,view(gpu)

@pytest.mark.parametrize("name",OPS)
@pytest.mark.parametrize("layout",["contiguous","transpose","offset","sliced","rank8","scalar","empty"])
@pytest.mark.parametrize("variant",["functional","out","inplace"])
def test_layouts_and_storage(name,layout,variant):
    a,ac,av,ag=pair(layout,1);b,bc,bv,bg=pair(layout,2);c,cc,cv,cg=pair(layout,3)
    expected=apply(name,ac,bc,cc,variant)
    vk.reset_execution_stats();actual=apply(name,ag,bg,cg,variant)
    stats=vk.execution_stats()
    assert stats["cpu_fallback_calls"]==stats["host_read_calls"]==0,stats
    torch.testing.assert_close(actual.cpu(),expected,rtol=3e-5,atol=3e-6)
    if variant!="functional":
        torch.testing.assert_close(av.cpu(),a,rtol=3e-5,atol=3e-6)
        assert actual is ag

@pytest.mark.parametrize("name",OPS)
@pytest.mark.parametrize("variant",["functional","out","inplace"])
def test_broadcast_and_noncontiguous_destination(name,variant):
    _,a,_,v=pair("transpose",4)
    b=torch.linspace(.4,1.4,3).reshape(1,1,3)
    c=torch.linspace(.1,.7,5).reshape(1,5,1)
    bv,cv=b.to("vulkan"),c.to("vulkan")
    expected=apply(name,a,b,c,variant)
    vk.reset_execution_stats();actual=apply(name,v,bv,cv,variant)
    assert vk.execution_stats()["host_read_calls"]==0
    assert vk.execution_stats()["cpu_fallback_calls"]==0
    torch.testing.assert_close(actual.cpu(),expected,rtol=3e-5,atol=3e-6)

@pytest.mark.parametrize("name",OPS)
def test_one_dispatch_for_contiguous_destination(name):
    if not all(vk.performance_options()[k] for k in ("gpu_resident", "deferred_submissions")):pytest.skip("deferred dispatch counter")
    a,b,c=[torch.randn(17,23).to("vulkan") for _ in range(3)]
    vk.synchronize();before=vk.queue_stats()
    apply(name,a,b,c,"out");vk.synchronize();after=vk.queue_stats()
    assert after["recorded_ops"]-before["recorded_ops"]==1

@pytest.mark.parametrize("name",OPS)
def test_partial_overlap_rejected(name):
    for device in ["cpu","vulkan"]:
        base=torch.arange(9.,device="cpu").to(device)
        a,b,c=base[:-1],torch.ones(8).to(device),torch.ones(8).to(device)
        with pytest.raises(RuntimeError,match="single memory location|single memory"):
            if name=="lerp_scalar":torch.lerp(a,b,.31,out=base[1:])
            elif name=="lerp_tensor":torch.lerp(a,b,c,out=base[1:])
            elif name in ["maximum","minimum"]:getattr(torch,name)(a,b,out=base[1:])
            else:getattr(torch,name)(a,b,c,out=base[1:])

@pytest.mark.parametrize("name",OPS)
def test_expanded_destination_rejected(name):
    for device in ["cpu","vulkan"]:
        a=torch.zeros(1).to(device).expand(2,3)
        b=torch.ones(2,3).to(device);c=b.clone()
        with pytest.raises(RuntimeError,match="single memory location"):
            apply(name,a,b,c,"out")

@pytest.mark.parametrize("name",OPS)
def test_lazy_clone_cow(name):
    a=torch.randn(3,7);b=torch.randn_like(a)+2;c=torch.randn_like(a)+2
    v,bv,cv=[x.to("vulkan") for x in (a,b,c)]
    clone=torch._lazy_clone(v)
    expected=apply(name,a.clone(),b,c,"inplace")
    vk.reset_execution_stats();actual=apply(name,v,bv,cv,"inplace")
    assert vk.execution_stats()["cpu_fallback_calls"]==0
    torch.testing.assert_close(actual.cpu(),expected,rtol=3e-5,atol=3e-6)
    torch.testing.assert_close(clone.cpu(),a)

@pytest.mark.parametrize("name",OPS)
def test_special_values_and_stride(name):
    values=torch.tensor([-float("inf"),-1e38,-1.,-0.,0.,1.,1e38,float("inf"),float("nan")])
    a=values[:,None].expand(9,9).clone().T;b=values[None,:].expand(9,9).clone().T
    c=torch.full_like(a,.81)
    av,bv,cv=[x.to("vulkan") for x in (a,b,c)]
    expected=apply(name,a,b,c,"functional");actual=apply(name,av,bv,cv,"functional").cpu()
    torch.testing.assert_close(actual,expected,rtol=3e-5,atol=3e-6,equal_nan=True)
    for fn in [torch.isnan,torch.isposinf,torch.isneginf]:torch.testing.assert_close(fn(actual),fn(expected))
    zeros=(actual==0)&(expected==0)
    # Mixed +/-zero extrema ties differ even between CPU scalar/SIMD and CUDA.
    # Equal-sign ties and every other operation still have a defined sign here.
    if name in ["maximum","minimum"]:
        zeros &= torch.signbit(a)==torch.signbit(b)
    torch.testing.assert_close(torch.signbit(actual[zeros]),torch.signbit(expected[zeros]))

@pytest.mark.parametrize("weight",[-2.,0.,.25,.5,.999,1.,2.,float("inf"),float("nan")])
def test_lerp_weight_formula(weight):
    a=torch.tensor([-float("inf"),-1e38,-1.,0.,1.,1e38,float("inf")])
    b=a.flip(0)
    actual=torch.lerp(a.to("vulkan"),b.to("vulkan"),weight).cpu()
    expected=torch.lerp(a,b,weight)
    torch.testing.assert_close(actual,expected,rtol=3e-5,atol=3e-6,equal_nan=True)

@pytest.mark.parametrize("name",["addcmul","addcdiv"])
@pytest.mark.parametrize("magnitude",[1e20,1e38])
@pytest.mark.parametrize("sign",[-1.,1.])
def test_scalar_multiply_precedes_product_or_division(name,magnitude,sign):
    a=torch.tensor([0.]);b=torch.tensor([magnitude*sign]);c=torch.tensor([1/magnitude if name=="addcmul" else magnitude])
    expected=getattr(torch,name)(a,b,c,value=magnitude)
    actual=getattr(torch,name)(a.to("vulkan"),b.to("vulkan"),c.to("vulkan"),value=magnitude)
    torch.testing.assert_close(actual.cpu(),expected)

@pytest.mark.parametrize("name",["lerp_scalar","lerp_tensor","addcmul","addcdiv"])
def test_broadcast_gradients(name):
    torch.manual_seed(27)
    cpu=[(torch.randn(s)+1.8).requires_grad_() for s in [(2,3,5),(1,3,1),(2,1,5)]]
    gpu=[x.detach().to("vulkan").requires_grad_() for x in cpu]
    expected=apply(name,*cpu,"functional");actual=apply(name,*gpu,"functional")
    grad=torch.randn_like(expected)
    expected.backward(grad);actual.backward(grad.to("vulkan"))
    for x,v in zip(cpu,gpu):
        if x.grad is None:assert v.grad is None
        else:torch.testing.assert_close(v.grad.cpu(),x.grad,rtol=4e-5,atol=4e-6)

@pytest.mark.parametrize("amsgrad",[False,True])
def test_adamw_multiple_deferred_updates_and_states(amsgrad):
    torch.manual_seed(31)
    cpu=torch.nn.Parameter(torch.randn(7,11).T)
    gpu=torch.nn.Parameter(cpu.detach().to("vulkan"))
    ref=torch.optim.AdamW([cpu],lr=.001,amsgrad=amsgrad,foreach=False)
    actual=torch.optim.AdamW([gpu],lr=.001,amsgrad=amsgrad,foreach=False)
    for _ in range(12):
        grad=torch.randn_like(cpu);cpu.grad=grad;gpu.grad=grad.to("vulkan")
        vk.reset_execution_stats();actual.step()
        assert vk.execution_stats()["cpu_fallback_calls"]==vk.execution_stats()["host_read_calls"]==0
        ref.step()
    vk.synchronize()
    torch.testing.assert_close(gpu.cpu(),cpu,rtol=3e-5,atol=3e-6)
    for key in ["exp_avg","exp_avg_sq"]+(["max_exp_avg_sq"] if amsgrad else []):
        torch.testing.assert_close(actual.state[gpu][key].cpu(),ref.state[cpu][key],rtol=3e-5,atol=3e-6)
