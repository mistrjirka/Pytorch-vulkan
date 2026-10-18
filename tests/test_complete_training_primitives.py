import copy
import pytest
import torch
import torch.nn.functional as F
import torch_vulkan

@pytest.mark.parametrize('mode',['sum','mean','max'])
@pytest.mark.parametrize('last',[False,True])
@pytest.mark.parametrize('dtype',[torch.int32,torch.int64])
def test_embedding_bag(mode,last,dtype):
    torch.manual_seed(42)
    x=torch.randn(11,7,requires_grad=True);v=x.detach().to('vulkan').requires_grad_()
    indices=torch.tensor([2,3,0,3,7,2,4],dtype=dtype);offsets=torch.tensor([0,3,3,6]+([7] if last else []),dtype=dtype)
    a=F.embedding_bag(indices,x,offsets,mode=mode,padding_idx=0,include_last_offset=last)
    iv,ov=indices.to('vulkan'),offsets.to('vulkan')
    torch_vulkan.reset_execution_stats()
    b=F.embedding_bag(iv,v,ov,mode=mode,padding_idx=0,include_last_offset=last)
    b.sum().backward();stats=torch_vulkan.execution_stats();a.sum().backward()
    assert stats['cpu_fallback_calls']==0,stats
    assert stats['host_read_max_bytes']<=4,stats
    torch.testing.assert_close(b.cpu(),a,rtol=2e-5,atol=2e-6)
    torch.testing.assert_close(v.grad.cpu(),x.grad,rtol=2e-5,atol=2e-6)

@pytest.mark.parametrize('frequency',[False,True])
def test_embedding_bag_sample_grad(frequency):
    x=torch.randn(9,5,requires_grad=True);ps=torch.randn(5,requires_grad=True);i=torch.tensor([1,1,2,2,3]);o=torch.tensor([0,2])
    xv,pv=[t.detach().to('vulkan').requires_grad_() for t in (x,ps)]
    a=F.embedding_bag(i,x,o,mode='sum',per_sample_weights=ps,scale_grad_by_freq=frequency)
    b=F.embedding_bag(i.to('vulkan'),xv,o.to('vulkan'),mode='sum',per_sample_weights=pv,scale_grad_by_freq=frequency)
    a.sum().backward();b.sum().backward()
    torch.testing.assert_close(b.cpu(),a,rtol=2e-5,atol=2e-6)
    for z,w in ((xv,x),(pv,ps)):torch.testing.assert_close(z.grad.cpu(),w.grad,rtol=2e-5,atol=2e-6)

@pytest.mark.parametrize('op',[torch.tril,torch.triu])
@pytest.mark.parametrize('dtype',[torch.float32,torch.bool,torch.int32,torch.int64])
@pytest.mark.parametrize('diagonal',[-3,0,2])
def test_triangle(op,dtype,diagonal):
    x=torch.arange(60).reshape(3,4,5).to(dtype).transpose(1,2)
    torch.testing.assert_close(op(x.to('vulkan'),diagonal).cpu(),op(x,diagonal))

@pytest.mark.parametrize('amsgrad',[False,True])
def test_ordinary_adamw(amsgrad):
    torch.manual_seed(98)
    x=torch.nn.Parameter(torch.randn(7,9));v=torch.nn.Parameter(x.detach().to('vulkan'))
    a=torch.optim.AdamW([x],lr=.003,amsgrad=amsgrad,foreach=False)
    b=torch.optim.AdamW([v],lr=.003,amsgrad=amsgrad,foreach=False)
    for step in range(3):
        grad=torch.randn_like(x);x.grad=grad;v.grad=grad.to('vulkan')
        torch_vulkan.reset_execution_stats();b.step();stats=torch_vulkan.execution_stats();a.step()
        assert stats['cpu_fallback_calls']==0,stats
        assert stats['host_read_max_bytes']==0,stats
        torch.testing.assert_close(v.cpu(),x,rtol=2e-5,atol=2e-6)
        for key in ('exp_avg','exp_avg_sq'):
            torch.testing.assert_close(b.state[v][key].cpu(),a.state[x][key],rtol=2e-5,atol=2e-6)
