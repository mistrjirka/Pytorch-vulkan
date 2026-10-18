import pytest
import torch
import torch_vulkan

@pytest.mark.parametrize('dtype',[torch.int32,torch.int64])
def test_integer_binary_and_reduction(dtype):
    x=torch.tensor([[1,-2,3],[-123456789,734,10000000]],dtype=dtype)
    if dtype==torch.int64:x=x*1234567891
    y=torch.tensor([3,-7,23],dtype=dtype)
    xv,yv=x.to('vulkan'),y.to('vulkan')
    torch_vulkan.reset_execution_stats()
    a=torch.add(xv,yv,alpha=3);b=xv*yv;s=xv.sum(0);s2=xv.t().sum(1,keepdim=True);s3=xv.sum()
    stats=torch_vulkan.execution_stats();assert stats['cpu_fallback_calls']==0 and stats['host_read_max_bytes']==0,stats
    for a,b in [(a,torch.add(x,y,alpha=3)),(b,x*y),(s,x.sum(0)),(s2,x.t().sum(1,keepdim=True)),(s3,x.sum())]:torch.testing.assert_close(a.cpu(),b)


def test_bool_sum_and_empty_integer_sum():
    x=torch.tensor([[True,False,True],[False,True,False]])
    torch.testing.assert_close(x.to('vulkan').sum(0).cpu(),x.sum(0))
    x=torch.empty(3,0,dtype=torch.int64)
    torch.testing.assert_close(x.to('vulkan').sum(1).cpu(),x.sum(1))
