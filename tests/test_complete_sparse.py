import pytest
import torch
import torch_vulkan

@pytest.mark.parametrize('shape,sd',[((3,5),2),((2,3,4),2),((2,3,4),1),((),0),((0,5),2),((3,0),1)])
def test_sparse_roundtrip(shape,sd):
    torch.manual_seed(51)
    x=torch.randn(shape);x[x.abs()<.6]=0
    expected=x.to_sparse(sd);v=x.to('vulkan')
    torch_vulkan.reset_execution_stats()
    s=v.to_sparse(sd);dense=s.to_dense();stats=torch_vulkan.execution_stats()
    assert s.device.type=='vulkan' and s.layout==torch.sparse_coo
    assert stats['cpu_fallback_calls']==0 and stats['host_read_max_bytes']<=4,stats
    torch.testing.assert_close(s.indices().cpu(),expected.indices())
    torch.testing.assert_close(s.values().cpu(),expected.values())
    torch.testing.assert_close(dense.cpu(),x)


def test_sparse_constructor():
    i=torch.tensor([[0,1,1],[2,0,3]]);v=torch.tensor([1.,2.,3.])
    s=torch.sparse_coo_tensor(i.to('vulkan'),v.to('vulkan'),(3,4),is_coalesced=True,check_invariants=True)
    expected=torch.sparse_coo_tensor(i,v,(3,4),is_coalesced=True,check_invariants=True)
    torch.testing.assert_close(s.to_dense().cpu(),expected.to_dense())


def test_sparse_roundtrip_gradient():
    x=torch.randn(3,5,requires_grad=True);v=x.detach().to('vulkan').requires_grad_()
    a=x.to_sparse().to_dense();b=v.to_sparse().to_dense();g=torch.randn_like(x)
    a.backward(g);b.backward(g.to('vulkan'))
    torch.testing.assert_close(v.grad.cpu(),x.grad)


def test_coalesce_duplicates():
    i=torch.tensor([[2,0,2,1,0],[1,3,1,2,3]]);v=torch.randn(5,4)
    expected=torch.sparse_coo_tensor(i,v,(3,5,4),check_invariants=True).coalesce()
    actual=torch.sparse_coo_tensor(i.to('vulkan'),v.to('vulkan'),(3,5,4),check_invariants=True).coalesce()
    torch.testing.assert_close(actual.indices().cpu(),expected.indices())
    torch.testing.assert_close(actual.values().cpu(),expected.values())
    torch.testing.assert_close(actual.to_dense().cpu(),expected.to_dense())


def test_sparse_invalid_indices():
    with pytest.raises(RuntimeError):
        torch.sparse_coo_tensor(torch.tensor([[0,5],[0,0]],device='vulkan'),torch.ones(2,device='vulkan'),(3,4),check_invariants=True)


def test_sparse_inferred_shape_and_constructor_gradient():
    i=torch.tensor([[2,0,2,1,0],[1,3,1,2,3]])
    v=torch.randn(5,requires_grad=True);vv=v.detach().to('vulkan').requires_grad_()
    a=torch.sparse_coo_tensor(i,v,check_invariants=True).coalesce().to_dense()
    b=torch.sparse_coo_tensor(i.to('vulkan'),vv,check_invariants=True).coalesce().to_dense()
    g=torch.randn_like(a);a.backward(g);b.backward(g.to('vulkan'))
    torch.testing.assert_close(b.cpu(),a)
    torch.testing.assert_close(vv.grad.cpu(),v.grad)
