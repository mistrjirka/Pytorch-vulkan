import pytest
import torch
import torch.nn.functional as F
import torch_vulkan

@pytest.mark.parametrize('dim',[0,1,2,-1])
@pytest.mark.parametrize('operation',['gather','scatter_add','scatter','index_select','index_add'])
def test_indexed_forward_backward(dim,operation):
    torch.manual_seed(52)
    x=torch.randn(3,5,7,requires_grad=True);d=dim%3
    shape=list(x.shape);shape[d]=2
    source=torch.randn(shape,requires_grad=True)
    # Unique per slice for scatter overwrite: its duplicate-index derivative is undefined.
    view=[1]*3;view[d]=2
    index=torch.tensor([0,x.size(d)-1]).reshape(view).expand(shape).clone()
    xv=x.detach().to('vulkan').requires_grad_();sv=source.detach().to('vulkan').requires_grad_();iv=index.to('vulkan')
    if operation=='gather':a=torch.gather(x,dim,index);b=torch.gather(xv,dim,iv)
    elif operation=='scatter_add':a=x.scatter_add(dim,index,source);b=xv.scatter_add(dim,iv,sv)
    elif operation=='scatter':a=x.scatter(dim,index,source);b=xv.scatter(dim,iv,sv)
    elif operation=='index_select':a=x.index_select(dim,torch.tensor([0,x.size(d)-1]));b=xv.index_select(dim,torch.tensor([0,x.size(d)-1],device='vulkan'))
    else:a=x.index_add(dim,torch.tensor([0,x.size(d)-1]),source,alpha=.7);b=xv.index_add(dim,torch.tensor([0,x.size(d)-1],device='vulkan'),sv,alpha=.7)
    grad=torch.randn_like(a);a.backward(grad);b.backward(grad.to('vulkan'))
    torch.testing.assert_close(b.cpu(),a,rtol=1e-5,atol=1e-6)
    torch.testing.assert_close(xv.grad.cpu(),x.grad,rtol=1e-5,atol=1e-6)
    if source.grad is not None:torch.testing.assert_close(sv.grad.cpu(),source.grad,rtol=1e-5,atol=1e-6)

@pytest.mark.parametrize('dtype',[torch.int32,torch.int64])
@pytest.mark.parametrize('frequency',[False,True])
@pytest.mark.parametrize('padding',[None,0,6])
def test_embedding_dense_backward(dtype,frequency,padding):
    torch.manual_seed(52)
    x=torch.randn(7,13,requires_grad=True);v=x.detach().to('vulkan').requires_grad_()
    i=torch.tensor([[0,1,2,1,0],[1,1,3,6,1]],dtype=dtype)
    a=F.embedding(i,x,padding_idx=padding,scale_grad_by_freq=frequency)
    b=F.embedding(i.to('vulkan'),v,padding_idx=padding,scale_grad_by_freq=frequency)
    g=torch.randn_like(a);a.backward(g);b.backward(g.to('vulkan'))
    torch.testing.assert_close(b.cpu(),a)
    torch.testing.assert_close(v.grad.cpu(),x.grad,rtol=1e-5,atol=2e-6)

@pytest.mark.parametrize('op',['scatter_add_','scatter_'])
def test_indexed_mutation_strided_storage(op):
    x=torch.arange(60.).reshape(5,12);v=x.to('vulkan')
    a,b=x[1:4,1::2],v[1:4,1::2]
    i=torch.tensor([[0,2,4],[1,3,5],[2,3,4]]);s=torch.randn(3,3)
    ver=v._version;getattr(a,op)(1,i,s);getattr(b,op)(1,i.to('vulkan'),s.to('vulkan'))
    assert v._version==ver+1;torch.testing.assert_close(v.cpu(),x)

@pytest.mark.parametrize('shape,dim',[((3,7),1),((3,5,7),0),((3,5,7),1),((3,5,7),2),((2,257),1)])
@pytest.mark.parametrize('descending',[False,True])
def test_sort_values_indices_gradients(shape,dim,descending):
    torch.manual_seed(65)
    x=torch.randn(shape,requires_grad=True);v=x.detach().to('vulkan').requires_grad_()
    a,i=torch.sort(x,dim=dim,descending=descending,stable=True);b,j=torch.sort(v,dim=dim,descending=descending,stable=True)
    torch.testing.assert_close(b.cpu(),a,rtol=0,atol=0);torch.testing.assert_close(j.cpu(),i,rtol=0,atol=0)
    g=torch.randn_like(a);a.backward(g);b.backward(g.to('vulkan'));torch.testing.assert_close(v.grad.cpu(),x.grad)

@pytest.mark.parametrize('descending',[False,True])
def test_sort_ties_nan(descending):
    x=torch.tensor([float('nan'),3.,-0.,0.,3.,-float('inf'),float('inf'),float('nan')])
    a,i=x.sort(stable=True,descending=descending);b,j=x.to('vulkan').sort(stable=True,descending=descending)
    torch.testing.assert_close(b.cpu(),a,equal_nan=True);torch.testing.assert_close(j.cpu(),i)

@pytest.mark.parametrize('dim',[0,1,-1])
@pytest.mark.parametrize('largest',[False,True])
def test_topk_backward(dim,largest):
    x=torch.randn(4,9,requires_grad=True);v=x.detach().to('vulkan').requires_grad_()
    a,i=x.topk(3,dim,largest);b,j=v.topk(3,dim,largest)
    g=torch.randn_like(a);a.backward(g);b.backward(g.to('vulkan'))
    torch.testing.assert_close(b.cpu(),a);torch.testing.assert_close(j.cpu(),i);torch.testing.assert_close(v.grad.cpu(),x.grad)

@pytest.mark.parametrize('dim',[0,1,2,-1])
def test_cumsum_backward(dim):
    x=torch.randn(3,5,269,requires_grad=True);v=x.detach().to('vulkan').requires_grad_()
    a=x.cumsum(dim);b=v.cumsum(dim);g=torch.randn_like(a);a.backward(g);b.backward(g.to('vulkan'))
    torch.testing.assert_close(b.cpu(),a,rtol=2e-4,atol=1e-5);torch.testing.assert_close(v.grad.cpu(),x.grad,rtol=2e-4,atol=2e-5)

@pytest.mark.parametrize('dims',[None,(0,),(1,),(0,2),(-1,)])
@pytest.mark.parametrize('correction',[0,1,2.5])
def test_variance_backward(dims,correction):
    x=torch.randn(3,5,7,requires_grad=True);v=x.detach().to('vulkan').requires_grad_()
    a=x.var(dim=dims,correction=correction);b=v.var(dim=dims,correction=correction)
    g=torch.randn_like(a);a.backward(g);b.backward(g.to('vulkan'))
    torch.testing.assert_close(b.cpu(),a,rtol=1e-5,atol=1e-6);torch.testing.assert_close(v.grad.cpu(),x.grad,rtol=2e-5,atol=2e-6)



def test_deterministic_embedding_backward():
    previous=torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(True)
        w=torch.randn(7,9,device='vulkan').requires_grad_();i=torch.tensor([1,2,2,1,3,2,1],device='vulkan');g=torch.randn(7,9,device='vulkan')
        outputs=[]
        for _ in range(3):
            w.grad=None;F.embedding(i,w).backward(g);outputs.append(w.grad.cpu())
        for x in outputs[1:]:torch.testing.assert_close(x,outputs[0],rtol=0,atol=0)
    finally:torch.use_deterministic_algorithms(previous)
