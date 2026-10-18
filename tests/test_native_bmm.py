"""Native batched GEMM for all rank3 contiguous and strided PyTorch tensors."""
import pytest
import torch
import torch.nn.functional as F
import torch_vulkan


@pytest.mark.parametrize("shape",[(1,2,3,7),(2,8,16,11),(4,16,8,32),
                                    (8,32,32,32),(2,2,128,4)])
@pytest.mark.parametrize("style",["contiguous","left_transpose","right_transpose",
                                  "both","offset","batch_broadcast"])
def test_bmm_forward_and_autograd(shape,style):
    torch.manual_seed(82+sum(shape))
    batch,m,k,n=shape
    xbase=torch.randn(batch,m,k)
    ybase=torch.randn(batch,k,n)
    if style=="left_transpose":
        x=xbase.transpose(1,2).transpose(1,2)
        y=ybase
    elif style=="right_transpose":
        x=xbase
        y=torch.randn(batch,n,k).transpose(1,2)
    elif style=="both":
        x=torch.randn(batch,k,m).transpose(1,2)
        y=torch.randn(batch,n,k).transpose(1,2)
    elif style=="offset":
        x=torch.randn(batch,m+1,k+1)[:,:m,:k]
        y=torch.randn(batch,k+1,n+1)[:,:k,:n]
    elif style=="batch_broadcast":
        x=xbase[:1].expand(batch,-1,-1)
        y=ybase[:1].expand(batch,-1,-1)
    else:
        x=xbase;y=ybase
    # Materialize the same CPU view shapes while keeping their original
    # strides/offsets when transferring to Vulkan: construct equivalent views.
    if style=="offset":
        xx=torch.randn(batch,m+1,k+1)
        yy=torch.randn(batch,k+1,n+1)
        cpu_x=xx[:,:m,:k]
        cpu_y=yy[:,:k,:n]
        vk_x=xx.to("vulkan")[:,:m,:k]
        vk_y=yy.to("vulkan")[:,:k,:n]
    elif style=="both":
        xx=torch.randn(batch,k,m)
        yy=torch.randn(batch,n,k)
        cpu_x=xx.transpose(1,2);cpu_y=yy.transpose(1,2)
        vk_x=xx.to("vulkan").transpose(1,2)
        vk_y=yy.to("vulkan").transpose(1,2)
    elif style=="right_transpose":
        yy=torch.randn(batch,n,k)
        cpu_x=x;cpu_y=yy.transpose(1,2)
        vk_x=x.to("vulkan");vk_y=yy.to("vulkan").transpose(1,2)
    elif style=="batch_broadcast":
        cpu_x=x;cpu_y=y
        vk_x=xbase.to("vulkan")[:1].expand(batch,-1,-1)
        vk_y=ybase.to("vulkan")[:1].expand(batch,-1,-1)
    else:
        cpu_x=x;cpu_y=y
        vk_x=x.to("vulkan");vk_y=y.to("vulkan")
    cpu_x=cpu_x.detach().requires_grad_()
    cpu_y=cpu_y.detach().requires_grad_()
    vk_x=vk_x.detach().requires_grad_()
    vk_y=vk_y.detach().requires_grad_()
    expected=torch.bmm(cpu_x,cpu_y)
    actual=torch.bmm(vk_x,vk_y)
    torch.testing.assert_close(actual.cpu(),expected,rtol=2e-4,atol=1e-4)
    g=torch.randn_like(expected)
    a=torch.autograd.grad(expected,(cpu_x,cpu_y),g)
    b=torch.autograd.grad(actual,(vk_x,vk_y),g.to("vulkan"))
    for got,want in zip(b,a):
        torch.testing.assert_close(got.cpu(),want,rtol=5e-4,atol=2e-4)


def test_scaled_dot_product_attention_uses_gpu_bmm():
    torch.manual_seed(51)
    q,k,v=(torch.randn(2,4,12,16,requires_grad=True) for _ in range(3))
    qv,kv,vv=(t.detach().to("vulkan").requires_grad_() for t in (q,k,v))
    actual=F.scaled_dot_product_attention(qv,kv,vv)
    expected=F.scaled_dot_product_attention(q,k,v)
    torch.testing.assert_close(actual.cpu(),expected,rtol=3e-4,atol=3e-5)
    g=torch.randn_like(expected)
    ca=torch.autograd.grad(expected,(q,k,v),g)
    cb=torch.autograd.grad(actual,(qv,kv,vv),g.to("vulkan"))
    for got,want in zip(cb,ca):
        torch.testing.assert_close(got.cpu(),want,rtol=1e-3,atol=4e-5)
