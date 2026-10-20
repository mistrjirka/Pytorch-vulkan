"""Check general Vulkan mm.out fast path versus torch CPU semantics."""
import pytest
import torch
import torch_vulkan


@pytest.mark.parametrize("shape", [(13, 17, 9), (64, 32, 20), (256, 256, 32),
                                  (1, 2, 3), (0, 16, 9), (7, 0, 11)])
def test_mm_out_cpu_parity(shape):
    m,k,n=shape
    gen=torch.Generator().manual_seed(1245)
    x=torch.randn(m,k,generator=gen)
    y=torch.randn(k,n,generator=gen)
    a=x.to('vulkan')
    b=y.to('vulkan')
    out=torch.empty((m,n),device='vulkan')
    ref=torch.empty((m,n))
    assert torch.mm(x,y,out=ref) is ref
    assert torch.mm(a,b,out=out) is out
    torch.testing.assert_close(out.cpu(),ref,atol=4e-4,rtol=4e-4)
    assert torch_vulkan.execution_stats()['cpu_fallback_calls']==0


@pytest.mark.parametrize("left_transpose,right_transpose",[(False,True),(True,False),(True,True)])
def test_mm_out_strided_inputs(left_transpose,right_transpose):
    m,k,n=12,23,15
    x=torch.randn(k,m).t() if left_transpose else torch.randn(m,k)
    y=torch.randn(n,k).t() if right_transpose else torch.randn(k,n)
    a=x.to('vulkan')
    b=y.to('vulkan')
    cpu=torch.empty(m,n)
    gpu=torch.empty(m,n,device='vulkan')
    torch.mm(x,y,out=cpu)
    torch.mm(a,b,out=gpu)
    torch.testing.assert_close(gpu.cpu(),cpu,atol=4e-4,rtol=4e-4)


def test_mm_out_strided_destination_and_resize():
    a=torch.randn(13,17)
    b=torch.randn(17,19)
    va,vb=a.to('vulkan'),b.to('vulkan')
    cpu=torch.empty(19,13).t()
    gpu=torch.empty(19,13,device='vulkan').t()
    torch.mm(a,b,out=cpu)
    torch.mm(va,vb,out=gpu)
    torch.testing.assert_close(gpu.cpu(),cpu,atol=4e-4,rtol=4e-4)
    resized=torch.empty(0,device='vulkan')
    torch.mm(va,vb,out=resized)
    torch.testing.assert_close(resized.cpu(),cpu,atol=4e-4,rtol=4e-4)


def test_mm_out_alias_inputs():
    a=torch.randn(7,7)
    b=torch.randn(7,7)
    expected=a@b
    va,vb=a.to('vulkan'),b.to('vulkan')
    torch.mm(va,vb,out=va)
    torch.testing.assert_close(va.cpu(),expected,atol=4e-4,rtol=4e-4)
