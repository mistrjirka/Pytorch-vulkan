"""Strided Vulkan bmm.out writes directly when safe, otherwise preserves ATen."""
import pytest
import torch
import torch_vulkan


@pytest.mark.parametrize("shape", [(1, 8, 13, 7), (3, 17, 33, 11), (2, 1, 19, 7), (0, 8, 13, 7), (1, 5, 0, 3)])
def test_direct_bmm_out_matches_cpu(shape):
    b,m,k,n = shape
    gen = torch.Generator().manual_seed(144)
    a=torch.randn(b,m,k,generator=gen)
    c=torch.randn(b,k,n,generator=gen)
    va,vc=a.to("vulkan"),c.to("vulkan")
    expected=torch.empty((b,m,n),dtype=torch.float32)
    actual=torch.empty((b,m,n),device="vulkan",dtype=torch.float32)
    assert torch.bmm(a,c,out=expected) is expected
    assert torch.bmm(va,vc,out=actual) is actual
    torch.testing.assert_close(actual.cpu(),expected,atol=2e-4,rtol=2e-4)
    assert torch_vulkan.execution_stats()["cpu_fallback_calls"]==0


@pytest.mark.parametrize("transpose", [(True,False),(False,True),(True,True)])
def test_bmm_out_strided_inputs(transpose):
    aa=torch.randn(3,20,11)
    bb=torch.randn(3,11,20)
    a=aa.transpose(1,2) if transpose[0] else aa
    b=bb.transpose(1,2) if transpose[1] else bb
    k=a.size(2)
    # Build matching shapes with actual strides for both operands.
    if transpose[1]:
        b=torch.randn(3,7,k).transpose(1,2)
    else:
        b=torch.randn(3,k,7)
    va,vb=a.to('vulkan'),b.to('vulkan')
    cpu=torch.empty(3,a.size(1),7)
    gpu=torch.empty_like(cpu,device='vulkan')
    torch.bmm(a,b,out=cpu)
    torch.bmm(va,vb,out=gpu)
    torch.testing.assert_close(gpu.cpu(),cpu,atol=3e-4,rtol=3e-4)


def test_bmm_out_noncontiguous_destination_and_resize():
    a=torch.randn(2,13,19)
    b=torch.randn(2,19,17)
    va,vb=a.to('vulkan'),b.to('vulkan')
    cpu=torch.empty(2,17,13).transpose(1,2)
    gpu=torch.empty(2,17,13,device='vulkan').transpose(1,2)
    torch.bmm(a,b,out=cpu)
    torch.bmm(va,vb,out=gpu)
    torch.testing.assert_close(gpu.cpu(),cpu,atol=2e-4,rtol=2e-4)
    resized=torch.empty(0,device='vulkan')
    torch.bmm(va,vb,out=resized)
    torch.testing.assert_close(resized.cpu(),cpu,atol=2e-4,rtol=2e-4)


def test_bmm_out_aliases_input_snapshot():
    a=torch.randn(1,5,5)
    b=torch.randn(1,5,5)
    expected=a.clone()
    torch.bmm(a.clone(),b,out=expected)
    va=a.to('vulkan')
    vb=b.to('vulkan')
    torch.bmm(va,vb,out=va)
    torch.testing.assert_close(va.cpu(),expected,atol=2e-4,rtol=2e-4)
