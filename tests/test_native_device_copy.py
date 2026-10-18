import os
import pytest
import torch
import torch_vulkan


@pytest.mark.parametrize('dtype', [torch.float32, torch.int64])
@pytest.mark.parametrize('layout', ['contiguous', 'transpose', 'offset', 'stepped', 'broadcast'])
def test_native_device_copy(dtype, layout):
    a = torch.arange(63, dtype=dtype).reshape(7, 9)
    av = a.to('vulkan')
    if layout == 'transpose': a, av = a.t(), av.t()
    if layout == 'offset': a, av = a[1:, 1:], av[1:, 1:]
    if layout == 'stepped': a, av = a[:, ::2], av[:, ::2]
    if layout == 'broadcast': a, av = a[:1], av[:1]
    shape = (5, 9) if layout == 'broadcast' else a.shape
    out = torch.empty(shape, dtype=dtype, device='vulkan')
    version = out._version
    out.copy_(av)
    assert out._version == version + 1
    torch.testing.assert_close(out.cpu(), a.expand(shape), rtol=0, atol=0)


def test_native_device_copy_deferred_no_wait_and_lifetime():
    if os.getenv('TORCH_VULKAN_GPU_RESIDENT') != '1' or os.getenv('TORCH_VULKAN_DEFER_SUBMISSIONS') != '1':
        pytest.skip('requires resident/deferred execution')
    x = torch.arange(123, dtype=torch.float32).to('vulkan')
    out = torch.empty_like(x)
    torch_vulkan.synchronize()
    before = torch_vulkan.queue_stats()['submissions']
    out.copy_(x + 3)
    assert torch_vulkan.queue_stats()['submissions'] == before
    torch.testing.assert_close(out.cpu(), torch.arange(123, dtype=torch.float32) + 3)


def test_device_copy_fallback_preserves_partial_storage_and_overlap_errors():
    cpu = torch.full((7, 9), 17.)
    gpu = cpu.to('vulkan')
    src = torch.arange(20.).reshape(5, 4)
    cpu[1:6, 1:9:2].copy_(src)
    gpu[1:6, 1:9:2].copy_(src.to('vulkan'))
    torch.testing.assert_close(gpu.cpu(), cpu)
    with pytest.raises(RuntimeError, match='single memory location'):
        gpu.flatten()[1:].copy_(gpu.flatten()[:-1])


def test_device_copy_dtype_cast_and_autograd():
    x = torch.arange(12., requires_grad=True)
    xv = x.detach().to('vulkan').requires_grad_()
    y = xv.clone()
    y.sum().backward()
    torch.testing.assert_close(xv.grad.cpu(), torch.ones_like(x))
    out = torch.empty(12, dtype=torch.int64, device='vulkan')
    out.copy_(xv.detach())
    torch.testing.assert_close(out.cpu(), x.detach().long())


def test_copy_exact_expanded_alias_is_noop():
    x = torch.arange(7.).to('vulkan').expand(3, 7)
    x.copy_(x)
    torch.testing.assert_close(x.cpu(), torch.arange(7.).expand(3, 7))
    with pytest.raises(RuntimeError, match='single memory location'):
        x.copy_(torch.ones(3, 7, device='vulkan'))
