"""Regressions for the concrete CUDA universality audit correctness failures."""
import os
import pytest
import torch
import torch.nn.functional as F
import torch_vulkan


@pytest.mark.parametrize('dtype', [torch.float64, torch.int32, torch.int64])
def test_mm_rejects_mixed_dtype(dtype):
    x = torch.arange(12, dtype=torch.float32).reshape(3, 4).to('vulkan')
    y = torch.arange(8, dtype=dtype).reshape(4, 2).to('vulkan')
    for lhs, rhs in ((x, y), (y.t(), x.t())):
        with pytest.raises(RuntimeError, match='same dtype'):
            torch.mm(lhs, rhs)


@pytest.mark.parametrize('cpu_left', [False, True])
def test_mm_rejects_mixed_device(cpu_left):
    a, b = torch.randn(3, 4), torch.randn(4, 2)
    a, b = (a, b.to('vulkan')) if cpu_left else (a.to('vulkan'), b)
    with pytest.raises(RuntimeError, match='same.*device'):
        torch.mm(a, b)


@pytest.mark.parametrize('shape', [(0, 3, 7), (3, 7, 0), (3, 0, 7), (0, 0, 0)])
def test_mm_empty_dimensions(shape):
    m, k, n = shape
    a, b = torch.empty(m, k), torch.empty(k, n)
    torch.testing.assert_close((a.to('vulkan') @ b.to('vulkan')).cpu(), a @ b)


@pytest.mark.parametrize('dtype', [torch.int32, torch.int64])
@pytest.mark.parametrize('layout', ['contiguous', 'transpose', 'offset', 'stepped', 'expanded', 'scalar', 'empty'])
def test_embedding_integer_views(dtype, layout):
    weight = torch.arange(60.).reshape(5, 12)[:, ::2]
    wv = torch.arange(60.).reshape(5, 12).to('vulkan')[:, ::2]
    index = torch.tensor([[0, 4, 2], [1, 2, 0]], dtype=dtype)
    iv = index.to('vulkan')
    if layout == 'transpose': index, iv = index.t(), iv.t()
    if layout == 'offset': index, iv = index[1:], iv[1:]
    if layout == 'stepped': index, iv = index[:, ::2], iv[:, ::2]
    if layout == 'expanded': index, iv = index[:1].expand(4, 3), iv[:1].expand(4, 3)
    if layout == 'scalar': index, iv = index[1, 2], iv[1, 2]
    if layout == 'empty': index, iv = index[:0], iv[:0]
    torch.testing.assert_close(F.embedding(iv, wv).cpu(), F.embedding(index, weight), rtol=0, atol=0)


@pytest.mark.parametrize('dtype', [torch.int32, torch.int64])
def test_embedding_large_exact_index(dtype):
    weight = torch.zeros(2**24 + 2, 1)
    weight[2**24], weight[2**24 + 1] = 456, 123
    index = torch.tensor([2**24 + 1, 2**24, 0], dtype=dtype)
    torch.testing.assert_close(F.embedding(index.to('vulkan'), weight.to('vulkan')).cpu(),
                               F.embedding(index, weight), rtol=0, atol=0)


@pytest.mark.parametrize('dtype', [torch.int32, torch.int64])
@pytest.mark.parametrize('index', [-1, 4, 2**24 + 1])
@pytest.mark.parametrize('width', [0, 3])
def test_embedding_invalid_indices(dtype, index, width):
    with pytest.raises(IndexError, match='index out of range'):
        F.embedding(torch.tensor([index], dtype=dtype, device='vulkan'),
                    torch.empty(4, width, device='vulkan'))


@pytest.mark.parametrize('index', [2**32, 2**32 + 1, -(2**32), 2**63 - 1])
def test_embedding_rejects_int64_high_words(index):
    with pytest.raises(IndexError, match='index out of range'):
        F.embedding(torch.tensor([index], device='vulkan'), torch.empty(4, 3, device='vulkan'))


@pytest.mark.parametrize("count", [1003, 33001])
def test_embedding_validation_tail_and_recovery(count):
    w = torch.randn(7, 3).to('vulkan')
    i = torch.zeros(count, dtype=torch.int64)
    i[-1] = 7
    with pytest.raises(IndexError):
        F.embedding(i.to('vulkan'), w)
    i[-1] = 6
    torch.testing.assert_close(F.embedding(i.to('vulkan'), w).cpu(), F.embedding(i, w.cpu()))


def test_embedding_empty_table_and_zero_width():
    for shape in [(0, 3), (4, 0)]:
        w = torch.empty(shape, device='vulkan')
        y = F.embedding(torch.empty(2, 0, dtype=torch.long, device='vulkan'), w)
        assert y.shape == (2, 0, shape[1])
    with pytest.raises(IndexError):
        F.embedding(torch.tensor([0], device='vulkan'), torch.empty(0, 3, device='vulkan'))
    assert F.embedding(torch.tensor([0], device='vulkan'), torch.empty(4, 0, device='vulkan')).shape == (1, 0)


@pytest.mark.parametrize('padding_idx', [None, 0, -1])
@pytest.mark.parametrize('frequency', [False, True])
def test_embedding_dense_backward_flags(padding_idx, frequency):
    w = torch.randn(7, 4, requires_grad=True)
    v = w.detach().to('vulkan').requires_grad_()
    i = torch.tensor([0, 1, 1, 3, 6])
    x = F.embedding(i, w, padding_idx=padding_idx, scale_grad_by_freq=frequency)
    y = F.embedding(i.to('vulkan'), v, padding_idx=padding_idx, scale_grad_by_freq=frequency)
    grad = torch.randn_like(x)
    x.backward(grad); y.backward(grad.to('vulkan'))
    torch.testing.assert_close(y.cpu(), x)
    torch.testing.assert_close(v.grad.cpu(), w.grad)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64, torch.bool])
def test_embedding_rejects_noninteger_indices(dtype):
    with pytest.raises(RuntimeError, match='int32 or int64'):
        F.embedding(torch.tensor([0, 1], dtype=dtype, device='vulkan'), torch.empty(4, 3, device='vulkan'))


@pytest.mark.parametrize('shape,fmt', [((2, 3, 5, 7), torch.channels_last),
                                       ((2, 3, 4, 5, 7), torch.channels_last_3d)])
def test_factory_memory_format_and_copy(shape, fmt):
    cpu = torch.empty(shape, memory_format=fmt)
    gpu = torch.empty(shape, memory_format=fmt, device='vulkan')
    assert gpu.stride() == cpu.stride()
    assert gpu.is_contiguous(memory_format=fmt)
    value = torch.randn(shape)
    gpu.copy_(value)
    torch.testing.assert_close(gpu.cpu(), value)


def test_channels_last_factory_convolution_backward():
    x = torch.randn(2, 3, 7, 9).contiguous(memory_format=torch.channels_last).requires_grad_()
    v = torch.empty(x.shape, device='vulkan', memory_format=torch.channels_last)
    v.copy_(x.detach()); v.requires_grad_()
    w = torch.randn(4, 3, 3, 3, requires_grad=True)
    wv = w.detach().to('vulkan').requires_grad_()
    a, b = F.conv2d(x, w, padding=1), F.conv2d(v, wv, padding=1)
    grad = torch.randn_like(a)
    a.backward(grad); b.backward(grad.to('vulkan'))
    torch.testing.assert_close(b.cpu(), a, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(v.grad.cpu(), x.grad, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(wv.grad.cpu(), w.grad, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize('factory', [torch.empty, lambda shape, **kw: torch.empty_strided(shape, (3, 1), **kw)])
def test_factory_rejects_invalid_device(factory):
    with pytest.raises(RuntimeError, match='only logical device 0'):
        factory((2, 3), device='vulkan:17')


def test_factory_default_dtype_and_invalid_options():
    old = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        assert torch.empty(3, device='vulkan').dtype == torch.float64
    finally:
        torch.set_default_dtype(old)
    with pytest.raises(RuntimeError):
        torch.empty((-1, 3), device='vulkan')
    with pytest.raises(RuntimeError):
        torch.empty((2**62, 4), device='vulkan')
    with pytest.raises(RuntimeError):
        torch.empty((2, 3), device='vulkan', memory_format=torch.channels_last)
    with pytest.raises(RuntimeError):
        torch.empty((2, 3), device='vulkan', pin_memory=True)


@pytest.mark.parametrize('layout', ['contiguous', 'transpose', 'offset', 'stepped', 'expanded', 'empty'])
def test_zero_preserves_aliases_and_version(layout):
    cpu = torch.arange(63.).reshape(7, 9)
    gpu = cpu.to('vulkan')
    a, b = cpu, gpu
    if layout == 'transpose': a, b = a.t(), b.t()
    if layout == 'offset': a, b = a[1:5], b[1:5]
    if layout == 'stepped': a, b = a[:, ::2], b[:, ::2]
    if layout == 'expanded': a, b = a[:1].expand(3, 9), b[:1].expand(3, 9)
    if layout == 'empty': a, b = a[:0], b[:0]
    version = gpu._version
    a.zero_(); b.zero_()
    assert gpu._version == version + 1
    torch.testing.assert_close(gpu.cpu(), cpu, rtol=0, atol=0)


def test_zero_deferred_does_not_submit():
    if os.getenv('TORCH_VULKAN_GPU_RESIDENT') != '1' or os.getenv('TORCH_VULKAN_DEFER_SUBMISSIONS') != '1':
        pytest.skip('requires resident/deferred execution')
    x = torch.ones(128, device='vulkan')
    torch_vulkan.synchronize()
    before = torch_vulkan.queue_stats()['submissions']
    x.zero_()
    assert torch_vulkan.queue_stats()['submissions'] == before
    assert x.cpu().count_nonzero() == 0


@pytest.mark.parametrize('dtype', [torch.int32, torch.bool])
@pytest.mark.parametrize('layout', ['contiguous', 'transpose', 'stepped', 'offset', 'broadcast'])
def test_exact_integer_and_bool_device_copy(dtype, layout):
    if dtype == torch.int32:
        cpu = torch.tensor([-(2**31), -1, 0, 2**24 + 1, 2**31 - 1, 2139095041] * 5, dtype=dtype).reshape(5, 6)
    else:
        cpu = (torch.arange(30).reshape(5, 6) % 3 == 0)
    gpu = cpu.to('vulkan')
    if layout == 'transpose': cpu, gpu = cpu.t(), gpu.t()
    if layout == 'stepped': cpu, gpu = cpu[:, ::2], gpu[:, ::2]
    if layout == 'offset': cpu, gpu = cpu[1:, 1:], gpu[1:, 1:]
    if layout == 'broadcast': cpu, gpu = cpu[:1].expand(5, 6), gpu[:1].expand(5, 6)
    out = torch.empty(cpu.shape, dtype=dtype, device='vulkan')
    out.copy_(gpu)
    torch.testing.assert_close(out.cpu(), cpu, rtol=0, atol=0)


@pytest.mark.parametrize('source', ['cpu', 'vulkan'])
def test_copy_rejects_invalid_device(source):
    x = torch.ones(3, device=source)
    with pytest.raises(RuntimeError, match='only logical device 0'):
        x.to('vulkan:17')
