"""NLL reductions, views, ignored labels, and queued label writes."""
import pytest
import torch
import torch_vulkan


@pytest.mark.parametrize('reduction', [0, 1, 2])
@pytest.mark.parametrize('shape', [(7,), (4, 7)])
@pytest.mark.parametrize('weighted', [False, True])
@pytest.mark.parametrize('strided', [False, True])
def test_nll_forward_backward(reduction, shape, weighted, strided):
    torch.manual_seed(19)
    source = torch.randn(*(shape[:-1] + (shape[-1] * 2,)))
    xv = source.to('vulkan')
    x = source[..., 1::2] if strided else source[..., :7].contiguous()
    xv = xv[..., 1::2] if strided else x.to('vulkan')
    target = torch.tensor([1, -100, 3, 6]) if len(shape) == 2 else torch.tensor(3)
    if len(shape) == 2 and strided:
        t = torch.tensor([8, 1, 8, -100, 8, 3, 8, 6])
        target, tv = t[1::2], t.to('vulkan')[1::2]
    else:
        tv = target.to('vulkan')
    w = torch.rand(7) if weighted else None
    wv = w.to('vulkan') if weighted else None
    a, total = torch.ops.aten.nll_loss_forward(x, target, w, reduction, -100)
    av, totalv = torch.ops.aten.nll_loss_forward(xv, tv, wv, reduction, -100)
    torch.testing.assert_close(av.cpu(), a, rtol=3e-6, atol=2e-7)
    torch.testing.assert_close(totalv.cpu(), total)
    grad = torch.randn_like(a)
    expected = torch.ops.aten.nll_loss_backward(grad, x, target, w, reduction, -100, total)
    actual = torch.ops.aten.nll_loss_backward(grad.to('vulkan'), xv, tv, wv, reduction, -100, totalv)
    torch.testing.assert_close(actual.cpu(), expected, rtol=3e-6, atol=2e-7)


@pytest.mark.parametrize('ignore', [-100, -(2**40), 2**40])
@pytest.mark.parametrize('reduction', [0, 1, 2])
def test_nll_all_ignored(ignore, reduction):
    x, t = torch.randn(3, 7), torch.full((3,), ignore, dtype=torch.long)
    cpu = torch.ops.aten.nll_loss_forward(x, t, None, reduction, ignore)
    gpu = torch.ops.aten.nll_loss_forward(x.to('vulkan'), t.to('vulkan'), None, reduction, ignore)
    for a, b in zip(cpu, gpu):
        torch.testing.assert_close(b.cpu(), a, equal_nan=True)


@pytest.mark.parametrize('reduction', [0, 1, 2])
def test_nll_vector_target_and_empty(reduction):
    for x, t in [(torch.randn(7), torch.tensor([3])), (torch.empty(0, 7), torch.empty(0, dtype=torch.long))]:
        cpu = torch.ops.aten.nll_loss_forward(x, t, None, reduction, -100)
        gpu = torch.ops.aten.nll_loss_forward(x.to('vulkan'), t.to('vulkan'), None, reduction, -100)
        for a, b in zip(cpu, gpu):
            torch.testing.assert_close(b.cpu(), a, equal_nan=True)


@pytest.mark.parametrize('bad', [-1, 7, 2**40])
def test_nll_invalid_label(bad):
    with pytest.raises((IndexError, RuntimeError), match='out of bounds'):
        torch.ops.aten.nll_loss_forward(torch.randn(2, 7).to('vulkan'), torch.tensor([0, bad]).to('vulkan'), None, 1, -100)


def test_nll_gpu_written_labels():
    x = torch.randn(4, 7)
    tv = torch.tensor([0, 1, 2, 3]).to('vulkan')
    tv.add_(1)
    actual = torch.nn.functional.nll_loss(x.to('vulkan'), tv)
    torch.testing.assert_close(actual.cpu(), torch.nn.functional.nll_loss(x, torch.tensor([1, 2, 3, 4])))
    tv.add_(7)
    with pytest.raises((IndexError, RuntimeError), match='out of bounds'):
        torch.nn.functional.nll_loss(x.to('vulkan'), tv)


@pytest.mark.parametrize('reduction,grad_shape', [(0, ()), (0, (1, 1)), (1, (1, 1)), (2, (1, 1))])
def test_nll_backward_rejects_gradient_rank(reduction, grad_shape):
    with pytest.raises(RuntimeError, match='gradient shape'):
        torch.ops.aten.nll_loss_backward(torch.ones(grad_shape).to('vulkan'), torch.randn(1, 3).to('vulkan'), torch.tensor([1]).to('vulkan'), None, reduction, -100, torch.tensor(1.).to('vulkan'))


@pytest.mark.parametrize('weight_value', [0., -2., float('inf'), float('nan')])
@pytest.mark.parametrize('reduction', [0, 1, 2])
def test_nll_weight_edge_cases(weight_value, reduction):
    x, t, w = torch.randn(3, 7), torch.tensor([0, 1, 2]), torch.full((7,), weight_value)
    cpu = torch.ops.aten.nll_loss_forward(x, t, w, reduction, -100)
    xv, tv, wv = x.to('vulkan'), t.to('vulkan'), w.to('vulkan')
    gpu = torch.ops.aten.nll_loss_forward(xv, tv, wv, reduction, -100)
    for a, b in zip(cpu, gpu):
        torch.testing.assert_close(b.cpu(), a, equal_nan=True)
    grad = torch.ones_like(cpu[0])
    a = torch.ops.aten.nll_loss_backward(grad, x, t, w, reduction, -100, cpu[1])
    b = torch.ops.aten.nll_loss_backward(grad.to('vulkan'), xv, tv, wv, reduction, -100, gpu[1])
    torch.testing.assert_close(b.cpu(), a, equal_nan=True)
