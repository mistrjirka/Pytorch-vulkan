import copy
import pytest
import torch
import torch.nn.functional as F
import torch_vulkan

@pytest.mark.parametrize('rank,transposed', [(3, False), (1, True), (2, True), (3, True)])
@pytest.mark.parametrize('groups', [1, 2, 4])
@pytest.mark.parametrize('layout', ['contiguous', 'offset', 'channels_last'])
def test_convolution_nd_forward_and_gradients(rank, transposed, groups, layout):
    torch.manual_seed(91)
    spatial = (4, 5, 6)[-rank:]
    co = 8
    x = torch.randn((2, 4, *spatial))
    kernel = (2, 3, 2)[-rank:]
    w = torch.randn((4, co // groups, *kernel) if transposed else (co, 4 // groups, *kernel))
    b = torch.randn(co)
    if layout == 'offset':
        def view(t, dev):
            base = torch.empty((*t.shape[:-1], t.shape[-1] * 2 + 1), device=dev)
            out = base[..., 1::2]
            out.copy_(t.to(dev))
            return out.detach().requires_grad_()
        refs = [t.clone().requires_grad_() for t in (x,w,b)]
        vals = [view(x, 'vulkan'), view(w, 'vulkan'), b.to('vulkan').requires_grad_()]
    else:
        if layout == 'channels_last' and rank > 1:
            x = x.contiguous(memory_format=torch.channels_last_3d if rank == 3 else torch.channels_last)
        refs = [t.clone().requires_grad_() for t in (x,w,b)]
        vals = [t.to('vulkan').requires_grad_() for t in (x,w,b)]
    fn = getattr(F, f'conv_transpose{rank}d' if transposed else f'conv{rank}d')
    kw = dict(stride=(1,2,2)[-rank:], padding=(1,1,0)[-rank:], dilation=(1,1,2)[-rank:], groups=groups)
    if transposed: kw['output_padding'] = (0,1,1)[-rank:]
    expected, actual = fn(*refs, **kw), fn(*vals, **kw)
    grad = torch.randn_like(expected)
    expected.backward(grad); actual.backward(grad.to('vulkan'))
    torch.testing.assert_close(actual.cpu(), expected, rtol=3e-4, atol=1e-4)
    for a,b in zip(vals,refs):torch.testing.assert_close(a.grad.cpu(), b.grad, rtol=4e-4, atol=2e-4)

@pytest.mark.parametrize('transposed', [False, True])
def test_conv3d_empty_batch(transposed):
    x = torch.empty(0, 2, 3, 4, 5, requires_grad=True)
    w = torch.randn((2,3,2,2,2) if transposed else (3,2,2,2,2), requires_grad=True)
    fn = F.conv_transpose3d if transposed else F.conv3d
    a,b = x.to('vulkan').detach().requires_grad_(),w.to('vulkan').detach().requires_grad_()
    expected, actual = fn(x,w),fn(a,b)
    expected.sum().backward();actual.backward(torch.empty_like(actual))
    torch.testing.assert_close(actual.cpu(),expected)
    torch.testing.assert_close(b.grad.cpu(),w.grad)

@pytest.mark.parametrize('layers,bidirectional,projection,bias', [(1,False,0,True),(1,False,0,False),(2,True,0,True),(1,True,3,True)])
def test_lstm_forward_backward(layers,bidirectional,projection,bias):
    torch.manual_seed(71)
    ref = torch.nn.LSTM(5,7,num_layers=layers,bidirectional=bidirectional,proj_size=projection,bias=bias,batch_first=True)
    vk = copy.deepcopy(ref).to('vulkan')
    n = layers * (2 if bidirectional else 1)
    inputs = [torch.randn(2,4,5,requires_grad=True),torch.randn(n,2,projection or 7,requires_grad=True),torch.randn(n,2,7,requires_grad=True)]
    vals = [x.detach().to('vulkan').requires_grad_() for x in inputs]
    e, (eh,ec) = ref(inputs[0],tuple(inputs[1:]))
    a, (ah,ac) = vk(vals[0],tuple(vals[1:]))
    grads = [torch.randn_like(x) for x in [e,eh,ec]]
    torch.autograd.backward([e,eh,ec],grads)
    torch.autograd.backward([a,ah,ac],[g.to('vulkan') for g in grads])
    for x,y in zip([a,ah,ac],[e,eh,ec]):torch.testing.assert_close(x.cpu(),y,rtol=3e-4,atol=2e-5)
    for x,y in zip([*vals,*vk.parameters()],[*inputs,*ref.parameters()]):torch.testing.assert_close(x.grad.cpu(),y.grad,rtol=5e-4,atol=3e-5)
