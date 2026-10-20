"""BatchNorm parity across dimensionality, layouts, statistics and autograd."""
import itertools
import pytest
import torch
import torch_vulkan


def layout(x,kind,device):
    if kind=='channels_last': return x.to(device).contiguous(memory_format=torch.channels_last)
    if kind=='transpose': return x.transpose(-1,-2).contiguous().to(device).transpose(-1,-2)
    if kind=='slice':
        storage=torch.zeros((*x.shape[:-1],x.shape[-1]*2+3),dtype=x.dtype)
        storage[...,1:1+2*x.shape[-1]:2]=x
        return storage.to(device)[...,1:1+2*x.shape[-1]:2]
    if kind=='broadcast': return x[:1].to(device).expand(x.shape)
    return x.to(device)


@pytest.mark.parametrize('shape,kind',[
    ((7,5),'contiguous'),((2,3,11),'slice'),((2,5,7,9),'channels_last'),
    ((2,5,7,9),'transpose'),((2,3,4,5,7),'slice'),((3,2,4,7),'broadcast'),
    ((2,3,2,3,2,2),'transpose'),
])
@pytest.mark.parametrize('training,affine,stats',[(True,True,True),(True,False,False),(False,True,True),(False,False,True)])
def test_native_batch_norm(shape,kind,training,affine,stats):
    torch.manual_seed(90)
    x=torch.randn(shape);g=torch.randn(shape);c=shape[1]
    w=torch.randn(c) if affine else None;b=torch.randn(c) if affine else None
    rm=torch.randn(c) if stats else None;rv=torch.rand(c)+.3 if stats else None
    def execute(device):
        xx=layout(x,kind,device);gg=layout(g,kind,device)
        # Strided offset affine and running statistics exercise both reads and mutation.
        # The CPU optimized backward assumes contiguous channel vectors.
        # Use the same logical values contiguously for the reference.
        def vec(v):
            if v is None:return None
            return v.clone() if device=='cpu' else layout(v,'slice',device)
        ww,bb,mm,vv=map(vec,(w,b,rm,rv))
        out,mean,inv=torch.ops.aten.native_batch_norm(xx,ww,bb,mm,vv,training,.23,1e-4)
        grad=torch.ops.aten.native_batch_norm_backward(gg,xx,ww,mm,vv,mean,inv,training,1e-4,[True]*3)
        return (out,mean,inv,*grad,*([mm,vv] if stats else []))
    ref=execute('cpu');got=execute('vulkan')
    for a,b in zip(got,ref):torch.testing.assert_close(a.cpu(),b,rtol=3e-4,atol=3e-5)


@pytest.mark.parametrize('mask',list(itertools.product([False,True],repeat=3)))
@pytest.mark.parametrize('training',[False,True])
def test_batch_norm_masks(mask,training):
    torch.manual_seed(8)
    x=torch.randn(2,5,7);g=torch.randn_like(x);w=torch.randn(5);m=torch.randn(5);v=torch.rand(5)+.5
    def run(device):
        xx,gg,ww,mm,vv=[t.to(device) for t in (x,g,w,m,v)]
        _,sm,si=torch.ops.aten.native_batch_norm(xx,ww,None,mm,vv,training,.1,1e-5)
        return torch.ops.aten.native_batch_norm_backward(gg,xx,ww,mm,vv,sm,si,training,1e-5,mask)
    for yes,a,b in zip(mask,run('vulkan'),run('cpu')):
        if yes:torch.testing.assert_close(a.cpu(),b,rtol=3e-4,atol=3e-5)
        else:assert a is None


@pytest.mark.parametrize('offset,noise,eps',[(10000.,.1,1e-5),(3.,0.,0.),(3.,0.,1e-5)])
def test_batch_norm_stable_variance(offset,noise,eps):
    torch.manual_seed(4)
    x=(torch.randn(4,3,31,19)*noise+offset)
    # Double reference exposes cancellation rather than reproducing float CPU reduction error.
    ref=torch.ops.aten.native_batch_norm(x.double(),None,None,None,None,True,.1,eps)
    got=torch.ops.aten.native_batch_norm(x.to('vulkan'),None,None,None,None,True,.1,eps)
    torch.testing.assert_close(got[1].cpu(),ref[1].float(),rtol=0,atol=.001)
    torch.testing.assert_close(got[2].cpu(),ref[2].float(),rtol=3e-5,atol=1e-5)
    torch.testing.assert_close(got[0].cpu(),ref[0].float(),rtol=3e-4,atol=.008)


@pytest.mark.parametrize('ndim',[1,2,3])
@pytest.mark.parametrize('training',[True,False])
def test_batch_norm_module_chain(ndim,training):
    torch.manual_seed(77)
    cls=getattr(torch.nn,f'BatchNorm{ndim}d')
    cpu=torch.nn.Sequential(cls(3),torch.nn.ReLU(),cls(3)).train(training)
    vk=torch.nn.Sequential(cls(3),torch.nn.ReLU(),cls(3)).train(training).to('vulkan')
    vk.load_state_dict(cpu.state_dict())
    x=torch.randn((2,3)+((5,)*ndim));xc=x.clone().requires_grad_();xv=x.to('vulkan').requires_grad_()
    yc=cpu(xc);yv=vk(xv)
    yc.sum().backward();yv.sum().backward() # broadcast, zero-stride output gradient
    torch.testing.assert_close(yv.cpu(),yc,atol=3e-5,rtol=3e-4)
    torch.testing.assert_close(xv.grad.cpu(),xc.grad,atol=4e-5,rtol=4e-4)
    # The mathematically zero scale gradient sums 250 centered FP32 values.
    # Lavapipe and NVIDIA differ in reduction/FMA rounding (observed 5.6e-5).
    for pc,pv in zip(cpu.parameters(),vk.parameters()):torch.testing.assert_close(pv.grad.cpu(),pc.grad,atol=1e-4,rtol=4e-4)
    for bc,bv in zip(cpu.buffers(),vk.buffers()):torch.testing.assert_close(bv.cpu(),bc,atol=3e-5,rtol=3e-4)


def test_batch_norm_single_sample_and_no_cpu_fallback():
    x=torch.ones(1,3).to('vulkan');m=torch.zeros(3).to('vulkan');v=torch.ones(3).to('vulkan')
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
        y,mean,inv=torch.ops.aten.native_batch_norm(x,None,None,m,v,True,.1,1e-5)
        torch.ops.aten.native_batch_norm_backward(x,x,None,m,v,mean,inv,True,1e-5,[True]*3)
        torch_vulkan.synchronize()
    assert not any(e.key=='aten::_to_copy' for e in prof.key_averages())
    torch.testing.assert_close(y.cpu(),torch.zeros(1,3))
    assert v.cpu().isnan().all()


# Precision regressions use the values actually representable in FP32 as their
# input, then promote to FP64. Reproducing CPU FP32 cancellation is not a
# correctness oracle for almost-constant channels or two-value reductions.
def _precision_batch_norm(shape, kind, offset, noise, eps, affine, *,
                          training=True, mask=(True, True, True), grad_offset=0.0):
    generator = torch.Generator().manual_seed(397 + sum(shape))
    x = torch.randn(shape, generator=generator) * noise + offset
    g = torch.randn(shape, generator=generator) * 0.3 + grad_offset
    channels = shape[1]
    weight = torch.linspace(-1.3, 1.7, channels) if affine else None
    bias = torch.linspace(-0.4, 0.2, channels) if affine else None
    running_mean = torch.linspace(-3.0, 7.0, channels)
    running_var = torch.linspace(0.7, 2.1, channels)
    momentum = 0.23

    # FP64 reference uses contiguous channel vectors; the Vulkan side also
    # checks offset/strided vectors, including running-stat mutation.
    xd, gd = x.double(), g.double()
    wd, bd = [t.double() if t is not None else None for t in (weight, bias)]
    md, vd = running_mean.double(), running_var.double()
    expected, mean_d, inv_d = torch.ops.aten.native_batch_norm(
        xd, wd, bd, md, vd, training, momentum, eps)
    expected_grads = torch.ops.aten.native_batch_norm_backward(
        gd, xd, wd, md, vd, mean_d, inv_d, training, eps, mask)

    xv, gv = layout(x, kind, 'vulkan'), layout(g, kind, 'vulkan')
    def vector(v):
        return None if v is None else layout(v, 'slice', 'vulkan')
    wv, bv, mv, vv = map(vector, (weight, bias, running_mean, running_var))
    actual, mean_v, inv_v = torch.ops.aten.native_batch_norm(
        xv, wv, bv, mv, vv, training, momentum, eps)
    actual_grads = torch.ops.aten.native_batch_norm_backward(
        gv, xv, wv, mv, vv, mean_v, inv_v, training, eps, mask)
    torch.testing.assert_close(actual.cpu().double(), expected, rtol=4e-5, atol=2e-5)
    if training:
        torch.testing.assert_close(mean_v.cpu(), mean_d.float(), rtol=0, atol=0.001)
        torch.testing.assert_close(inv_v.cpu().double(), inv_d, rtol=4e-5, atol=2e-5)
    else:
        assert mean_v.numel() == 0 and inv_v.numel() == 0
    for enabled, actual_grad, expected_grad in zip(mask, actual_grads, expected_grads):
        if enabled:
            torch.testing.assert_close(actual_grad.cpu().double(), expected_grad,
                                       rtol=8e-5, atol=5e-5)
        else:
            assert actual_grad is None
    torch.testing.assert_close(mv.cpu(), md.float(), rtol=2e-7, atol=0.001)
    torch.testing.assert_close(vv.cpu().double(), vd, rtol=2e-6, atol=2e-7, equal_nan=True)
    return x, g, actual_grads, mean_v, inv_v


@pytest.mark.parametrize('shape,kind,offset,noise,eps,affine', [
    pytest.param((2, 3), 'contiguous', 10000.0, .1, 1e-5, True, id='pair-batch-high-offset'),
    pytest.param((2, 3), 'slice', 10000.0, .1, 0.0, False, id='pair-batch-high-offset-eps0'),
    pytest.param((2, 3), 'slice', 1.0, 1e-4, 1e-5, False, id='pair-batch-near-equal'),
    pytest.param((2, 3), 'contiguous', 1.0, 1e-4, 0.0, True, id='pair-batch-near-equal-eps0'),
    pytest.param((2, 3), 'slice', 3.0, 0.0, 1e-5, True, id='pair-batch-constant'),
    pytest.param((2, 3), 'contiguous', 3.0, 0.0, 0.0, False, id='pair-batch-constant-eps0'),
    pytest.param((1, 3, 1, 2), 'channels_last', 10000.0, .1, 1e-5, True, id='pair-spatial-high-offset'),
    pytest.param((1, 3, 1, 2), 'slice', 10000.0, .1, 0.0, False, id='pair-spatial-high-offset-eps0'),
    pytest.param((1, 3, 2, 1), 'transpose', 1.0, 1e-4, 1e-5, False, id='pair-spatial-near-equal'),
    pytest.param((1, 3, 1, 2), 'slice', 1.0, 1e-4, 0.0, True, id='pair-spatial-near-equal-eps0'),
    pytest.param((1, 3, 1, 2), 'channels_last', 3.0, 0.0, 1e-5, True, id='pair-spatial-constant'),
    pytest.param((1, 3, 2, 1), 'transpose', 3.0, 0.0, 0.0, False, id='pair-spatial-constant-eps0'),
    *[pytest.param((1, 3, samples), 'slice', 10000.0, .1, eps, True,
                   id=f'general-reduction{samples}-eps{eps}')
      for samples in (3, 8, 129) for eps in (0.0, 1e-5)],
])
def test_batch_norm_centering_and_small_reduction_precision(shape, kind, offset, noise, eps, affine):
    x, _, grads, _, _ = _precision_batch_norm(shape, kind, offset, noise, eps, affine)
    if x.numel() // shape[1] == 2:
        # The two input derivatives must be equal and opposite, including
        # epsilon zero and constant inputs. This catches asymmetric residuals.
        per_channel = grads[0].cpu().movedim(1, 0).reshape(shape[1], 2)
        torch.testing.assert_close(per_channel[:, 0], -per_channel[:, 1], rtol=0, atol=0)


@pytest.mark.parametrize('affine', [False, True])
def test_batch_norm_pair_common_gradient_offset_cancellation(affine):
    x, g, grads, _, inv = _precision_batch_norm(
        (1, 3, 1, 2), 'slice', 10000.0, .1, 1e-5, affine, grad_offset=10000.0)
    # dWeight depends on the difference of the two gradients, not their large
    # common offset. Compare against a direct FP64 pair identity as well.
    pair_x = x.double().reshape(3, 2)
    pair_g = g.double().reshape(3, 2)
    expected_dw = .5 * (pair_x[:, 0] - pair_x[:, 1]) * (pair_g[:, 0] - pair_g[:, 1]) * inv.cpu().double()
    torch.testing.assert_close(grads[1].cpu().double(), expected_dw, rtol=3e-6, atol=1e-7)


@pytest.mark.parametrize('mask', [(False, False, False), (True, False, False),
                                 (False, True, False), (False, False, True)])
def test_batch_norm_pair_precision_output_masks(mask):
    _precision_batch_norm((1, 3, 1, 2), 'slice', 10000.0, .1, 1e-5, True, mask=mask)


@pytest.mark.parametrize('shape,kind,eps,affine', [
    ((2, 3), 'slice', 0.0, True),
    ((2, 3), 'contiguous', 1e-5, False),
    ((1, 3, 1, 2), 'channels_last', 0.0, False),
    ((1, 3, 2, 1), 'transpose', 1e-5, True),
])
def test_batch_norm_two_values_evaluation_uses_running_statistics(shape, kind, eps, affine):
    # The training pair identity would force opposite dInputs; inference must
    # instead independently scale each gradient using the supplied variance.
    _precision_batch_norm(shape, kind, 10000.0, .1, eps, affine, training=False)


@pytest.mark.parametrize('eps', [0.0, 1e-5])
def test_batch_norm_single_value_precision_convention(eps):
    x, g, grads, mean, inv = _precision_batch_norm(
        (1, 3, 1, 1), 'slice', 3.0, 0.0, eps, True)
    torch.testing.assert_close(grads[0].cpu(), torch.zeros_like(x), rtol=0, atol=0)
    torch.testing.assert_close(grads[1].cpu(), torch.zeros(3), rtol=0, atol=0)
    torch.testing.assert_close(grads[2].cpu(), g.reshape(3), rtol=0, atol=0)
    if eps == 0:
        torch.testing.assert_close(inv.cpu(), torch.zeros(3), rtol=0, atol=0)


def test_batch_norm_eval_dweight_with_overflowing_dbias():
    # Finite individual gradients can overflow their sum. Evaluation must not
    # introduce 0*infinity into an otherwise exactly zero scale gradient.
    x = torch.ones(2, 3).to('vulkan')
    g = torch.full((2, 3), 2e38).to('vulkan')
    mean = torch.ones(3).to('vulkan')
    var = torch.ones(3).to('vulkan')
    empty = torch.empty(0).to('vulkan')
    dx, dw, db = torch.ops.aten.native_batch_norm_backward(
        g, x, None, mean, var, empty, empty, False, 0., [True]*3)
    torch.testing.assert_close(dx.cpu(), torch.full((2, 3), 2e38))
    torch.testing.assert_close(dw.cpu(), torch.zeros(3), rtol=0, atol=0)
    assert db.cpu().isposinf().all()


def test_batch_norm_training_dweight_with_overflowing_dbias():
    x = torch.tensor([[-1.], [-1.], [1.], [1.]]).to('vulkan')
    g = torch.full((4, 1), 1e38).to('vulkan')
    _, mean, inv = torch.ops.aten.native_batch_norm(
        x, None, None, None, None, True, .1, 1e-5)
    dx, dw, db = torch.ops.aten.native_batch_norm_backward(
        g, x, None, None, None, mean, inv, True, 1e-5, [False, True, False])
    assert dx is None and db is None
    torch.testing.assert_close(dw.cpu(), torch.zeros(1), rtol=0, atol=0)


def test_batch_norm_pair_finite_opposite_large_gradients():
    # Half the difference is finite even when forming the difference first
    # would overflow. Small gamma leaves the exact dInput representable.
    x = torch.zeros(2, 1)
    g = torch.tensor([[3e38], [-3e38]])
    w = torch.tensor([1e-20])
    results = []
    for device, dtype in [('cpu', torch.float64), ('vulkan', torch.float32)]:
        xx, gg, ww = [t.to(device=device, dtype=dtype) for t in (x, g, w)]
        _, mean, inv = torch.ops.aten.native_batch_norm(
            xx, ww, None, None, None, True, .1, 1e-5)
        results.append(torch.ops.aten.native_batch_norm_backward(
            gg, xx, ww, None, None, mean, inv, True, 1e-5, [True]*3))
    for actual, expected in zip(results[1], results[0]):
        torch.testing.assert_close(actual.cpu().double(), expected, rtol=3e-6, atol=0)

@pytest.mark.parametrize('shape', [
    (1, 16, 4, 4), (2, 32, 16, 17), (4, 64, 96, 96),
    (1, 128, 81, 123), (1, 256, 1, 1),
])
def test_eval_elementwise_and_legacy_match_cpu(shape, monkeypatch):
    """Evaluation dispatch can change occupancy without changing BN semantics."""
    gen=torch.Generator().manual_seed(709)
    x=torch.randn(shape, generator=gen)*0.6+0.25
    c=shape[1]
    weight=torch.randn(c, generator=gen)*0.3+1
    bias=torch.randn(c, generator=gen)*0.2
    mean=torch.randn(c, generator=gen)
    var=torch.rand(c, generator=gen)+0.7
    tensors=(x,weight,bias,mean,var)
    def evaluate(inputs):
        return torch.ops.aten.native_batch_norm(*inputs,False,.1,1e-5)[0]
    reference=evaluate(tensors)
    device_inputs=tuple(t.to('vulkan') for t in tensors)
    before=torch_vulkan.execution_stats()['cpu_fallback_calls']
    monkeypatch.delenv('TORCH_VULKAN_BN_EVAL_LEGACY',raising=False)
    parallel=evaluate(device_inputs)
    monkeypatch.setenv('TORCH_VULKAN_BN_EVAL_LEGACY','1')
    legacy=evaluate(device_inputs)
    torch.testing.assert_close(parallel.cpu(),reference,rtol=3e-4,atol=3e-5)
    torch.testing.assert_close(legacy.cpu(),reference,rtol=3e-4,atol=3e-5)
    assert torch_vulkan.execution_stats()['cpu_fallback_calls']==before
