import gc
import pytest
import torch
import torch_vulkan

def test_stream_context_events_and_lifetimes():
    default=torch_vulkan.current_stream()
    s=torch.Stream(device='vulkan');other=torch_vulkan.Stream()
    assert s!=default and other!=s
    begin=torch_vulkan.Event(enable_timing=True);end=torch_vulkan.Event(enable_timing=True)
    with s:
        assert torch_vulkan.current_stream()==s
        begin.record(s)
        x=torch.arange(64.,device='cpu').to('vulkan')
        y=x.sin().mul(3)
        del x
        end.record(s)
    assert torch_vulkan.current_stream()==default
    with other:
        other.wait_event(end)
        z=y.cos()
    end.synchronize();assert end.query()
    assert begin.elapsed_time(end)>=0
    torch.testing.assert_close(z.cpu(),(torch.arange(64.).sin()*3).cos(),rtol=1e-4,atol=3e-5)
    other.synchronize();assert other.query()
    for _ in range(5):
        with s:
            y=torch.ones(1000,device='vulkan').sin()
            end.record(s)
        del y
    gc.collect();torch_vulkan.synchronize();assert end.query()

@pytest.mark.parametrize('training',[False,True])
def test_default_compile(training):
    torch._dynamo.reset()
    def fn(x,w):return (x.sin()@w).relu().square().mean()
    x=torch.randn(7,9,requires_grad=training);w=torch.randn(9,5,requires_grad=training)
    vx=x.detach().to('vulkan').requires_grad_(training);vw=w.detach().to('vulkan').requires_grad_(training)
    compiled=torch.compile(fn,fullgraph=True)
    expected=fn(x,w);actual=compiled(vx,vw)
    torch.testing.assert_close(actual.cpu(),expected,rtol=3e-4,atol=3e-5)
    if training:
        expected.backward();actual.backward()
        for a,b in ((vx,x),(vw,w)):torch.testing.assert_close(a.grad.cpu(),b.grad,rtol=4e-4,atol=4e-5)
    # The second invocation runs the cached compiled wrapper.
    torch.testing.assert_close(compiled(vx,vw).cpu(),expected,rtol=3e-4,atol=3e-5)

@pytest.mark.parametrize('dropout',[0.,.2])
def test_compiled_attention(dropout):
    import torch.nn.functional as F
    torch.manual_seed(17)
    inputs=[torch.randn(2,3,5,7,device='vulkan').requires_grad_() for _ in range(3)]
    fn=lambda q,k,v:F.scaled_dot_product_attention(q,k,v,is_causal=True,dropout_p=dropout)
    torch_vulkan.manual_seed(43);expected=fn(*inputs);g=torch.randn_like(expected)
    grads=torch.autograd.grad(expected,inputs,g)
    compiled=torch.compile(fn,fullgraph=True)
    torch_vulkan.manual_seed(43);actual=compiled(*inputs);got=torch.autograd.grad(actual,inputs,g)
    torch.testing.assert_close(actual.cpu(),expected.cpu(),rtol=3e-4,atol=3e-5)
    for x,y in zip(got,grads):torch.testing.assert_close(x.cpu(),y.cpu(),rtol=4e-4,atol=4e-5)


def test_compiled_dropout_device_rng():
    import torch.nn.functional as F
    fn=torch.compile(lambda x:F.dropout(x,.3,True),fullgraph=True)
    x=torch.ones(11,13,device='vulkan').requires_grad_()
    torch_vulkan.manual_seed(71);cpu=torch.get_rng_state().clone();first=fn(x)
    torch_vulkan.manual_seed(71);second=fn(x)
    torch.testing.assert_close(first.cpu(),second.cpu(),rtol=0,atol=0)
    assert torch.equal(cpu,torch.get_rng_state())
    first.sum().backward();torch.testing.assert_close(x.grad.cpu(),first.cpu())
    torch_vulkan.reset_execution_stats();fn(x).sum().backward();stats=torch_vulkan.execution_stats()
    assert stats['cpu_fallback_calls']==0 and stats['host_read_max_bytes']==0,stats


def test_default_compile_spatial_network():
    import torch.nn.functional as F
    fn=lambda x,w:F.avg_pool3d(F.conv3d(x,w,padding=1).relu(),2).square().mean()
    x=torch.randn(2,2,6,8,10,requires_grad=True);w=torch.randn(3,2,3,3,3,requires_grad=True)
    inputs=[t.detach().to('vulkan').requires_grad_() for t in (x,w)]
    compiled=torch.compile(fn,fullgraph=True);a=fn(x,w);b=compiled(*inputs)
    a.backward();b.backward();torch.testing.assert_close(b.cpu(),a,rtol=4e-4,atol=4e-5)
    for v,c in zip(inputs,(x,w)):torch.testing.assert_close(v.grad.cpu(),c.grad,rtol=6e-4,atol=6e-5)
