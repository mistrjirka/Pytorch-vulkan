"""FP32 max pooling values, exact int64 indices, gradients and runtime layouts."""
import pytest
import torch
import torch.nn.functional as F
import torch_vulkan
from test_batch_norm import layout

CASES=[
 ((2,3,9,11),2,None,0,1,False),
 ((1,5,7,9),(3,5),(2,3),(1,2),1,False),
 ((2,3,9,11),(3,2),(1,2),(1,1),(2,3),True),
 ((1,4,8,7),3,1,1,1,False),
 ((1,2,5,6),(2,3),(3,4),0,1,True),
 ((1,2,1,7),(1,3),(1,2),0,1,True),
 ((2,3,7,9),1,1,0,1,False),
 ((1,3,13,17),(5,7),(2,3),(2,3),(2,1),True),
]
@pytest.mark.parametrize('shape,kernel,stride,pad,dilation,ceil',CASES)
@pytest.mark.parametrize('kind',['contiguous','channels_last','transpose','slice','broadcast'])
def test_pool_values_indices_gradients(shape,kernel,stride,pad,dilation,ceil,kind):
 torch.manual_seed(912)
 x=torch.randn(shape)
 xc=layout(x,kind,'cpu').detach().requires_grad_();xv=layout(x,kind,'vulkan').detach().requires_grad_()
 yc,ic=F.max_pool2d(xc,kernel,stride,pad,dilation,ceil,return_indices=True)
 yv,iv=F.max_pool2d(xv,kernel,stride,pad,dilation,ceil,return_indices=True)
 g=torch.randn(yc.shape)
 yc.backward(layout(g,kind,'cpu'));yv.backward(layout(g,kind,'vulkan'))
 torch.testing.assert_close(yv.cpu(),yc,atol=0,rtol=0)
 assert iv.dtype==torch.int64
 torch.testing.assert_close(iv.cpu(),ic,atol=0,rtol=0)
 torch.testing.assert_close(xv.grad.cpu(),xc.grad,atol=1e-6,rtol=1e-6)

@pytest.mark.parametrize('kind',['slice','transpose','broadcast'])
def test_pool_strided_saved_indices(kind):
 torch.manual_seed(19)
 x=torch.randn(2,3,9,11)
 y,indices=F.max_pool2d(x,3,1,1,return_indices=True)
 indices=layout(indices,kind,'cpu')
 # Broadcast must describe identical per-batch windows on both devices.
 g=torch.randn(y.shape)
 ref=torch.ops.aten.max_pool2d_with_indices_backward(g,x,[3],[1],[1],[1],False,indices)
 got=torch.ops.aten.max_pool2d_with_indices_backward(layout(g,'slice','vulkan'),x.to('vulkan'),
   [3],[1],[1],[1],False,layout(indices,kind,'vulkan'))
 torch.testing.assert_close(got.cpu(),ref,atol=1e-6,rtol=1e-6)

@pytest.mark.parametrize('values',[
 [1.,1.,1.,1.],[-float('inf')]*4,
 [float('nan'),1.,2.,float('nan')],[-0.,0.,0.,0.],
 [1.,float('inf'),float('inf'),-1.],
])
def test_pool_ties_nan_infinity(values):
 x=torch.tensor(values).reshape(1,1,2,2)
 xc=x.clone().requires_grad_();xv=x.to('vulkan').requires_grad_()
 yc,ic=F.max_pool2d(xc,2,return_indices=True);yv,iv=F.max_pool2d(xv,2,return_indices=True)
 yc.backward(torch.ones_like(yc));yv.backward(torch.ones_like(yv))
 torch.testing.assert_close(yv.cpu(),yc,atol=0,rtol=0,equal_nan=True)
 torch.testing.assert_close(iv.cpu(),ic,atol=0,rtol=0)
 torch.testing.assert_close(xv.grad.cpu(),xc.grad,atol=0,rtol=0)
 assert torch.equal(torch.signbit(yv.cpu()),torch.signbit(yc))

@pytest.mark.parametrize('shape',[(3,7,9),(0,3,7,9)])
def test_pool_unbatched_and_empty(shape):
 x=torch.randn(shape)
 xc=x.clone().requires_grad_();xv=x.to('vulkan').requires_grad_()
 yc,ic=F.max_pool2d(xc,3,2,1,ceil_mode=True,return_indices=True)
 yv,iv=F.max_pool2d(xv,3,2,1,ceil_mode=True,return_indices=True)
 yc.backward(torch.ones_like(yc));yv.backward(torch.ones_like(yv))
 for a,b in [(yv.cpu(),yc),(iv.cpu(),ic),(xv.grad.cpu(),xc.grad)]:torch.testing.assert_close(a,b,atol=0,rtol=0)


def test_pool_empty_dilated_window_forward():
 x=torch.ones(1,1,2,2)
 ref=F.max_pool2d(x,2,1,1,3,True,return_indices=True)
 got=F.max_pool2d(x.to('vulkan'),2,1,1,3,True,return_indices=True)
 for a,b in zip(got,ref):torch.testing.assert_close(a.cpu(),b,atol=0,rtol=0)


def test_pool_deferred_chain_no_fallback():
 x=torch.randn(2,3,9,11)
 def run(device):
  v=x.to(device).detach().requires_grad_()
  y=F.max_pool2d(F.max_pool2d(v,3,1,1),2,2,ceil_mode=True)
  y.sum().backward()
  return y,v.grad
 ref=run('cpu');got=run('vulkan')
 for a,b in zip(got,ref):torch.testing.assert_close(a.cpu(),b,atol=0,rtol=0)
 xv=x.to('vulkan')
 with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
  y,idx=torch.ops.aten.max_pool2d_with_indices(xv,[3],[1],[1],[1],False)
  torch.ops.aten.max_pool2d_with_indices_backward(y,xv,[3],[1],[1],[1],False,idx)
  torch_vulkan.synchronize()
 assert not any(e.key=='aten::_to_copy' for e in prof.key_averages())

@pytest.mark.parametrize('kernel,stride,pad,dilation',[(0,1,0,1),(3,0,0,1),(3,1,2,1),(3,1,0,0),(11,1,0,1)])
def test_pool_invalid_parameters(kernel,stride,pad,dilation):
 for device in ['cpu','vulkan']:
  with pytest.raises(RuntimeError):F.max_pool2d(torch.ones(1,2,5,7).to(device),kernel,stride,pad,dilation)
