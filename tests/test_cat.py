"""GPU concatenation, its autograd views, and packing for existing arithmetic."""
import pytest
import torch
import torch_vulkan
from test_batch_norm import layout

@pytest.mark.parametrize('dim',[0,1,2,3,-1,-4])
@pytest.mark.parametrize('kind',['contiguous','channels_last','transpose','slice','broadcast'])
def test_cat_dimensions_layouts_backward(dim,kind):
 torch.manual_seed(33)
 shape=[2,3,4,5];shape[dim]=2
 x=torch.randn(shape);shape[dim]=3;y=torch.randn(shape);shape[dim]=1;z=torch.randn(shape)
 cpu=[layout(t,kind,'cpu').detach().requires_grad_() for t in (x,y,z)]
 vk=[layout(t,kind,'vulkan').detach().requires_grad_() for t in (x,y,z)]
 yc=torch.cat(cpu,dim);yv=torch.cat(vk,dim);g=torch.randn_like(yc)
 yc.backward(g);yv.backward(g.to('vulkan'))
 torch.testing.assert_close(yv.cpu(),yc,atol=0,rtol=0)
 for a,b in zip(vk,cpu):torch.testing.assert_close(a.grad.cpu(),b.grad,atol=0,rtol=0)

@pytest.mark.parametrize('shapes,dim',[
 ([(0,),(0,)],0),([(0,),(2,3),(0,)],1),([(2,0,4),(2,0,4)],1),
 ([(2,0,4),(3,0,4)],0),([(3,)],-1),([(2,3,2,1,2),(2,1,2,1,2)],1),
])
def test_cat_empty_and_general_ranks(shapes,dim):
 cpu=[torch.randn(s) for s in shapes]
 got=torch.cat([t.to('vulkan') for t in cpu],dim)
 torch.testing.assert_close(got.cpu(),torch.cat(cpu,dim),atol=0,rtol=0)

@pytest.mark.parametrize('dtype1,dtype2',[(torch.int64,torch.int64),(torch.float32,torch.float64),(torch.float32,torch.int64)])
def test_cat_preserves_dtype_promotion_fallback(dtype1,dtype2):
 a=torch.arange(6).reshape(2,3).to(dtype1);b=torch.arange(4).reshape(2,2).to(dtype2)
 torch.testing.assert_close(torch.cat([a.to('vulkan'),b.to('vulkan')],1).cpu(),torch.cat([a,b],1))


def test_cat_noncontiguous_arithmetic_no_host_copy():
 torch.manual_seed(18)
 a=torch.randn(2,3,4,5);b=torch.randn(1,3,1,5)
 av=layout(a,'slice','vulkan');bv=layout(b,'transpose','vulkan')
 with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as prof:
  c=torch.add(av,bv,alpha=.25)
  y=torch.cat([torch.relu(c),av],dim=1)
  z=torch.mul(y,2.)
  torch_vulkan.synchronize()
 ref=torch.cat([torch.relu(a+b*.25),a],dim=1)*2
 torch.testing.assert_close(z.cpu(),ref,atol=1e-6,rtol=1e-6)
 assert not any(e.key=='aten::_to_copy' for e in prof.key_averages())

@pytest.mark.parametrize('shapes,dim',[([(2,3),(3,4)],0),([(),()],0),([(2,3),(2,3)],2)])
def test_cat_invalid_shapes(shapes,dim):
 for device in ['cpu','vulkan']:
  with pytest.raises((RuntimeError,IndexError)):
   torch.cat([torch.ones(s).to(device) for s in shapes],dim)

@pytest.mark.parametrize('scalar',[2,2.5,torch.tensor(2.5,dtype=torch.float64),torch.tensor(-3,dtype=torch.int64)])
@pytest.mark.parametrize('reverse',[False,True])
def test_cpu_scalar_promotion_and_layout(scalar,reverse):
 x=torch.randn(2,3,4,5)
 xv=layout(x,'slice','vulkan')
 a,b=(scalar,x) if reverse else (x,scalar)
 av,bv=(scalar,xv) if reverse else (xv,scalar)
 for op in [lambda a,b:torch.mul(a,b),lambda a,b:torch.add(a,b,alpha=.25)]:
  got=op(av,bv)
  assert got.dtype==torch.float32
  torch.testing.assert_close(got.cpu(),op(a,b),atol=1e-6,rtol=1e-6)


def test_cat_repeated_bindings_and_deferred_chain():
 x=torch.randn(2,3,4)
 def run(device):
  v=x.to(device).detach().requires_grad_()
  first=torch.cat([v,v,v],-1)
  out=torch.cat([first,first],1)
  out.sum().backward()
  return out,v.grad
 for a,b in zip(run('vulkan'),run('cpu')):torch.testing.assert_close(a.cpu(),b,rtol=0,atol=0)


def test_cpu_scalar_tensor_autograd():
 x=torch.randn(2,3)
 def run(device):
  v=x.to(device).detach().requires_grad_();s=torch.tensor(2.5,requires_grad=True)
  out=torch.add(v*s,s,alpha=.25)
  out.sum().backward()
  return out,v.grad,s.grad
 for a,b in zip(run('vulkan'),run('cpu')):torch.testing.assert_close(a.cpu(),b,atol=1e-6,rtol=1e-6)
