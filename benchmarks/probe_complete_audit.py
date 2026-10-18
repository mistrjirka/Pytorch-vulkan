"""Original 39-case audit, with execution/host-read counters added for verification.

Non-FP32 matrix/add cases and AMP remain explicitly outside this milestone.
"""
import argparse,copy,json,os,sys,traceback,hashlib
from pathlib import Path
import torch
from torch.nn import functional as F
p=argparse.ArgumentParser();p.add_argument('--backend',required=True);p.add_argument('--out',required=True);p.add_argument('--case');a=p.parse_args()
torch.set_num_threads(4);torch.manual_seed(123)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
if a.backend=='vulkan':
 import torch_vulkan as vk
 sync=vk.synchronize; device=vk.device_name()
else:sync=torch.cuda.synchronize;device=torch.cuda.get_device_name()
cases={}
def add(name,fn,shapes,dtype=torch.float32,backward=True):
 xs=[torch.randn(*shape).to(dtype) for shape in shapes];cases[name]=(fn,xs,backward)
add('conv1d',lambda x,w:F.conv1d(x,w,padding=1),[(2,4,13),(6,4,3)])
add('conv2d_10x10',lambda x,w:F.conv2d(x,w),[(1,2,16,17),(3,2,10,10)])
add('conv2d_11x11',lambda x,w:F.conv2d(x,w),[(1,2,16,17),(3,2,11,11)])
add('conv3d',lambda x,w:F.conv3d(x,w,padding=1),[(1,2,5,6,7),(3,2,3,3,3)])
add('conv_transpose2d',lambda x,w:F.conv_transpose2d(x,w,stride=2,padding=1),[(2,4,7,8),(4,3,3,3)])
add('avg_pool3d',lambda x:F.avg_pool3d(x,2),[(2,3,5,6,7)])
add('max_pool3d',lambda x:F.max_pool3d(x,2),[(2,3,5,6,7)])
add('resize_bilinear_aa',lambda x:F.interpolate(x,size=(9,11),mode='bilinear',align_corners=False,antialias=True),[(2,3,13,17)])
add('resize_trilinear',lambda x:F.interpolate(x,size=(7,8,9),mode='trilinear',align_corners=False),[(1,2,3,4,5)])
add('grid_sample',lambda x,g:F.grid_sample(x,g.tanh(),align_corners=False),[(1,2,5,6),(1,3,4,2)])
add('reflection_pad2d',lambda x:F.pad(x,(1,2,2,1),mode='reflect'),[(2,3,5,6)])
for name,dt in [('fp32',torch.float32),('fp16',torch.float16),('bf16',torch.bfloat16),('fp64',torch.float64),('complex64',torch.complex64)]:
 add('mm_'+name,lambda x,w:x@w,[(7,9),(9,5)],dt)
 add('add_'+name,lambda x,y:x+y,[(7,9),(7,9)],dt)
add('add_int64',lambda x,y:x+y,[(7,9),(7,9)],torch.int64,False)
add('sum_int64',lambda x:x.sum(1),[(7,9)],torch.int64,False)
add('index_select',lambda x:torch.index_select(x,1,torch.tensor([3,0,2],device=x.device)),[(4,5)])
add('gather',lambda x:torch.gather(x,1,torch.tensor([[0,2],[1,3],[3,0],[4,2]],device=x.device)),[(4,5)])
add('scatter_add',lambda x,s:torch.scatter_add(x,1,torch.tensor([[0,2],[1,3],[3,0],[4,2]],device=x.device),s),[(4,5),(4,2)])
add('embedding',lambda w:F.embedding(torch.tensor([[0,2,3],[2,1,5]],device=w.device),w),[(7,4)])
add('embedding_bag',lambda w:F.embedding_bag(torch.tensor([0,2,3,2,1,5],device=w.device),w,torch.tensor([0,3],device=w.device),mode='mean'),[(7,4)])
add('sort',lambda x:torch.sort(x,dim=-1).values,[(3,7)])
add('topk',lambda x:torch.topk(x,3,dim=-1).values,[(3,7)])
add('cumsum',lambda x:torch.cumsum(x,dim=-1),[(3,7)])
add('variance',lambda x:torch.var(x,dim=1,correction=0),[(3,7)])
add('fft',lambda x:torch.fft.fft(x),[(3,8)],backward=False)
add('linalg_solve',lambda x,b:torch.linalg.solve(x@x.T+torch.eye(4,device=x.device),b),[(4,4),(4,2)])
add('sparse_roundtrip',lambda x:x.to_sparse().to_dense(),[(3,5)],backward=False)
add('mse_loss',lambda x,y:F.mse_loss(x,y),[(3,5),(3,5)])
add('cross_entropy',lambda x:F.cross_entropy(x,torch.tensor([0,2,1],device=x.device)),[(3,5)])
add('sdpa',lambda q,k,v:F.scaled_dot_product_attention(q,k,v,dropout_p=0),[(2,2,7,8)]*3)
add('sdpa_causal',lambda q,k,v:F.scaled_dot_product_attention(q,k,v,dropout_p=0,is_causal=True),[(2,2,7,8)]*3)
results={'backend':a.backend,'torch':torch.__version__,'torch_git':torch.version.git_version,'device':device,'cases':{},'special':{}}
if a.backend=='vulkan':results['extension_sha256']=hashlib.sha256(Path(vk._C.__file__).read_bytes()).hexdigest()
regkey='PrivateUse1' if a.backend=='vulkan' else 'CUDA';results['direct_registrations']=sorted(torch._C._dispatch_get_registrations_for_dispatch_key(regkey))
def save():Path(a.out).write_text(json.dumps(results,indent=2)+'\n')
for name,(fn,xs,bwd) in cases.items():
 if a.case and a.case!=name:continue
 print('PV_CASE_BEGIN',name,file=sys.stderr,flush=True)
 d={'backward_requested':bwd}
 try:
  ref=[x.detach().requires_grad_(bwd) for x in xs];out=fn(*ref)
  grad=torch.ones_like(out)
  if bwd:out.backward(grad)
  inputs=[x.detach().to(a.backend).requires_grad_(bwd) for x in xs];g=grad.to(a.backend)
  sync();print('PV_OPERATION_BEGIN',name,file=sys.stderr,flush=True)
  if a.backend=='vulkan':vk.reset_execution_stats()
  actual=fn(*inputs);sync();d['forward']='ok'
  if bwd:actual.backward(g);sync();d['backward']='ok'
  if a.backend=='vulkan':
   d['execution']=vk.execution_stats();assert d['execution']['cpu_fallback_calls']==0,d['execution']
  print('PV_OPERATION_END',name,file=sys.stderr,flush=True)
  tol=0.03 if xs[0].dtype in [torch.float16,torch.bfloat16] else 0.0008
  torch.testing.assert_close(actual.cpu(),out,rtol=tol,atol=tol)
  d['output_max_abs']=float((actual.cpu()-out).abs().max()) if out.numel() else 0
  if bwd:
   errs=[]
   for x,y in zip(inputs,ref):
    if y.grad is None:assert x.grad is None;continue
    torch.testing.assert_close(x.grad.cpu(),y.grad,rtol=tol,atol=tol);errs.append(float((x.grad.cpu()-y.grad).abs().max()))
   d['gradient_max_abs']=max(errs,default=0)
  d['status']='ok'
 except Exception as e:d.update(status='error',error_type=type(e).__name__,error=str(e)[:1600])
 results['cases'][name]=d;save();print('PV_CASE_END',name,d['status'],file=sys.stderr,flush=True)
if not a.case:
 def special(name,fn):
  print('PV_SPECIAL_BEGIN',name,file=sys.stderr,flush=True)
  try:r=fn();sync();results['special'][name]={'status':'ok','result':r}
  except Exception as e:results['special'][name]={'status':'error','error_type':type(e).__name__,'error':str(e)[:1600]}
  save();print('PV_SPECIAL_END',name,file=sys.stderr,flush=True)
 def dropout():
  x=torch.ones(2048,device='cpu').to(a.backend).requires_grad_();y=F.dropout(x,p=.25,training=True);y.backward(torch.ones_like(y));z=y.cpu();torch.testing.assert_close(x.grad.cpu(),z);assert set(z.unique().tolist()).issubset({0.,float(torch.tensor(4/3))});return {'dropped':int((z==0).sum())}
 def rng():
  torch.manual_seed(177);cpu_before=torch.get_rng_state().clone();x=torch.randn(16,device=a.backend).cpu();cpu_changed=not torch.equal(cpu_before,torch.get_rng_state());torch.manual_seed(177);y=torch.randn(16,device=a.backend).cpu();return {'torch_manual_seed_repeats':torch.equal(x,y),'changes_cpu_rng_state':cpu_changed}
 def device_seed():
  mod=vk if a.backend=='vulkan' else torch.cuda;mod.manual_seed_all(99);x=torch.randn(32,device=a.backend).cpu();mod.manual_seed_all(99);y=torch.randn(32,device=a.backend).cpu();return {'device_manual_seed_all_repeats':torch.equal(x,y)}
 def generator():return str(torch.Generator(device=a.backend))
 def autocast():
  x=torch.randn(8,8).to(a.backend)
  with torch.autocast(device_type=a.backend,dtype=torch.float16):y=x@x
  return str(y.dtype)
 def lstm():
  ref=torch.nn.LSTM(5,7,batch_first=True);gpu=copy.deepcopy(ref).to(a.backend);x=torch.randn(2,4,5);expected=ref(x)[0];actual=gpu(x.to(a.backend))[0];expected.sum().backward();actual.backward(torch.ones_like(actual));torch.testing.assert_close(actual.cpu(),expected,rtol=1e-3,atol=1e-4)
  for p,q in zip(gpu.parameters(),ref.parameters()):torch.testing.assert_close(p.grad.cpu(),q.grad,rtol=1e-3,atol=1e-4)
  return 'forward/backward CPU parity'
 def optimizer():
  x=torch.ones(8).to(a.backend).requires_grad_();opt=torch.optim.AdamW([x],lr=.01,foreach=False)
  for i in range(2):x.grad=torch.ones_like(x);opt.step()
  return x.detach().cpu().tolist()
 def second_grad():
  x=torch.randn(7).to(a.backend).requires_grad_();y=x.sin();g=torch.autograd.grad(y,x,torch.ones_like(y),create_graph=True)[0];g2=torch.autograd.grad(g,x,torch.ones_like(g))[0];torch.testing.assert_close(g2.cpu(),-x.detach().cpu().sin(),rtol=1e-4,atol=1e-5);return 'sin second derivative matches CPU'
 for name,fn in [('dropout',dropout),('rng',rng),('device_seed',device_seed),('generator',generator),('autocast',autocast),('lstm',lstm),('standard_adamw',optimizer),('second_order_grad',second_grad)]:special(name,fn)
print(json.dumps({'backend':a.backend,'cases':{k:v['status'] for k,v in results['cases'].items()},'special':results['special']},indent=2),flush=True)
