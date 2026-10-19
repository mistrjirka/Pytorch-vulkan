import json,torch
print(json.dumps({'torch':torch.__version__,'torch_git':torch.version.git_version,'device':torch.cuda.get_device_name(0),'uuid':str(torch.cuda.get_device_properties(0).uuid)}),flush=True)
values=[0,1,2**63-1,2**63,2**64-1]
for device in ['cpu','cuda']:
 x=torch.tensor(values,dtype=torch.uint64,device=device)
 for name in ['eq','ne','lt','le','gt','ge']:
  try:
   result=getattr(torch,name)(x[:,None],x[None,:]);print('uint64',device,name,result.cpu().tolist(),flush=True)
  except RuntimeError as e:print('uint64',device,name,'unsupported',str(e).splitlines()[0],flush=True)
 for value in [-1,2**64-1,3.5]:print('fill',device,value,torch.empty(3,dtype=torch.uint64,device=device).fill_(value).cpu().tolist(),flush=True)
 try:
  g=torch.Generator(device=device).manual_seed(823);r=torch.empty(10000,dtype=torch.uint64,device=device).random_(5,100,generator=g);print('bounded_rng',device,all(5<=v<100 for v in r.cpu().tolist()),flush=True)
 except RuntimeError as e:print('bounded_rng',device,str(e).splitlines()[0],flush=True)
for p in [0.,.5,1.,2.,3.,float('inf')]:
 x=torch.randn(0,3,device='cuda',requires_grad=True);y=torch.pdist(x,p);y.sum().backward();print('cuda_empty_pdist',p,list(x.grad.shape),x.grad.numel(),flush=True)
for dtype in [torch.float32,torch.int32,torch.int64,torch.bool]:
 x=torch.tensor([0,1,2,3,1],dtype=dtype);a=torch.hash_tensor(x);b=torch.hash_tensor(x.cuda()).cpu();assert torch.equal(a,b);print('hash_tensor_cpu_cuda',str(dtype),a.tolist(),flush=True)
x=torch.tensor([-2.,-1.,-.5,0.,.5,1.,2.]);assert torch.equal(torch.histc(x,4,-1,1),torch.histc(x.cuda(),4,-1,1).cpu());print('histc_cpu_cuda_boundaries',True,flush=True)
