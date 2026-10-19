import json,torch,sys
backend=sys.argv[1]
if backend=='vulkan':import torch_vulkan
if backend=='cuda':torch.cuda.set_device(0)
torch.manual_seed(757);source=torch.randn(3,4,5).transpose(0,2);source[1,2,0]=float('nan');source[2,:,1]=float('nan');rows=[]
for ignore in [False,True]:
 for interpolation in ['linear','lower','higher','midpoint','nearest']:
  fn=torch.nanquantile if ignore else torch.quantile
  for dim,keep in [(None,False),(0,False),(1,True),(-1,True)]:
   for q in [torch.tensor(.37),torch.tensor([0.,.1,.5,.9,1.])]:
    host=source.detach().requires_grad_();device=source.to(backend).requires_grad_()
    a=fn(device,q.to(backend),dim=dim,keepdim=keep,interpolation=interpolation);b=fn(host,q,dim=dim,keepdim=keep,interpolation=interpolation)
    ga=torch.autograd.grad(a.sum(),device)[0].cpu();gb=torch.autograd.grad(b.sum(),host)[0];same=torch.isclose(ga,gb,equal_nan=True)
    rows.append({'ignore_nan':ignore,'interpolation':interpolation,'dim':dim,'keep':keep,'q':q.tolist(),'different_gradients':int((~same).sum()),'locations':torch.nonzero(~same).tolist(),'backend_grad':ga.tolist(),'cpu_grad':gb.tolist()})
json.dump(rows,open('/tmp/pv-quantile29-'+backend+'.json','w'));print(backend,'cases',len(rows),'gradient cases different from CPU',sum(r['different_gradients']>0 for r in rows))
