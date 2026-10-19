import json,sys,torch
backend=sys.argv[1]
if backend=='vulkan':import torch_vulkan as vk
x=torch.load('/tmp/pv-native44-control-input.pt',weights_only=True)
def compare(actual,expected):
 a=actual.detach().cpu();b=expected.detach();diff=(a-b).abs();allowed=1e-5+1.3e-6*b.abs();bad=~torch.isclose(a,b,rtol=1.3e-6,atol=1e-5,equal_nan=True)
 return {'outside_default_tolerance':int(bad.sum()),'max_finite_difference':diff[torch.isfinite(diff)].max().item() if torch.isfinite(diff).any() else 0}
result={'backend':backend,'torch':torch.__version__,'torch_git':torch.version.git_version,'renorm':{}}
for p in [0.,1.,2.,-1.,float('inf')]:
 expected=x['embedding'].clone();torch.embedding_renorm_(expected,x['indices'],.75,p)
 actual=x['embedding'].to(backend);torch.embedding_renorm_(actual,x['indices'].to(backend),.75,p);result['renorm'][str(p)]=compare(actual,expected)
outputs=[]
for device in ['cpu',backend]:
 a=x['a'].to(device).to_sparse_csr().detach().requires_grad_();b=x['b'].to(device).detach().requires_grad_();c=x['c'].to(device).detach().requires_grad_()
 out=torch.sparse.sampled_addmm(a,b,c,alpha=.6,beta=.2);out.to_dense().square().sum().backward();outputs.append([out.to_dense(),a.grad.to_dense(),b.grad,c.grad])
result['sampled_addmm']=[compare(a,b) for a,b in zip(outputs[1],outputs[0])]
try:
 a=x['a'].to(backend).to_sparse_csr();b=x['c'].T.to(backend);y=torch.sparse.mm(a,b,reduce='mean');result['spmm_reduce_mean']='supported'
except Exception as e:result['spmm_reduce_mean']={'status':'unsupported','type':type(e).__name__,'message':str(e).splitlines()[0]}
if backend=='vulkan':result['execution_stats']=vk.execution_stats()
print(json.dumps(result,allow_nan=False),flush=True)
