import json,sys,torch
backend=sys.argv[1]
if backend=='vulkan':import torch_vulkan
x=torch.load('/tmp/pv-ldl40-control-input.pt',weights_only=True);a,b=x['a'],x['b'];expected=x['ld'];piv=x['pivots']
ld,pivots,info=torch.linalg.ldl_factor_ex(a.to(backend));ld,pivots,info=ld.cpu(),pivots.cpu(),info.cpu()
error=(ld-expected).abs();allowed=1e-5+1.3e-6*expected.abs()
solution=torch.linalg.ldl_solve(ld.double(),pivots,b.double());residual=(a.double()@solution-b.double()).abs().amax(dim=-2)
scale=a.double().abs().sum(-1).amax(-1).unsqueeze(-1)*solution.abs().amax(-2)+b.double().abs().amax(-2)
print(json.dumps({'backend':backend,'torch':torch.__version__,'torch_git':torch.version.git_version,'same_pivots':torch.equal(pivots,piv),'info':info.tolist(),'factor_max_cpu_difference':error.max().item(),'factor_outside_default_tolerance':int((error>allowed).sum()),'max_normalized_backward_error':(residual/scale).max().item()}),flush=True)
