import json,sys,torch
backend=sys.argv[1]
if backend=='vulkan':import torch_vulkan
x=torch.load('/tmp/pv-ldl40-control-input.pt',weights_only=True)
ld,pivots,b,a,expected=[x[n] for n in ['ld','pivots','b','a','expected']]
actual=torch.linalg.ldl_solve(ld.to(backend),pivots.to(backend),b.to(backend)).cpu()
gold=torch.linalg.solve(a.double(),b.double())
difference=(actual-expected).abs();allowed=1e-5+1.3e-6*expected.abs()
print(json.dumps({'backend':backend,'torch':torch.__version__,'torch_git':torch.version.git_version,'max_cpu_difference':difference.max().item(),'outside_default_tolerance':int((difference>allowed).sum()),'max_normalized_difference':(difference/allowed).max().item(),'actual_fp64_error':(actual.double()-gold).abs().max().item(),'cpu_fp64_error':(expected.double()-gold).abs().max().item(),'actual_fp64_residual':(a.double()@actual.double()-b.double()).abs().max().item(),'cpu_fp64_residual':(a.double()@expected.double()-b.double()).abs().max().item()}),flush=True)
