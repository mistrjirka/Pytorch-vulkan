"""Matched FP32 feature-network training with CPU parity and binary provenance."""
import argparse,copy,hashlib,json,os,statistics,sys,time
from pathlib import Path
import torch
from torch.nn import functional as F
from feature_networks import CASES,make_case

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--backend',choices=['vulkan','cuda','cpu'],required=True)
    ap.add_argument('--model',choices=CASES,required=True)
    ap.add_argument('--warmup',type=int,default=12)
    ap.add_argument('--repeats',type=int,default=50)
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();torch.set_num_threads(6)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    report={'model':a.model,'backend':a.backend,'warmup':a.warmup,'repeats':a.repeats,'torch_version':torch.__version__}
    if a.backend=='vulkan':
        import torch_vulkan
        from torch_vulkan.optim import AdamW
        sync=torch_vulkan.synchronize
        report['device']=torch_vulkan.device_name()
        report['package']=torch_vulkan.__file__
        report['extension_sha256']=hashlib.sha256(Path(torch_vulkan._C.__file__).read_bytes()).hexdigest()
        shaders=Path(torch_vulkan.__file__).resolve().parents[1]/'csrc/shaders'
        report['shader_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in shaders.glob('*.spv')}
    else:
        AdamW=lambda p,lr:torch.optim.AdamW(p,lr=lr,foreach=False)
        sync=torch.cuda.synchronize if a.backend=='cuda' else lambda:None
        report['device']=torch.cuda.get_device_name() if a.backend=='cuda' else 'CPU'
    ref,x=make_case(a.model);model=copy.deepcopy(ref).to(a.backend)
    report['input_sha256']=hashlib.sha256(x.numpy().tobytes()).hexdigest()
    report['weights_sha256']=hashlib.sha256(b''.join(v.numpy().tobytes() for v in ref.state_dict().values())).hexdigest()
    xx=x.detach().requires_grad_();inp=x.to(a.backend).detach().requires_grad_()
    target=ref(xx);pred=model(inp)
    target.square().mean().backward();pred.square().mean().backward();sync()
    pairs=[('output',pred.detach().cpu(),target.detach()),('input_gradient',inp.grad.cpu(),xx.grad)]
    pairs +=[(f'gradient/{n}',p.grad.cpu(),q.grad) for (n,p),(_,q) in zip(model.named_parameters(),ref.named_parameters())]
    errors={}
    for n,p,q in pairs:
        torch.testing.assert_close(p,q,rtol=8e-4,atol=8e-5)
        errors[n]=float((p-q).abs().max())
    report['parity_max_abs']=errors
    opt=AdamW(model.parameters(),lr=1e-4)
    samples=[]
    for i in range(a.warmup+a.repeats):
        if os.environ.get("TORCH_VULKAN_TRACE_FALLBACK"):
            print(f"PV_PHASE step={i} measured={i>=a.warmup}",file=sys.stderr,flush=True)
        sync();start=time.perf_counter()
        opt.zero_grad(set_to_none=True);inp.grad=None
        loss=model(inp).square().mean();loss.backward();opt.step()
        sync();ms=(time.perf_counter()-start)*1000
        if i>=a.warmup:samples.append(ms)
    if os.environ.get("TORCH_VULKAN_TRACE_FALLBACK"):
        print("PV_PHASE finished",file=sys.stderr,flush=True)
    report.update(status='ok',training_ms={'median':statistics.median(samples),'min':min(samples),'max':max(samples),'samples':samples},input_shape=list(x.shape))
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2)+'\n')
    print(a.model,a.backend,report['training_ms']['median'],'ms',flush=True)
if __name__=='__main__':main()
