"""Diagnose large-batch ResNet gradient sensitivity; excluded from timing traces."""
import argparse,copy,hashlib,json,sys
from pathlib import Path
import torch
from compare_cuda_vulkan import data_for_case
from benchmark_network_matrix import digest

def metric(a,b):
    a,b=a.double(),b.double()
    return dict(max_abs=float((a-b).abs().max()),relative_l2=float((a-b).norm()/b.norm().clamp_min(1e-30)),reference_max=float(b.abs().max()))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--backend",choices=["cuda","vulkan"],required=True);p.add_argument("--out",type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if a.backend=="vulkan":import torch_vulkan
    sys.path.insert(0,"/workspace/pytorch-XDNA2-vulkan-zoo-reference/benchmarks");import model_zoo
    torch.manual_seed(42);base,_,loss=model_zoo.MODELS["resnet18"]()
    x,target=data_for_case("resnet18",(32,3,64,64),torch.Generator().manual_seed(123))
    report=dict(backend=a.backend,shape=list(x.shape),weights_sha256=digest(base.state_dict().values()),input_sha256=digest([x,target]),diagnostic="ReLU snapshots intentionally copied to CPU; never used in timed/profiled workload",torch=torch.__version__,torch_git=torch.version.git_version)
    def run(device,dtype):
        m=copy.deepcopy(base).to(device=device,dtype=dtype).train();inp=x.detach().to(device=device,dtype=dtype).requires_grad_();label=target.to(device)
        acts={};counts={};handles=[]
        def hook(name):
            def capture(module,args):
                index=counts.get(name,0);counts[name]=index+1
                acts[f"{name}/{index}"]=args[0].detach().cpu().clone()
            return capture
        for name,module in m.named_modules():
            if isinstance(module,torch.nn.ReLU):handles.append(module.register_forward_pre_hook(hook(name)))
        out=m(inp);value=loss(out,label);value.backward()
        for h in handles:h.remove()
        grads={"input":inp.grad.detach().cpu(),**{n:p.grad.detach().cpu() for n,p in m.named_parameters()}}
        return dict(output=out.detach().cpu(),grads=grads,activations=acts)
    cpu=run("cpu",torch.float32);double=run("cpu",torch.float64);gpu=run(a.backend,torch.float32)
    report["comparisons"]={}
    for label,actual,reference in [("gpu_vs_cpu32",gpu,cpu),("gpu_vs_cpu64",gpu,double),("cpu32_vs_cpu64",cpu,double)]:
        d=dict(output=metric(actual["output"],reference["output"]),gradients={k:metric(v,reference["grads"][k]) for k,v in actual["grads"].items()},relu_sign_disagreements=[])
        for name,act in actual["activations"].items():
            ref=reference["activations"][name];mask=(act>0)!=(ref>0)
            if mask.any():
                d["relu_sign_disagreements"].append(dict(name=name,count=int(mask.sum()),actual_abs_max=float(act[mask].abs().max()),reference_abs_max=float(ref[mask].abs().max())))
        report["comparisons"][label]=d
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2)+"\n")
    for k,d in report["comparisons"].items():print(k,d["output"],"conv1 grad",d["gradients"]["conv1.weight"],"ReLU sign disagreements",sum(v["count"] for v in d["relu_sign_disagreements"]))

if __name__=="__main__":main()
