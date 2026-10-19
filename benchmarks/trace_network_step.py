"""Trace normal deferred training steps; synchronize only at step boundaries."""
import argparse,copy,hashlib,json,os,sys,time
from pathlib import Path
import nvtx
import torch
from benchmark_network_matrix import ZOO_CASES,digest,delta
from compare_cuda_vulkan import CONFIG,data_for_case
from feature_networks import make_case

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend",choices=["cuda","vulkan"],required=True);p.add_argument("--case",required=True);p.add_argument("--scale",default="medium");p.add_argument("--warmup",type=int,default=8);p.add_argument("--repeats",type=int,default=3);p.add_argument("--out",type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(4);torch.set_num_interop_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    vk=None
    if a.backend=="vulkan":
        import torch_vulkan as vk
        assert os.environ.get("TORCH_VULKAN_DEFER_SUBMISSIONS")=="1"
        assert os.environ.get("TORCH_VULKAN_PROFILE_TIMESTAMPS")!="1"
        sync=vk.synchronize;device=vk.device_name()
    else:sync=torch.cuda.synchronize;device=torch.cuda.get_device_name()
    torch.manual_seed(42)
    if a.case in ZOO_CASES:
        sys.path.insert(0,"/workspace/pytorch-XDNA2-vulkan-zoo-reference/benchmarks");import model_zoo
        cpu,_,loss_fn=model_zoo.MODELS[a.case]()
        x,target=data_for_case(a.case,CONFIG[a.case][a.scale],torch.Generator().manual_seed(123))
    else:
        cpu,x=make_case(a.case);target=None;loss_fn=lambda out,_:out.square().mean()
    row=dict(case=a.case,backend=a.backend,scale=a.scale,mode="normal_trace",shape=list(x.shape),device=device,torch=torch.__version__,torch_git=torch.version.git_version,repeats=a.repeats,warmup=a.warmup,optimizer="torch.optim.AdamW(lr=1e-4,foreach=False,amsgrad=True)",input_sha256=digest([x]+([target] if target is not None else [])),weights_sha256=digest(cpu.state_dict().values()),driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),synchronization="before/after each full training step; no forced phase synchronization",phase_ranges="host enqueue attribution only",environment={k:v for k,v in os.environ.items() if k.startswith("TORCH_VULKAN_") or k=="CUDA_VISIBLE_DEVICES"})
    m=cpu.to(a.backend).train();inp=x.to(a.backend).requires_grad_(x.is_floating_point());label=target.to(a.backend) if target is not None else None
    opt=torch.optim.AdamW(m.parameters(),lr=1e-4,foreach=False,amsgrad=True)
    h={}
    def zero():opt.zero_grad(set_to_none=True);inp.grad=None
    def forward():h["out"]=m(inp)
    def loss():h["loss"]=loss_fn(h["out"],label)
    def backward():h["loss"].backward()
    phases=dict(zero_grad=zero,forward=forward,loss=loss,backward=backward,optimizer=opt.step)
    for _ in range(a.warmup):
        for fn in phases.values():fn()
        h.clear()
    sync()
    if vk:vk.reset_execution_stats();qb,cb=vk.queue_stats(),vk.cache_stats()
    samples=[]
    for _ in range(a.repeats):
        sync()
        with nvtx.annotate("measured_step",domain="pv-profiling"):
            start=time.perf_counter()
            for name,fn in phases.items():
                with nvtx.annotate(name,domain="pv-profiling"):fn()
            h.clear()
            with nvtx.annotate("final_sync",domain="pv-profiling"):sync()
            samples.append((time.perf_counter()-start)*1000)
    row["instrumented_step_ms"]=samples
    if vk:
        row["execution"]=vk.execution_stats();row["queue_delta"]=delta(qb,vk.queue_stats());row["cache_delta"]=delta(cb,vk.cache_stats())
        assert row["execution"]["cpu_fallback_calls"]==0 and row["execution"]["host_read_max_bytes"]<=4
    row["status"]="passed";a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(row,indent=2)+"\n")
    print(a.case,a.backend,row["instrumented_step_ms"],flush=True)
if __name__=="__main__":main()
