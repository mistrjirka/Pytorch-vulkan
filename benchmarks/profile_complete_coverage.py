"""Matched completion-network training; separate wall and instrumented GPU runs."""
import argparse,copy,hashlib,json,os,statistics,time
from pathlib import Path
import torch
from feature_networks import COMPLETION_CASES,make_case

def summary(values):
    return dict(median=statistics.median(values),min=min(values),max=max(values),samples=values)

def delta(before,after):
    return {name:{key:value-before.get(name,{}).get(key,0) for key,value in row.items()} for name,row in after.items() if row.get('samples',0)>before.get(name,{}).get('samples',0)}

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--backend',choices=['vulkan','cuda'],required=True);p.add_argument('--mode',choices=['wall','gpu'],default='wall');p.add_argument('--warmup',type=int,default=8);p.add_argument('--repeats',type=int,default=25);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if a.backend=='vulkan':
        import torch_vulkan as vk
        sync=vk.synchronize;device=vk.device_name()
        if a.mode=='gpu':assert os.environ.get('TORCH_VULKAN_PROFILE_TIMESTAMPS')=='1' and os.environ.get('TORCH_VULKAN_DEFER_SUBMISSIONS')!='1'
    else:sync=torch.cuda.synchronize;device=torch.cuda.get_device_name()
    report=dict(backend=a.backend,mode=a.mode,device=device,torch=torch.__version__,torch_git=torch.version.git_version,optimizer='torch.optim.AdamW(foreach=False,amsgrad=True)',dtype='float32',threads=4,warmup=a.warmup,repeats=a.repeats,models={})
    if a.backend=='vulkan':report['extension_sha256']=hashlib.sha256(Path(vk._C.__file__).read_bytes()).hexdigest()
    for name in COMPLETION_CASES:
        cpu,x=make_case(name);model=copy.deepcopy(cpu).to(a.backend);inp=x.to(a.backend).requires_grad_();opt=torch.optim.AdamW(model.parameters(),lr=.0001,foreach=False,amsgrad=True)
        cpu_input=x.clone().requires_grad_();expected=cpu(cpu_input);observed=model(inp)
        expected.square().mean().backward();observed.square().mean().backward();sync()
        pairs=[('output',observed.detach().cpu(),expected.detach()),('input_gradient',inp.grad.cpu(),cpu_input.grad)]
        pairs += [(f'gradient/{name}',p.grad.cpu(),q.grad) for (name,p),(_,q) in zip(model.named_parameters(),cpu.named_parameters())]
        parity={}
        for label,actual,reference in pairs:
            torch.testing.assert_close(actual,reference,rtol=2e-3,atol=8e-5);parity[label]=float((actual-reference).abs().max())
        def step():
            opt.zero_grad(set_to_none=True);inp.grad=None
            y=model(inp);y.square().mean().backward();opt.step()
        for _ in range(a.warmup):step()
        sync();row={'parity_max_abs':parity,'shape':list(x.shape),'input_sha256':hashlib.sha256(x.numpy().tobytes()).hexdigest(),'weights_sha256':hashlib.sha256(b''.join(v.numpy().tobytes() for v in cpu.state_dict().values())).hexdigest()}
        if a.backend=='vulkan':vk.reset_execution_stats();before=vk.gpu_timing_stats();qb=vk.queue_stats();cb=vk.cache_stats()
        samples=[]
        if a.mode=='gpu' and a.backend=='cuda':
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                for _ in range(a.repeats):step()
                sync()
            row['kernels']=[{'name':e.name,'duration_us':e.time_range.elapsed_us()} for e in prof.events() if e.device_type==torch.autograd.DeviceType.CUDA and not e.is_user_annotation]
            row['operators']=[{'name':e.key,'calls':e.count,'self_cpu_us':e.self_cpu_time_total,'self_device_us':e.self_device_time_total} for e in prof.key_averages()]
        else:
            for _ in range(a.repeats):
                sync();start=time.perf_counter();step();sync();samples.append((time.perf_counter()-start)*1000)
            row['training_ms']=summary(samples)
        if a.backend=='vulkan':
            row['execution']=vk.execution_stats();assert row['execution']['cpu_fallback_calls']==0 and row['execution']['host_read_max_bytes']<=4,row['execution']
            row['gpu_timing']=delta(before,vk.gpu_timing_stats());row['queue_delta']={k:v-qb.get(k,0) for k,v in vk.queue_stats().items()};row['cache_delta']={k:v-cb.get(k,0) for k,v in vk.cache_stats().items()}
        report['models'][name]=row;a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2)+'\n');print(name,row.get('training_ms',{}).get('median','GPU profile'),flush=True)
    return report
if __name__=='__main__':main()
