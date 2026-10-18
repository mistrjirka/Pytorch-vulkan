"""CUDA GPU-event versus Vulkan GPU-timestamp FP32 GEMM comparison."""
import argparse,json,time,statistics,pathlib,os
import torch
SHAPES=[(128,784,1024),(128,1024,1024),(128,1024,10),
        (1024,128,1024),(32,1024,1024),(256,1024,1024),(1024,1024,1024)]
p=argparse.ArgumentParser()
p.add_argument('--backend',choices=['cuda','vulkan'],required=True)
p.add_argument('--out',type=pathlib.Path)
p.add_argument('--repeats',type=int,default=16)
a=p.parse_args()
torch.set_num_threads(8)
if a.backend=='cuda':
    assert torch.cuda.is_available()
    torch.backends.cuda.matmul.allow_tf32=False
    name=torch.cuda.get_device_name(0)
else:
    import torch_vulkan
    assert torch_vulkan.is_hardware_accelerated()
    name=torch_vulkan.device_name()
report=dict(device=name,torch=torch.__version__,backend=a.backend,shapes=[])
for m,k,n in SHAPES:
    torch.manual_seed(m+k+n)
    xcpu=torch.randn(m,k);wcpu=torch.randn(k,n)
    expected=xcpu@wcpu
    x=xcpu.to(a.backend);w=wcpu.to(a.backend)
    torch.testing.assert_close((x@w).cpu(),expected,rtol=4e-4,atol=4e-3)
    for _ in range(6):t=x@w
    if a.backend=='cuda':torch.cuda.synchronize()
    gpu_times=[];wall=[]
    if a.backend=='vulkan':
        profile_key=('mm_register' if (os.environ.get('TORCH_VULKAN_MM_KERNEL')=='register' or (os.environ.get('TORCH_VULKAN_MM_KERNEL')!='tiled' and m>=256 and n>=256 and k>=64)) else 'mm')
        before=torch_vulkan.gpu_timing_stats().get(profile_key,{})
    for _ in range(a.repeats):
        if a.backend=='cuda':
            s=torch.cuda.Event(enable_timing=True);e=torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize();start=time.perf_counter()
            s.record();t=x@w;e.record();e.synchronize()
            gpu_times.append(s.elapsed_time(e))
        else:
            start=time.perf_counter();t=x@w
        wall.append((time.perf_counter()-start)*1000)
    row=dict(m=m,k=k,n=n,wall_ms=round(statistics.median(wall),4))
    if a.backend=='cuda':
        row['kernel_ms']=round(statistics.median(gpu_times),5)
    else:
        after=torch_vulkan.gpu_timing_stats()[profile_key]
        count=after['samples']-before.get('samples',0)
        assert count==a.repeats,(before,after)
        for field in ('upload_ms','shader_ms','download_ms','host_eval_ms'):
            row[field]=round((after[field]-before.get(field,0))/count,5)
    report['shapes'].append(row)
    print(f"{m},{k},{n}: {row}",flush=True)
if a.out:
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2)+'\n')
