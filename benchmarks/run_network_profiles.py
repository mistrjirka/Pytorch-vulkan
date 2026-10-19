"""Separate matched phase, GPU-timestamp and normal-batching Nsight profiles."""
import argparse,json,os,subprocess,time
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cases",nargs="+",default=["mlp","unet","resnet18","mobilenet_v3","embedding","transformer","volume_autoencoder","recurrent_regressor","causal_attention_block","spectral_filter"])
    p.add_argument("--modes",nargs="+",choices=["phases","gpu","nsys"],default=["phases","gpu","nsys"])
    p.add_argument("--out",type=Path,default=Path("benchmarks/results/network-matrix-2026-10-19/diagnostics"))
    p.add_argument("--cuda-device",default="GPU-30acbf2b-a41a-8b88-e298-fe66c4138b29");p.add_argument("--vulkan-device",default="0")
    a=p.parse_args();root=Path(__file__).resolve().parents[1];out=a.out.resolve();out.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy();env.update(PYTHONPATH=str(root),CUDA_VISIBLE_DEVICES=a.cuda_device,TORCH_VULKAN_PHYSICAL_DEVICE_INDEX=a.vulkan_device,TORCH_VULKAN_GPU_RESIDENT="1",TORCH_VULKAN_FORBID_CPU_FALLBACK="1",OMP_NUM_THREADS="4",MKL_NUM_THREADS="4")
    records=json.loads((out/"index.json").read_text()) if (out/"index.json").exists() else []
    for mode in a.modes:
        cases=a.cases if mode!="nsys" else [n for n in a.cases if n in ["mlp","unet","resnet18","mobilenet_v3","transformer"]]
        pairs=[(name,"medium") for name in cases]+[(name,"reference") for name in cases if name in ["unet","transformer"]]
        for name,scale in pairs:
            for backend in ["cuda","vulkan"]:
                prefix=out/f"{mode}-{name}-{scale}-{backend}";output=prefix.with_suffix(".json")
                if output.exists():
                    prior=json.loads(output.read_text())
                    if prior.get("status")=="passed" and (mode!="nsys" or prior.get("mode")=="normal_trace" and prefix.with_suffix(".sqlite").exists()):continue
                current=env.copy();current["TORCH_VULKAN_DEFER_SUBMISSIONS"]="0" if mode=="gpu" else "1"
                if mode=="gpu" and backend=="vulkan":current["TORCH_VULKAN_PROFILE_TIMESTAMPS"]="1"
                else:current.pop("TORCH_VULKAN_PROFILE_TIMESTAMPS",None)
                script="benchmarks/trace_network_step.py" if mode=="nsys" else "benchmarks/benchmark_network_matrix.py"
                cmd=[str(root/(".venv-cuda" if backend=="cuda" else ".venv")/"bin/python"),script,"--case",name,"--scale",scale,"--backend",backend,"--warmup","8","--repeats","3" if mode in ["gpu","nsys"] else "10","--out",str(output)]
                if mode!="nsys":cmd+=["--mode",mode,"--skip-parity"]
                if mode=="nsys":cmd=["nsys","profile","--trace="+("cuda,nvtx" if backend=="cuda" else "vulkan,nvtx"),"--sample=none","--cpuctxsw=none","--force-overwrite=true","--output",str(prefix)]+(["--vulkan-gpu-workload=individual"] if backend=="vulkan" else [])+cmd
                start=time.time()
                with prefix.with_suffix(".log").open("w") as f:
                    try:r=subprocess.run(cmd,cwd=root,env=current,stdout=f,stderr=subprocess.STDOUT,timeout=180);code=r.returncode
                    except subprocess.TimeoutExpired:code="timeout"
                record=dict(mode=mode,case=name,scale=scale,backend=backend,returncode=code,started_at=start,finished_at=time.time(),seconds=time.time()-start,command=cmd)
                if mode=="nsys" and code==0:
                    r=subprocess.run(["nsys","export","--type","sqlite","--force-overwrite=true","--output",str(prefix.with_suffix(".sqlite")),str(prefix.with_suffix(".nsys-rep"))],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
                    record["export_returncode"]=r.returncode;prefix.with_suffix(".export.log").write_text(r.stdout)
                records.append(record);(out/"index.json").write_text(json.dumps(records,indent=2)+"\n");print(json.dumps(record),flush=True)
    for name,scale in [(n,"medium") for n in ["mlp","unet","resnet18","mobilenet_v3","transformer"]]+[(n,"reference") for n in ["unet","transformer"]]:
        cuda=out/f"nsys-{name}-{scale}-cuda.sqlite";vk=out/f"nsys-{name}-{scale}-vulkan.sqlite"
        if cuda.exists() and vk.exists():
            r=subprocess.run([str(root/".venv/bin/python"),"benchmarks/analyze_network_step_traces.py","--cuda",str(cuda),"--vulkan",str(vk),"--output",str(out/f"nsys-{name}-{scale}-analysis.json")],cwd=root)
            print(name,scale,"analysis",r.returncode,flush=True)
if __name__=="__main__":main()
