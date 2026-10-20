"""Serial matched network matrix; isolates failures and alternates API order."""
import argparse, json, os, subprocess, time
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rounds",type=int,default=3)
    p.add_argument("--start-round",type=int,default=0,
                   help="Resume from this round without rerunning already completed rounds")
    p.add_argument("--warmup",type=int,default=8)
    p.add_argument("--repeats",type=int,default=25)
    p.add_argument("--prefix",default="wall")
    p.add_argument("--scale",choices=["tiny","small","medium","reference"],default="medium")
    p.add_argument("--cuda-device",default="GPU-30acbf2b-a41a-8b88-e298-fe66c4138b29")
    p.add_argument("--vulkan-device",default="0")
    p.add_argument("--skip-parity",action="store_true")
    p.add_argument("--cases",nargs="+")
    p.add_argument("--out",type=Path,default=Path("benchmarks/results/network-matrix-2026-10-19"))
    a=p.parse_args()
    root=Path(__file__).resolve().parents[1]
    out=a.out.resolve();out.mkdir(parents=True,exist_ok=True)
    features=["pool_classifier","bilinear_autoencoder","bicubic_autoencoder","reduction_regressor","attention_block","volume_autoencoder","spatial_transformer","recurrent_regressor","retrieval_regressor","spectral_filter","solve_regressor","causal_attention_block"]
    zoo=["mlp","unet","resnet18","mobilenet_v3","embedding","transformer"]
    base=os.environ.copy()
    base.update(PYTHONPATH=str(root),CUDA_VISIBLE_DEVICES=a.cuda_device,TORCH_VULKAN_PHYSICAL_DEVICE_INDEX=a.vulkan_device,TORCH_VULKAN_GPU_RESIDENT="1",TORCH_VULKAN_DEFER_SUBMISSIONS="1",TORCH_VULKAN_FORBID_CPU_FALLBACK="1",OMP_NUM_THREADS="4",MKL_NUM_THREADS="4")
    # Diagnostic controls must never leak into primary wall timings.
    base.pop("TORCH_VULKAN_PROFILE_TIMESTAMPS",None)
    index=out/(a.prefix+"-index.json")
    log=json.loads(index.read_text()) if a.start_round and index.exists() else []
    if log and any(r["round"]>=a.start_round for r in log):
        raise ValueError("Existing index already contains the requested start round; refusing duplicate runs")
    for round_id in range(a.start_round,a.rounds):
        for i,name in enumerate(a.cases or zoo+features):
            for backend in (["cuda","vulkan"] if (i+round_id)%2==0 else ["vulkan","cuda"]):
                stem=f"{a.prefix}-r{round_id}-{name}-{backend}"
                cmd=[str(root/(".venv-cuda" if backend=="cuda" else ".venv")/"bin/python"),"benchmarks/benchmark_network_matrix.py","--backend",backend,"--case",name,"--scale",a.scale,"--warmup",str(a.warmup),"--repeats",str(a.repeats),"--out",str(out/(stem+".json"))]
                if a.skip_parity:cmd.append("--skip-parity")
                start=time.time()
                with (out/(stem+".log")).open("w") as f:
                    try:
                        run=subprocess.run(cmd,cwd=root,env=base,stdout=f,stderr=subprocess.STDOUT,timeout=240)
                        code=run.returncode
                    except subprocess.TimeoutExpired:
                        code="timeout"
                record=dict(round=round_id,case=name,backend=backend,returncode=code,seconds=time.time()-start)
                log.append(record)
                (out/(a.prefix+"-index.json")).write_text(json.dumps(log,indent=2)+"\n")
                if (out/(stem+".json")).exists():
                    r=json.loads((out/(stem+".json")).read_text())
                    record.update(status=r["status"],training=r.get("training_ms",{}).get("median"),inference=r.get("inference_ms",{}).get("median"),audit=r.get("training_execution"),error=r.get("error","").splitlines()[-1:])
                print(json.dumps(record),flush=True)

if __name__=="__main__":main()
