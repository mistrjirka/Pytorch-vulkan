"""Serial, rotating before/after/CUDA comparisons using the unchanged network harness."""
import argparse
import hashlib
import importlib.machinery
import json
import os
from pathlib import Path
import subprocess
import time

BASELINE = "183070e5f4bac3350571f3b3bd05cf38572ad924"

def identity(root):
    manifest = json.loads((root/".cmake-build/build-source-manifest.json").read_text())
    files = sorted([root/"CMakeLists.txt"] + [f for folder in ("csrc","torch_vulkan")
        for f in (root/folder).rglob("*") if f.suffix in (".cpp",".h",".comp",".spv",".py",".glsl")])
    hashes = {str(f.relative_to(root)): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
    assert hashes == manifest["backend_source_files"], "Build/runtime source mismatch"
    extension = next(root/"torch_vulkan"/("_C"+suffix)
        for suffix in importlib.machinery.EXTENSION_SUFFIXES
        if (root/"torch_vulkan"/("_C"+suffix)).exists())
    assert hashlib.sha256(extension.read_bytes()).hexdigest() == manifest["extension_sha256"]
    return dict(commit=subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"],text=True).strip(),
                **manifest)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--before",type=Path,default=Path("/workspace/Pytorch-vulkan"))
    p.add_argument("--out",type=Path,required=True)
    p.add_argument("--rounds",type=int,default=3)
    p.add_argument("--warmup",type=int,default=8)
    p.add_argument("--repeats",type=int,default=25)
    p.add_argument("--cases",nargs="+",default=["mlp","unet","resnet18","mobilenet_v3","embedding","transformer"])
    p.add_argument("--reference-cases",nargs="*",default=[])
    p.add_argument("--variants",nargs="+",choices=["before","after","cuda"],default=["before","after","cuda"])
    a=p.parse_args()
    after=Path(__file__).resolve().parents[1];before=a.before.resolve();out=a.out.resolve()
    out.mkdir(parents=True,exist_ok=True)
    roots={"before":before,"after":after,"cuda":after}
    snapshots={name:identity(root) for name,root in [("before",before),("after",after)]}
    assert snapshots["before"]["commit"] == BASELINE
    harness="benchmarks/benchmark_network_matrix.py"
    assert (before/harness).read_bytes() == (after/harness).read_bytes()
    (out/"source-identities.json").write_text(json.dumps(snapshots,indent=2)+"\n")
    env=os.environ.copy()
    # Wall measurements must not inherit diagnostic overrides.
    for key in list(env):
        if key.startswith("TORCH_VULKAN_"):env.pop(key)
    env.update(CUDA_VISIBLE_DEVICES="GPU-30acbf2b-a41a-8b88-e298-fe66c4138b29",
               TORCH_VULKAN_PHYSICAL_DEVICE_INDEX="0",TORCH_VULKAN_GPU_RESIDENT="1",
               TORCH_VULKAN_DEFER_SUBMISSIONS="1",TORCH_VULKAN_FORBID_CPU_FALLBACK="1",
               OMP_NUM_THREADS="4",MKL_NUM_THREADS="4")
    pairs=[(name,"medium") for name in a.cases]+[(name,"reference") for name in a.reference_cases]
    records=[];signatures={}
    for round_id in range(a.rounds):
        for case_index,(case,scale) in enumerate(pairs):
            rotation=(round_id+case_index)%len(a.variants)
            order=a.variants[rotation:]+a.variants[:rotation]
            for variant in order:
                root=roots[variant];backend="cuda" if variant=="cuda" else "vulkan"
                stem=f"r{round_id}-{case}-{scale}-{variant}"
                output=out/(stem+".json")
                python=before/(".venv-cuda" if backend=="cuda" else ".venv")/"bin/python"
                cmd=[str(python),harness,"--backend",backend,"--case",case,"--scale",scale,
                     "--warmup",str(a.warmup),"--repeats",str(a.repeats),"--out",str(output)]
                current=dict(env,PYTHONPATH=str(root));start=time.time()
                with (out/(stem+".log")).open("w") as log:
                    try:code=subprocess.run(cmd,cwd=root,env=current,stdout=log,stderr=subprocess.STDOUT,timeout=300).returncode
                    except subprocess.TimeoutExpired:code="timeout"
                row=json.loads(output.read_text()) if output.exists() else {}
                record=dict(round=round_id,case=case,scale=scale,variant=variant,
                            returncode=code,status=row.get("status","missing"),seconds=time.time()-start,
                            command=cmd,source_identity=variant if backend=="vulkan" else "CUDA")
                if code==0 and row["status"]=="passed":
                    signature={key:row[key] for key in ["input_sha256","weights_sha256","torch_git",
                        "optimizer","dtype","shape","harness_sha256","threads","warmup","repeats","tf32","device"]}
                    if (case,scale) in signatures:assert signatures[(case,scale)]==signature
                    else:signatures[(case,scale)]=signature
                    assert row["parity"]["passed"]
                    if backend=="vulkan":
                        assert row["extension_sha256"]==snapshots[variant]["extension_sha256"]
                        for mode in ["training","inference"]:
                            audit=row[mode+"_execution"]
                            assert audit["cpu_fallback_calls"]==0 and audit["host_read_max_bytes"]<=4,audit
                    record.update(training_ms=row["training_ms"]["median"],inference_ms=row["inference_ms"]["median"])
                records.append(record)
                (out/"index.json").write_text(json.dumps(records,indent=2)+"\n")
                print(json.dumps(record),flush=True)
    for variant in ["before","after"]:assert identity(roots[variant])==snapshots[variant],"Sources changed during measurements"
    assert all(r["returncode"]==0 and r["status"]=="passed" for r in records),"Rejected/failed runs retained in index.json"

if __name__=="__main__":main()
