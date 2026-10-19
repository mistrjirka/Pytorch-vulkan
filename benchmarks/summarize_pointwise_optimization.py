"""Summarize rotating before/after/CUDA runs and preserve their raw measurements."""
import argparse
from collections import defaultdict
import gzip
import json
from pathlib import Path
from statistics import median

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory",type=Path)
    p.add_argument("--rounds",type=int,default=3)
    a=p.parse_args();root=a.directory
    index=json.loads((root/"index.json").read_text())
    groups=defaultdict(lambda:defaultdict(list));raw=[]
    assert len({(r["round"],r["case"],r["scale"],r["variant"]) for r in index})==len(index)
    for r in index:
        assert r["returncode"]==0 and r["status"]=="passed",r
        file=f'r{r["round"]}-{r["case"]}-{r["scale"]}-{r["variant"]}.json'
        row=json.loads((root/file).read_text());assert row["parity"]["passed"]
        groups[(r["case"],r["scale"])][r["variant"]].append(row)
        raw.append(dict(file=file,record=row))
    summary=[]
    for (case,scale),variants in sorted(groups.items()):
        entry=dict(case=case,scale=scale,variants={})
        signatures=[]
        for variant in ["before","after","cuda"]:
            rows=variants[variant];assert len(rows)==a.rounds,(case,scale,variant,len(rows))
            signatures.extend({k:r[k] for k in ["input_sha256","weights_sha256","shape","harness_sha256",
                "torch_git","dtype","optimizer","threads","warmup","repeats","tf32","device"]} for r in rows)
            out=dict(parity_comparisons=[r["parity"]["comparisons"] for r in rows],
                     shape=rows[0]["shape"],parameters=rows[0]["parameters"])
            for mode in ["training","inference"]:
                medians=[r[mode+"_ms"]["median"] for r in rows]
                out[mode+"_ms"]=dict(median=median(medians),min_run_median=min(medians),max_run_median=max(medians),run_medians=medians)
                if variant!="cuda":
                    audits=[r[mode+"_execution"] for r in rows];steps=sum(r["repeats"] for r in rows)
                    assert all(r["cpu_fallback_calls"]==0 and r["host_read_max_bytes"]<=4 for r in audits)
                    out[mode+"_audit"]=dict(cpu_fallback_calls=0,steps=steps,
                        host_read_calls_per_step=sum(r["host_read_calls"] for r in audits)/steps,
                        host_read_bytes_per_step=sum(r["host_read_bytes"] for r in audits)/steps,
                        host_read_max_bytes=max(r["host_read_max_bytes"] for r in audits))
            if variant!="cuda":
                steps=sum(r["repeats"] for r in rows)
                out["training_queue_per_step"]={k:sum(r["training_queue_delta"][k] for r in rows)/steps for k in ["submissions","recorded_ops"]}
                out["training_cache_per_step"]={k:sum(r["training_cache_delta"][k] for r in rows)/steps for k in ["algo_hits","algo_misses","algo_evictions","pipeline_hits","pipeline_misses"]}
            entry["variants"][variant]=out
        assert all(s==signatures[0] for s in signatures)
        b,f,c=[entry["variants"][v]["training_ms"]["median"] for v in ["before","after","cuda"]]
        entry.update(training_speedup=b/f,training_reduction_percent=100*(1-f/b),
                     training_before_over_cuda=b/c,training_after_over_cuda=f/c)
        summary.append(entry)
    result=dict(method="Median of process medians; serial rotating API order, same FP32 workload and physical GPU. Primary wall timing includes standard AdamW; no forced phase synchronization.",
                networks=summary)
    (root/"summary.json").write_text(json.dumps(result,indent=2)+"\n")
    (root/"raw.jsonl.gz").write_bytes(gzip.compress("".join(json.dumps(r,separators=(",",":"))+"\n" for r in raw).encode(),mtime=0))
    for r in summary:
        print(r["case"],r["scale"],*[round(r["variants"][v]["training_ms"]["median"],3) for v in ["before","after","cuda"]],round(r["training_speedup"],3),round(r["training_after_over_cuda"],3))
if __name__=="__main__":main()
