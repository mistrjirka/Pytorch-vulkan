"""Validate, summarize and package matched network measurements."""
import argparse, collections, gzip, json, math, statistics
from pathlib import Path
from benchmark_network_matrix import ZOO_CASES

def summarize(values):
    ordered=sorted(values)
    return dict(median=statistics.median(values),min=min(values),max=max(values),
                p95=ordered[min(len(ordered)-1,math.ceil(.95*len(ordered))-1)])

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory",type=Path)
    p.add_argument("--allow-incomplete",action="store_true")
    p.add_argument("--pack",action="store_true")
    p.add_argument("--allow-rejected",action="store_true",help="Retain parity-rejected cases as N/A; never calculate their speed ratio")
    a=p.parse_args();base=a.directory
    groups=collections.defaultdict(lambda:collections.defaultdict(list));raw=[]
    for prefix in ["wall","reference"]:
        for f in sorted(base.glob(prefix+"-r*-*.json")):
            row=json.loads(f.read_text());raw.append(dict(file=f.name,record=row))
            group=("feature" if row["case"] not in ZOO_CASES else row["scale"],row["case"])
            groups[group][row["backend"]].append(row)
    result=dict(networks=[],failures=[],provenance=dict())
    for (scale,name),apis in sorted(groups.items()):
        entry=dict(case=name,scale=scale,backends={})
        allrows=[r for rows in apis.values() for r in rows]
        for key in ["input_sha256","weights_sha256","torch_git","optimizer","dtype","shape","harness_sha256","threads","warmup","repeats","tf32","device"]:
            values=[json.dumps(r[key],sort_keys=True) for r in allrows]
            assert len(set(values))==1,(scale,name,key,values)
            result["provenance"][key]=allrows[0][key] if key not in ["input_sha256","weights_sha256","shape"] else result["provenance"].get(key,"case-specific; validated across APIs and rounds")
        entry["shape"]=allrows[0]["shape"];entry["parameters"]=allrows[0]["parameters"]
        for api in ["cuda","vulkan"]:
            rows=apis.get(api,[])
            if not a.allow_incomplete:assert len(rows)==3,(scale,name,api,len(rows))
            good=[r for r in rows if r["status"]=="passed"]
            result["failures"] += [dict(case=name,scale=scale,backend=api,error=r.get("error")) for r in rows if r["status"]!="passed"]
            d=dict(runs=len(rows),passed_runs=len(good))
            if good:
                assert all(r["parity"].get("passed") for r in good)
                d["parity_comparisons_per_run"]=[r["parity"]["comparisons"] for r in good]
                for mode in ["training","inference"]:
                    samples=[s for r in good for s in r[mode+"_ms"]["samples"]]
                    medians=[r[mode+"_ms"]["median"] for r in good]
                    d[mode+"_ms"]=dict(median_of_run_medians=statistics.median(medians),run_medians=medians,
                                       samples_summary=summarize(samples))
                    if api=="vulkan":
                        audits=[r[mode+"_execution"] for r in good]
                        assert all(x["cpu_fallback_calls"]==0 for x in audits)
                        assert all(x["host_read_max_bytes"]<=4 for x in audits), (name, mode, audits)
                        steps=sum(r["repeats"] for r in good)
                        d[mode+"_audit"]=dict(cpu_fallback_calls=sum(x["cpu_fallback_calls"] for x in audits),
                                              host_read_calls_per_step=sum(x["host_read_calls"] for x in audits)/steps,
                                              host_read_bytes_per_step=sum(x["host_read_bytes"] for x in audits)/steps,
                                              host_read_max_bytes=max(x["host_read_max_bytes"] for x in audits))
                if api=="vulkan":
                    steps=sum(r["repeats"] for r in good)
                    for key in ["training_queue_delta","training_cache_delta"]:
                        d[key+"_per_step"]={k:sum(r[key][k] for r in good)/steps for k in good[0][key] if k not in ["largest_batch","bound_algorithms","pipeline_entries"]}
            entry["backends"][api]=d
        if all(entry["backends"][api].get("passed_runs") for api in ["cuda","vulkan"]):
            for mode in ["training","inference"]:
                entry[mode+"_vulkan_over_cuda"]=entry["backends"]["vulkan"][mode+"_ms"]["median_of_run_medians"]/entry["backends"]["cuda"][mode+"_ms"]["median_of_run_medians"]
        result["networks"].append(entry)
    if not a.allow_incomplete and not a.allow_rejected:
        assert not result["failures"], result["failures"]
    result["diagnostics"]={}
    diag=base/"diagnostics"
    for f in sorted(diag.glob("*.json")):
        if f.name=="index.json":continue
        r=json.loads(f.read_text())
        if f.name.startswith("gpu-") and r.get("status")=="passed":
            if r["backend"]=="vulkan":
                r["top_shaders_per_step"]=sorted([dict(name=name,**{k:v/r["repeats"] for k,v in stat.items()}) for name,stat in r["gpu_timing_delta"].items()],key=lambda s:s.get("shader_ms",0),reverse=True)
                r["shader_ms_per_step"]=sum(s["shader_ms"] for s in r["top_shaders_per_step"])
            else:
                groupsk=collections.defaultdict(lambda:dict(calls=0,ms=0))
                for e in r["kernels"]:
                    groupsk[e["name"]]["calls"]+=1/r["repeats"]
                    groupsk[e["name"]]["ms"]+=e["duration_us"]/1000/r["repeats"]
                r["top_kernels_per_step"]=sorted([dict(name=name,**v) for name,v in groupsk.items()],key=lambda s:s["ms"],reverse=True)
                r["kernel_ms_per_step"]=sum(k["ms"] for k in r["top_kernels_per_step"])
                r["top_kernels_per_step"]=r["top_kernels_per_step"][:15]
                r["top_operators"]=sorted(r.pop("operators"),key=lambda op:op["self_device_us"],reverse=True)[:15]
                r.pop("kernels")
        result["diagnostics"][f.name]=r
    (base/"summary.json").write_text(json.dumps(result,indent=2)+"\n")
    if a.pack:
        for prefix in ["wall","reference","smoke"]:
            records=[dict(file=f.name,record=json.loads(f.read_text())) for f in sorted(base.glob(prefix+"-r*-*.json"))]
            data="".join(json.dumps(r,separators=(",",":"))+"\n" for r in records).encode()
            (base/(prefix+"-raw.jsonl.gz")).write_bytes(gzip.compress(data,mtime=0))
        records=[dict(file=f.name,record=json.loads(f.read_text())) for f in sorted(diag.glob("*.json"))]
        (base/"diagnostics-raw.jsonl.gz").write_bytes(gzip.compress("".join(json.dumps(r,separators=(",",":"))+"\n" for r in records).encode(),mtime=0))
    for r in result["networks"]:
        if "training_vulkan_over_cuda" not in r:continue
        print(r["scale"],r["case"],"train",*(round(r["backends"][api]["training_ms"]["median_of_run_medians"],3) for api in ["cuda","vulkan"]),"ratio",round(r["training_vulkan_over_cuda"],2),"infer",round(r["inference_vulkan_over_cuda"],2))
    print("failed runs",len(result["failures"]))

if __name__=="__main__":main()
