"""Analyze Nsight Systems NVTX-matched CPU and GPU timelines from SQLite exports.

Stage ranges are matched by timestamp and are *already synchronized* by the
profile_matched_nsys.py driver. Vulkan workload ranges overlap (submit parent
and child activities), so we use interval union, not a sum of durations.
"""
from __future__ import annotations
import argparse, collections, json, re, sqlite3, statistics
from pathlib import Path

STAGES = ("zero_grad", "forward", "loss", "backward", "optimizer")


def overlap_ms(intervals, start, end):
    spans = []
    for a, b in intervals:
        a, b = max(a, start), min(b, end)
        if b > a:
            spans.append((a, b))
    spans.sort()
    merged = []
    for a, b in spans:
        if merged and a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(b, merged[-1][1]))
        else:
            merged.append((a, b))
    return sum((b-a) * 1e-6 for a,b in merged)


def count_inside(rows, start, end):
    selected = [r for r in rows if start <= r[0] < end]
    return len(selected), round(sum((min(r[1],end)-r[0])*1e-6 for r in selected),5)


def summarize(db, backend):
    conn=sqlite3.connect(str(db))
    conn.row_factory=sqlite3.Row
    events=conn.execute("SELECT e.start,e.end,coalesce(e.text,s.value) as name FROM NVTX_EVENTS e LEFT JOIN StringIds s ON e.textId=s.id WHERE e.end IS NOT NULL ORDER BY e.start").fetchall()
    runs=[(r["start"],r["end"]) for r in events if r["name"]=="measured_step"]
    stages={s:[] for s in STAGES}
    for start,end in runs:
        d={}
        for r in events:
            if r["name"] in STAGES and start <= r["start"] and r["end"] <= end:
                d[r["name"]]=(r["start"],r["end"])
        for name in STAGES:
            if name in d:stages[name].append(d[name])

    if backend=="cuda":
        times_gpu=[tuple(r) for r in conn.execute("SELECT start,end FROM CUPTI_ACTIVITY_KIND_KERNEL")]
        memcopy=[tuple(r) for r in conn.execute("SELECT start,end FROM CUPTI_ACTIVITY_KIND_MEMCPY")]
        api=[(r["start"],r["end"],r["name"]) for r in conn.execute("SELECT a.start,a.end,s.value as name FROM CUPTI_ACTIVITY_KIND_RUNTIME a LEFT JOIN StringIds s ON s.id=a.nameId")]
        opnames=("cudaLaunchKernel","cudaLaunchKernelExC","cudaLaunchKernelExC_ptsz","cudaStreamSynchronize","cudaDeviceSynchronize")
    else:
        has_workload=conn.execute("SELECT 1 FROM sqlite_master WHERE name='VULKAN_WORKLOAD'").fetchone()
        times_gpu=[tuple(r) for r in conn.execute("SELECT start,end FROM VULKAN_WORKLOAD")] if has_workload else []
        memcopy=[]
        api=[(r["start"],r["end"],r["name"]) for r in conn.execute("SELECT a.start,a.end,s.value as name FROM VULKAN_API a LEFT JOIN StringIds s ON s.id=a.nameId")]
        opnames=("vkQueueSubmit","vkWaitForFences","vkAllocateMemory","vkFreeMemory",
                 "vkCreateComputePipelines","vkCreateFence","vkDestroyFence","vkAllocateDescriptorSets")
    api=[(a,b,re.sub(r"_v[0-9]+$", "", name or "")) for a,b,name in api]
    # Some Nsight versions capture Vulkan GPU workloads only during startup.
    # Missing intervals during measured NVTX ranges mean unknown GPU time,
    # NOT that the Vulkan device was idle.
    gpu_trace_observed=bool(runs) and any(
        a < end and b > start
        for a,b in times_gpu for start,end in runs)
    # A few Vulkan intervals do not prove complete coverage: these traces can
    # omit entire submitted phases. Preserve observed intervals as lower bounds.
    gpu_trace_complete=gpu_trace_observed and backend=="cuda"
    res={}
    for name, intervals in stages.items():
        entries=[]
        for start,end in intervals:
            d={"wall_ms":(end-start)*1e-6,
                "observed_gpu_union_ms":overlap_ms(times_gpu,start,end),
                "gpu_union_ms":overlap_ms(times_gpu,start,end) if gpu_trace_complete else None,
               "gpu_events":sum(not (t1<=start or t0>=end) for t0,t1 in times_gpu) if gpu_trace_complete else None,
               "cuda_memcpy_union_ms":overlap_ms(memcopy,start,end)}
            for op in opnames:
                selected=[(a,b) for a,b,tag in api if tag==op]
                n,total=count_inside(selected,start,end)
                d[op+"_calls"]=n
                d[op+"_api_ms"]=total
            entries.append(d)
        if entries:
            keys=entries[0].keys()
            res[name]={"median_"+k:(round(statistics.median(d[k] for d in entries),4)
                       if entries[0][k] is not None else None) for k in keys}
            res[name]["per_run"]=entries
    return {"database":db.name,"backend":backend,"steps":len(runs),
            "gpu_trace_complete":gpu_trace_complete,
            "gpu_trace_status":("observed" if backend=="cuda" and gpu_trace_observed else
                                "partial_unverified" if gpu_trace_observed else "missing"),
            "gpu_trace_note":"Vulkan observed intervals are lower bounds; missing activity is not idle time.",
            "results":res}


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--cuda",type=Path,required=True)
    p.add_argument("--vulkan",type=Path,required=True)
    p.add_argument("--output",type=Path)
    args=p.parse_args()
    report={backend:summarize(getattr(args,backend),backend) for backend in ["cuda","vulkan"]}
    for backend,v in report.items():
        print("\n",backend.upper(), "steps",v["steps"])
        for stage,details in v["results"].items():
            busy=details["median_gpu_union_ms"]
            if busy is None:
                print(f"{stage:12s} wall={details['median_wall_ms']:8.3f}ms gpu_busy=N/A (incomplete GPU workload trace)")
            else:
                print(f"{stage:12s} wall={details['median_wall_ms']:8.3f}ms gpu_busy={busy:8.3f}ms gpu_events={details['median_gpu_events']:6.1f} non-GPU gap={details['median_wall_ms']-busy:8.3f}ms")
            extra={k.removeprefix("median_").removesuffix("_api_ms"):details[k]
                   for k in details if k.startswith("median_") and k.endswith("_api_ms")
                   and details[k] is not None and details[k]>0.001}
            if extra:print("  Vulkan/CUDA API ms:",extra)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2)+"\n")


if __name__=="__main__":main()
