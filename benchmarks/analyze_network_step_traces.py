"""Analyze full-step Nsight copies and synchronization without phase barriers."""
import argparse,collections,json,re,sqlite3,statistics
from pathlib import Path
from analyze_matched_traces import overlap_ms

def analyze(path,backend):
    c=sqlite3.connect(str(path));c.row_factory=sqlite3.Row
    tables={r[0] for r in c.execute("select name from sqlite_master where type='table'")}
    events=c.execute("select e.start,e.end,coalesce(e.text,s.value) as name from NVTX_EVENTS e left join StringIds s on e.textId=s.id where e.end is not null").fetchall()
    windows=[(r["start"],r["end"]) for r in events if r["name"]=="measured_step"];assert len(windows)==3,(path,len(windows))
    api_table="VULKAN_API" if backend=="vulkan" else "CUPTI_ACTIVITY_KIND_RUNTIME"
    api=[(r["start"],r["end"],re.sub(r"_v[0-9]+$","",r["name"] or "")) for r in c.execute(f"select a.start,a.end,s.value as name from {api_table} a left join StringIds s on a.nameId=s.id")]
    gpu_table="VULKAN_WORKLOAD" if backend=="vulkan" else "CUPTI_ACTIVITY_KIND_KERNEL"
    spans=[tuple(r) for r in c.execute(f"select start,end from {gpu_table}")] if gpu_table in tables else []
    copies=list(c.execute("select * from CUPTI_ACTIVITY_KIND_MEMCPY")) if "CUPTI_ACTIVITY_KIND_MEMCPY" in tables else []
    steps=[];stages=collections.defaultdict(list)
    def calls(start,end):
        out=collections.defaultdict(lambda:dict(calls=0,api_ms=0))
        for b,e,n in api:
            if start<=b<end:out[n]["calls"]+=1;out[n]["api_ms"]+=(min(e,end)-b)/1e6
        return dict(out)
    for start,end in windows:
        cp=collections.defaultdict(lambda:dict(calls=0,bytes=0,gpu_ms=0))
        for r in copies:
            if start<=r["start"]<end:
                kind=str(r["copyKind"] if "copyKind" in r.keys() else r["copyKindId"] if "copyKindId" in r.keys() else "unknown")
                cp[kind]["calls"]+=1;cp[kind]["bytes"]+=r["bytes"];cp[kind]["gpu_ms"]+=(r["end"]-r["start"])/1e6
        steps.append(dict(wall_ms=(end-start)/1e6,gpu_union_ms=overlap_ms(spans,start,end) if backend=="cuda" else None,observed_gpu_lower_bound_ms=overlap_ms(spans,start,end),api=calls(start,end),cuda_copies=dict(cp)))
        for r in events:
            if r["name"] in ["zero_grad","forward","loss","backward","optimizer","final_sync"] and start<=r["start"] and r["end"]<=end:
                stages[r["name"]].append(dict(host_range_ms=(r["end"]-r["start"])/1e6,api=calls(r["start"],r["end"])))
    def aggregate(rows):
        names={n for r in rows for n in r["api"]}
        return dict(api_per_step={n:dict(calls=sum(r["api"].get(n,{}).get("calls",0) for r in rows)/len(rows),api_ms=sum(r["api"].get(n,{}).get("api_ms",0) for r in rows)/len(rows)) for n in names})
    result=dict(backend=backend,database=path.name,steps=len(steps),phase_ranges_measure_host_enqueue_only=True,gpu_trace_complete=backend=="cuda",cuda_memcpy_table_present="CUPTI_ACTIVITY_KIND_MEMCPY" in tables,vulkan_api_warning="Command recording API coverage is partial; absent vkCmdCopyBuffer/vkCmdDispatch records do not establish zero copies/dispatches" if backend=="vulkan" else None,per_step=steps,median_wall_ms=statistics.median(r["wall_ms"] for r in steps),**aggregate(steps))
    result["stages"]={name:dict(mean_host_range_ms=statistics.mean(r["host_range_ms"] for r in rows),**aggregate(rows)) for name,rows in stages.items()}
    result["cuda_copy_kind_labels"]={str(r[0]):r[-1] for r in c.execute("select * from ENUM_CUDA_MEMCPY_OPER")} if "ENUM_CUDA_MEMCPY_OPER" in tables else {}
    kinds={k for row in steps for k in row["cuda_copies"]}
    result["cuda_copies_per_step"]={kind:dict(label=result["cuda_copy_kind_labels"].get(kind,"unknown"),**{key:sum(row["cuda_copies"].get(kind,{}).get(key,0) for row in steps)/len(steps) for key in ["calls","bytes","gpu_ms"]}) for kind in kinds}
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--cuda",type=Path,required=True);p.add_argument("--vulkan",type=Path,required=True);p.add_argument("--output",type=Path,required=True);a=p.parse_args()
    result=dict(method="Normal full-step batching; phase ranges describe CPU enqueue, not GPU phase duration",cuda=analyze(a.cuda,"cuda"),vulkan=analyze(a.vulkan,"vulkan"))
    a.output.write_text(json.dumps(result,indent=2)+"\n")
    print(a.output.name,{api:{k:r["api_per_step"].get(k) for k in (["vkQueueSubmit","vkWaitForFences","vkCmdCopyBuffer","vkAllocateDescriptorSets","vkUpdateDescriptorSets"] if api=="vulkan" else ["cudaLaunchKernel","cudaStreamSynchronize","cudaDeviceSynchronize","cudaMemcpyAsync"])} for api,r in result.items() if api in ["cuda","vulkan"]})
if __name__=="__main__":main()
