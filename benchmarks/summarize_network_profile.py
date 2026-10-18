"""Summarize saved network profiles without adding overlapping durations."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics


def copy_attribution(path, steps):
    trace=json.loads(path.read_text())
    threads=defaultdict(list)
    for e in trace['traceEvents']:
        if e.get('ph')=='X' and e.get('cat')=='cpu_op':
            threads[(e['pid'],e['tid'])].append(e)
    groups=defaultdict(lambda:dict(calls_per_step=0,inclusive_copy_ms_per_step=0))
    for events in threads.values():
        stack=[]
        for e in sorted(events,key=lambda e:(e['ts'],-e.get('dur',0))):
            end=e['ts']+e.get('dur',0)
            while stack and (stack[-1]['ts']+stack[-1]['dur']<=e['ts'] or
                             stack[-1]['ts']+stack[-1]['dur']<end-0.01):
                stack.pop()
            if e['name']=='aten::_to_copy':
                ancestors=[s['name'] for s in stack]
                if 'aten::_to_copy' not in ancestors:
                    candidates=[n for n in ancestors if n not in
                                ['aten::to','aten::_to_cpu','aten::_to_copy','aten::empty_strided']]
                    name=candidates[-1] if candidates else 'unattributed'
                    groups[name]['calls_per_step']+=1/steps
                    groups[name]['inclusive_copy_ms_per_step']+=e['dur']/(1000*steps)
            stack.append(e)
    return dict(sorted(groups.items(),key=lambda kv:kv[1]['inclusive_copy_ms_per_step'],reverse=True))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,required=True)
    p.add_argument('--trace-dir',type=Path,default=Path('/tmp'))
    args=p.parse_args()
    def read(n):return json.loads((args.results/(n+'.json')).read_text())
    result=dict(models={},activation={},notes=[
        'Wall medians pool 30 samples across two fresh processes; per-run medians expose variation.',
        'CPU self and inclusive operator times are instrumented and can include waits.',
        'API times are nested within stage times; do not add them to CPU profile durations.',
        'Vulkan GPU-workload coverage is incomplete; no GPU utilization or CPU-only residual is inferred.'])
    for model in ['unet','style_net','mlp']:
        row=dict(wall={},operators={},copy_attribution={})
        profs={b:read(model+'_'+b+'_operators') for b in ['vulkan','cuda']}
        for key in ['initial_state_sha256','input_sha256']:
            assert profs['vulkan'][key]==profs['cuda'][key],(model,key)
        row['matched_state_and_inputs']=True
        row['initial_losses']={b:d['first_loss'] for b,d in profs.items()}
        for b in ['vulkan','cuda']:
            data=[read(model+'_'+b+'_wall_'+str(i)) for i in [1,2]]
            row['wall'][b]={}
            for mode in ['inference','training']:
                samples=[t for d in data for t in d['modes'][mode]['samples_ms']]
                row['wall'][b][mode]=dict(pooled_median_ms=statistics.median(samples),
                    run_medians_ms=[d['modes'][mode]['median_ms'] for d in data],
                    min_ms=min(samples),max_ms=max(samples))
            groups=defaultdict(lambda:dict(self_cpu_ms=0,inclusive_cpu_ms=0,calls_per_step=0))
            for e in profs[b]['operators']:
                for key in groups[e['name']]:groups[e['name']][key]+=e[key]
            row['operators'][b]=dict(sorted(groups.items(),key=lambda kv:kv[1]['self_cpu_ms'],reverse=True))
        row['copy_attribution']=copy_attribution(args.trace_dir/('network_'+model+'_vulkan.trace.json'),profs['vulkan']['profiled_steps'])
        row['cuda_device_ms_sum']=sum(e['gpu_ms_per_step'] for e in profs['cuda']['device_events'])
        row['cuda_top_device_events']=profs['cuda']['device_events'][:12]
        row['vulkan_deferred_per_step']={k:v/profs['vulkan']['profiled_steps'] for k,v in profs['vulkan']['counter_delta']['queue'].items() if k!='largest_batch'}
        api=read(model+'_vulkan_api');row['vulkan_api_per_step']={}
        for stage,d in api['stages'].items():
            for name,v in d['api_per_step'].items():
                r=row['vulkan_api_per_step'].setdefault(name,dict(calls=0,ms=0))
                for k in r:r[k]+=v[k]
        result['models'][model]=row
    for variant in ['stock','composite']:
        data=[read('activation_'+variant+'_'+str(i)) for i in [1,2]]
        samples=[t for d in data for t in d['samples_ms']]
        result['activation'][variant]=dict(pooled_median_ms=statistics.median(samples),
            run_medians_ms=[d['median_ms'] for d in data],queue_per_step=data[0]['queue_per_step'])
    result['activation']['parity']=read('activation_parity')
    f=args.results/'style_net_composite_operators.json'
    if f.exists():
        d=json.loads(f.read_text())
        result['activation']['composite_copy_attribution']=copy_attribution(args.trace_dir/'network_style_net_composite.trace.json',d['profiled_steps'])
        result['activation']['composite_top_self_cpu']=d['operators'][:15]
    result['source_json_sha256']={f.name:hashlib.sha256(f.read_bytes()).hexdigest()
        for f in sorted(args.results.glob('*.json')) if f.name!='summary.json'}
    (args.results/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    for model,d in result['models'].items():
        print(model,{b:{m:v['run_medians_ms'] for m,v in r.items()} for b,r in d['wall'].items()})
        print('Copies:',d['copy_attribution'])
        print('API:',{k:v for k,v in d['vulkan_api_per_step'].items() if k in ['vkAllocateMemory','vkFreeMemory','vkQueueSubmit','vkWaitForFences','vkCreateComputePipelines']})
    print('Activation:',result['activation'])

if __name__=='__main__':main()
