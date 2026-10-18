"""Validate and summarize the matched constant-buffer experiment captures."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import sqlite3
import statistics


def stage_events(database, backend):
    connection = sqlite3.connect(database)
    ranges = connection.execute(
        'select n.start,n.end,coalesce(n.text,s.value) from NVTX_EVENTS n '
        'left join StringIds s on n.textId=s.id where n.end is not null').fetchall()
    steps = [(start, end) for start, end, name in ranges if name == 'measured_step']
    assert steps
    table, column = (('CUPTI_ACTIVITY_KIND_KERNEL', 'demangledName')
                     if backend == 'cuda' else ('VULKAN_WORKLOAD', 'nameId'))
    result = {}
    for stage in ['zero_grad', 'forward', 'loss', 'backward', 'optimizer']:
        intervals = [(start, end) for start, end, name in ranges if name == stage
                     and any(start >= s and end <= e for s, e in steps)]
        assert len(intervals) == len(steps)
        events = defaultdict(lambda: dict(calls_per_step=0, summed_ms_per_step=0))
        for start, end in intervals:
            rows = connection.execute(
                f'select s.value,count(*),sum(a.end-a.start) from {table} a '
                f'join StringIds s on a.{column}=s.id where a.start>=? and a.end<=? '
                'group by s.value', (start, end))
            for name, count, duration in rows:
                events[name]['calls_per_step'] += count / len(steps)
                events[name]['summed_ms_per_step'] += duration / 1e6 / len(steps)
        result[stage] = dict(
            mean_wall_ms=statistics.mean((end-start)/1e6 for start,end in intervals),
            recorded_events_per_step=sum(v['calls_per_step'] for v in events.values()),
            summed_event_ms_per_step=sum(v['summed_ms_per_step'] for v in events.values()),
            events=dict(sorted(events.items(), key=lambda p:p[1]['summed_ms_per_step'], reverse=True)))
    connection.close()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    parser.add_argument('--nsys-directory', type=Path, required=True)
    args = parser.parse_args()
    def read(name): return json.loads((args.directory/name).read_text())
    manifest = read('manifest.json')
    result = dict(models={}, vulkan_api={}, diagnostic_stages={}, notes=[
        'Latency: pooled 30 synchronized samples, two runs of 15 after 8 warmups.',
        'Nsight stages synchronize separately and are not normal step latency.',
        'Summed CUDA kernel durations are not GPU utilization or elapsed critical-path time.',
        'No Vulkan workload events were captured inside measured stages; zero events does not mean zero GPU work.'])
    for model in ['unet','style_net','mlp']:
        row = dict(wall={}, parity={})
        for variant in ['before','after','cuda']:
            captures = [read(f'{model}-{variant}{i}.json') for i in [1,2]]
            if variant != 'cuda':
                expected = manifest['baseline_extension_sha256' if variant == 'before' else 'backend_extension_sha256']
                assert all(d['backend_extension_sha256'] == expected for d in captures)
            row['wall'][variant] = {}
            for mode in ['inference','training']:
                samples = [v for d in captures for v in d['modes'][mode]['samples_ms']]
                row['wall'][variant][mode] = dict(pooled_median_ms=statistics.median(samples),
                    run_medians_ms=[d['modes'][mode]['median_ms'] for d in captures],
                    min_ms=min(samples), max_ms=max(samples))
        for reference in ['before','cuda']:
            d = read(f'{model}-parity-{reference}.json')
            assert d['passed']
            row['parity'][reference] = dict(passed=d['passed'], gradient_count=d['gradient_count'],
                output=d['output'], loss=d['loss'],
                max_gradient_abs=max(v['max_abs'] for v in d['gradients'].values()))
        result['models'][model] = row
    for variant in ['before','after']:
        d = read(f'style_net-{variant}-api.json')
        total = defaultdict(lambda:dict(calls=0,ms=0))
        for stage in d['stages'].values():
            for name,values in stage['api_per_step'].items():
                for key,value in values.items(): total[name][key] += value
        result['vulkan_api'][variant] = dict(sorted(total.items(),key=lambda p:p[1]['ms'],reverse=True))
    for variant in ['after','cuda']:
        result['diagnostic_stages'][variant] = stage_events(
            args.nsys_directory/f'style_net-{variant}.sqlite', 'cuda' if variant == 'cuda' else 'vulkan')
    (args.directory/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    for model,row in result['models'].items():
        print(model,{tag:values['training']['pooled_median_ms'] for tag,values in row['wall'].items()})


if __name__ == '__main__':
    main()
