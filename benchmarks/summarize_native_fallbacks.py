"""Summarize matched ABBA latency and operator/copy attribution captures."""
import argparse
import json
from pathlib import Path
import statistics


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('directory', type=Path)
    a = p.parse_args()
    result = dict(models={}, notes=[
        'Latency: two runs of 15 measured samples after 8 warmups; pooled median.',
        'Profile CPU intervals overlap and include instrumentation overhead.',
        'copy_ counts may represent GPU copies; _to_copy is also used for CPU scalar handling.',
        'NLL still validates class indices on the host; GPU-written labels require synchronization.'])
    manifest = json.loads((a.directory / 'manifest.json').read_text())
    for model in ['unet', 'style_net', 'mlp']:
        row = dict(wall={}, profiles={}, parity={})
        for variant in ['before', 'after', 'cuda']:
            captures = [json.loads((a.directory / f'{model}-{variant}{i}.json').read_text()) for i in [1, 2]]
            if variant != 'cuda':
                expected = manifest['baseline_extension_sha256' if variant == 'before' else 'backend_extension_sha256']
                assert all(d['backend_extension_sha256'] == expected for d in captures)
            row['wall'][variant] = {}
            for mode in ['inference', 'training']:
                samples = [v for d in captures for v in d['modes'][mode]['samples_ms']]
                row['wall'][variant][mode] = dict(pooled_median_ms=statistics.median(samples),
                    run_medians_ms=[d['modes'][mode]['median_ms'] for d in captures],
                    min_ms=min(samples), max_ms=max(samples))
        for backend in ['vulkan', 'cuda']:
            d = json.loads((a.directory / f'{model}-{backend}-profile.json').read_text())
            if backend == 'vulkan':
                assert d['backend_extension_sha256'] == manifest['backend_extension_sha256']
            totals = {}
            for e in d['operators']:
                t = totals.setdefault(e['name'], dict(calls_per_step=0., self_cpu_ms=0., inclusive_cpu_ms=0.))
                for k in t:
                    t[k] += e[k]
            row['profiles'][backend] = dict(operators=totals, counters=d['counter_delta'],
                device_events=d['device_events'], initial_state_sha256=d['initial_state_sha256'], input_sha256=d['input_sha256'])
        assert row['profiles']['vulkan']['initial_state_sha256'] == row['profiles']['cuda']['initial_state_sha256']
        assert row['profiles']['vulkan']['input_sha256'] == row['profiles']['cuda']['input_sha256']
        for reference in ['before', 'cuda']:
            d = json.loads((a.directory / f'{model}-parity-{reference}.json').read_text())
            assert d['passed']
            row['parity'][reference] = dict(passed=d['passed'], gradient_count=d['gradient_count'],
                output=d['output'], loss=d['loss'],
                max_gradient_abs=max(v['max_abs'] for v in d['gradients'].values()),
                max_gradient_relative_l2=max(v['relative_l2'] for v in d['gradients'].values()))
        result['models'][model] = row
    (a.directory / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    for model, row in result['models'].items():
        print(model, {k: v['training']['pooled_median_ms'] for k, v in row['wall'].items()})


if __name__ == '__main__':
    main()
