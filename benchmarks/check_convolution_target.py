#!/usr/bin/env python3
"""Audit complete isolated convolution phases against a matched CUDA run.

The probe itself performs CPU float64 forward/backward parity before timing;
this report enforces identical input hashes and workload geometry. The 1.5x
limit is a goal, not an automatic correctness condition.
"""
import argparse
import json
from pathlib import Path

PHASES = ('forward', 'dinput', 'dweight', 'dbias')
GEOMETRY = ('shape', 'cout', 'kernel', 'stride', 'padding', 'dilation', 'groups',
            'weight_shape', 'output_shape', 'input_sha256')


def analyze(vulkan, cuda, limit):
    if vulkan['backend'] != 'vulkan' or cuda['backend'] != 'cuda':
        raise ValueError('Expected Vulkan and CUDA probe outputs, in that order')
    if vulkan['device'] != cuda['device']:
        raise ValueError('Both runs must use the same physical GPU: ' +
                         str((vulkan['device'], cuda['device'])))
    if vulkan['dtype'] != cuda['dtype'] or vulkan['seed'] != cuda['seed']:
        raise ValueError('Mismatched dtype or input seed')
    vk_cases = {x['name']: x for x in vulkan['cases']}
    cu_cases = {x['name']: x for x in cuda['cases']}
    if vk_cases.keys() != cu_cases.keys():
        raise ValueError('Unmatched case inventory')
    rows = []
    for name in sorted(vk_cases):
        v, c = vk_cases[name], cu_cases[name]
        if any(v[k] != c[k] for k in GEOMETRY):
            raise ValueError('Mismatched geometry/inputs for ' + name)
        for phase in PHASES:
            vm = v['measurements'][phase]['wall']['median_ms']
            cm = c['measurements'][phase]['wall']['median_ms']
            if cm <= 0:
                raise ValueError('Nonpositive CUDA timing in ' + name)
            rows.append(dict(case=name, phase=phase, vulkan_ms=vm, cuda_ms=cm,
                             ratio=vm / cm, within_goal=vm <= cm * limit))
    return dict(gpu=vulkan['device'], dtype=vulkan['dtype'], target_ratio=limit,
                cases=len(vk_cases), phases=len(rows),
                within_target=sum(r['within_goal'] for r in rows),
                worst=sorted(rows, key=lambda r: r['ratio'], reverse=True)[:8],
                rows=rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('vulkan', type=Path)
    parser.add_argument('cuda', type=Path)
    parser.add_argument('--max-ratio', type=float, default=1.5)
    parser.add_argument('--json-out', type=Path)
    parser.add_argument('--require', action='store_true',
                        help='Exit nonzero unless every measured phase meets the goal')
    args = parser.parse_args()
    if args.max_ratio <= 0:
        parser.error('--max-ratio must be positive')
    report = analyze(json.loads(args.vulkan.read_text()),
                     json.loads(args.cuda.read_text()), args.max_ratio)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(report, indent=2) + '\n')
    print(f"{report['gpu']} ({report['dtype']}): "
          f"{report['within_target']}/{report['phases']} phases <= {args.max_ratio:.2f}x CUDA")
    print('Case                      Phase     Vulkan(ms) CUDA(ms) Ratio  Goal')
    for row in report['rows']:
        print(f"{row['case']:<25} {row['phase']:<9} {row['vulkan_ms']:>10.4f} "
              f"{row['cuda_ms']:>8.4f} {row['ratio']:>5.2f}x "
              f"{'yes' if row['within_goal'] else 'NO'}")
    if args.require and report['within_target'] != report['phases']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
