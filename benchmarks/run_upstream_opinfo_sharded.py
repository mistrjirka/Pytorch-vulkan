"""Run the entire pinned selection in disjoint, unmodified test-case shards.

Each worker uses run_upstream_opinfo.py's existing exact-name filter. This only
changes scheduling: original test bodies, samples, tolerances, skips and raw
outcomes are preserved. Aggregation verifies complete, duplicate-free coverage.
Multiple processes per GPU are for correctness testing, never performance data.
"""
import argparse
import collections
import gzip
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def read_json(path):
    data = path.read_bytes()
    if path.suffix == '.gz': data = gzip.decompress(data)
    return json.loads(data)


def digest(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--inventory', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--devices', type=int, nargs='+', default=[0, 1])
    parser.add_argument('--shards', type=int, default=6)
    parser.add_argument('--timeout', type=int, default=900)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    assert args.shards > 0 and args.devices and args.timeout > 0
    root = Path(__file__).resolve().parents[1]
    args.source = args.source.resolve(); args.out = args.out.resolve()
    original = read_json(args.inventory)
    assert digest(args.source) == original['source_sha256']
    cases = original['selected']; ids = [r['id'] for r in cases]
    assert len(ids) == len(set(ids))
    shards = [[] for _ in range(args.shards)]
    for case in ids:
        shards[int(hashlib.sha256(case.encode()).hexdigest(), 16) % args.shards].append(case)
    assert set().union(*map(set, shards)) == set(ids)
    assert sum(map(len, shards)) == len(ids)
    args.out.mkdir(parents=True, exist_ok=True)
    plan = dict(source_sha256=original['source_sha256'], cases=len(ids),
                coordinator_sha256=digest(Path(__file__)),
                partitions=[dict(shard=i, physical_device=args.devices[i % len(args.devices)],
                                 count=len(part), ids=part) for i, part in enumerate(shards)],
                method='Exact original test IDs; unchanged upstream samples/tolerances/skips',
                per_case_watchdog_seconds=args.timeout,
                performance_measurement=False)
    (args.out/'plan.json').write_text(json.dumps(plan, indent=2)+'\n')
    print(json.dumps({'planned': len(ids), 'shards': [len(s) for s in shards]}), flush=True)
    if args.dry_run: return
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    build_path = root/'.cmake-build/build-source-manifest.json'
    built = read_json(build_path)
    assert all(digest(root/p) == h for p, h in built['backend_source_files'].items())
    jobs = []
    try:
        for i, part in enumerate(shards):
            output = args.out/f'shard-{i}'
            assert not (output/'results.jsonl').exists(), 'Use a fresh output directory'
            cmd = [sys.executable, str(root/'benchmarks/run_upstream_opinfo.py'),
                   '--source', str(args.source), '--out', str(output), '--timeout', str(args.timeout)]
            for case in part: cmd.extend(['--match', case])
            env = dict(os.environ, TORCH_VULKAN_PHYSICAL_DEVICE_INDEX=str(args.devices[i % len(args.devices)]))
            log = (args.out/f'shard-{i}.log').open('w')
            proc = subprocess.Popen(cmd, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
            jobs.append((proc, log, output))
        while any(proc.poll() is None for proc, _, _ in jobs):
            counts = [sum(1 for _ in (out/'results.jsonl').open()) if (out/'results.jsonl').exists() else 0 for _, _, out in jobs]
            print(json.dumps({'completed': sum(counts), 'by_shard': counts}), flush=True)
            time.sleep(10)
        assert all(proc.returncode == 0 for proc, _, _ in jobs), 'A shard process failed'
    finally:
        for proc, log, _ in jobs:
            if proc.poll() is None: proc.terminate()
            log.close()
    rows = []; manifests = []
    for i, (_, _, output) in enumerate(jobs):
        inv = read_json(output/'inventory.json'); manifests.append(inv)
        found = [json.loads(s) for s in (output/'results.jsonl').read_text().splitlines()]
        assert len(found) == len(shards[i]) and {r['id'] for r in found} == set(shards[i])
        assert set(inv['planned_test_ids']) == set(shards[i])
        assert {r['id'] for r in inv['selected']} == set(ids)
        assert inv['backend_commit'] == commit
        assert inv['source_identity'] == 'recorded_after_build'
        assert inv['backend_source_sha256'] == built['backend_source_sha256']
        assert inv['extension_sha256'] == built['extension_sha256']
        for row in found: rows.append(dict(row, audit_shard=i, execution_device=inv['device']))
    assert len(rows) == len(ids) and len({r['id'] for r in rows}) == len(ids)
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip() == commit
    assert all(digest(root/p) == h for p, h in built['backend_source_files'].items())
    (args.out/'results.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in sorted(rows, key=lambda r:r['id'])))
    summary = dict(backend_commit=commit, backend_source_sha256=built['backend_source_sha256'],
                   extension_sha256=built['extension_sha256'], source_sha256=original['source_sha256'],
                   total_cases=len(rows), raw_outcomes=dict(collections.Counter(r['status'] for r in rows)),
                   passed_case_fallback_calls_or_attempts=sum(r['execution'].get('cpu_fallback_calls', 0) for r in rows if r['status']=='passed'),
                   by_device={d:dict(collections.Counter(r['status'] for r in rows if r['execution_device']==d)) for d in sorted({r['execution_device'] for r in rows})},
                   shard_source_identities=[dict(shard=i, device=m['device'], source_identity=m['source_identity'], planned_cases=len(shards[i])) for i,m in enumerate(manifests)],
                   original_selection_count=len(ids), complete_duplicate_free_selection=True,
                   interpretation=['Each original test case executes once across the listed GPUs, not once on each GPU.',
                                   'Skips, xfails, xpasses, errors and unvisited samples are not passes.',
                                   'Concurrent correctness runs provide no performance comparison.'])
    (args.out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps(summary), flush=True)


if __name__ == '__main__': main()
