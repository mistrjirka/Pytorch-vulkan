"""Summarize raw upstream outcomes without treating skips or blockers as passes."""
import argparse,collections,gzip,hashlib,json,re
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__);p.add_argument('directory',type=Path);p.add_argument('--device',default='rtx',help='Run subdirectory, e.g. rtx or v100');a=p.parse_args();root=a.directory

def read_rows(path):
    if path.exists():text=path.read_text()
    else:
        with gzip.open(str(path)+'.gz','rt') as f:text=f.read()
    return [json.loads(s) for s in text.splitlines() if s]

rows=read_rows(root/a.device/'results.jsonl')
ip=root/a.device/'inventory.json'
if ip.exists():inv=json.loads(ip.read_text())
else:
    with gzip.open(str(ip)+'.gz','rt') as f:inv=json.load(f)
assert len({x['id'] for x in rows})==len(rows)
assert {x['id'] for x in rows}=={x['id'] for x in inv['selected']}
retry=read_rows(root/'timeout-recheck/results.jsonl') if (root/'timeout-recheck/results.jsonl').exists() or (root/'timeout-recheck/results.jsonl.gz').exists() else []

def classify(r):
    text='\n'.join(r['details'])
    fallback=re.search(r'CPU fallback forbidden: (aten::[^\n(]+|Vulkan inline::[^\n]+)',text)
    if fallback:return 'forbidden_fallback',fallback.group(1)
    if 'Vulkan integer RNG supports' in text:return 'rng_dtype_rejection',None
    if 'Mismatched elements:' in text:return 'numerical_comparison',None
    if 'unexpectedly materializes' in text:return 'copy_on_write_contract',None
    if 'does not match' in text:return 'error_message_contract',None
    if 'not raised' in text:return 'missing_expected_exception',None
    if 'CUDA and ROCm GPUs' in text:return 'upstream_cuda_only_feature',None
    return r['status'],None

outcomes=collections.Counter(r['status'] for r in rows)
resolved={r['id']:r for r in rows}
for r in retry:
    assert r['id'] in resolved and resolved[r['id']]['status']=='timeout'
    resolved[r['id']]=r
resolved_outcomes=collections.Counter(r['status'] for r in resolved.values())
fallbacks=collections.Counter();categories=collections.Counter();by_op=collections.defaultdict(collections.Counter)
case_index=[]
for r in rows:
    cat,op=classify(r)
    if r['status'] not in ('passed','skipped','expected_failure','unexpected_success'):
        categories[cat]+=1
        if op:fallbacks[op]+=1
    by_op[r['operator']][r['status']]+=1
    case_index.append({k:r[k] for k in ('id','family','operator','dtype','status','seconds')}|{'classification':cat,'first_fallback':op})
summary={
 'backend_commit':inv['backend_commit'],'backend_source_sha256':inv.get('backend_source_sha256'),'source_identity':inv.get('source_identity'),'published_source_identity':json.loads((root/'source-identity.json').read_text()) if (root/'source-identity.json').exists() else None,'torch':inv['torch'],'torch_git':inv['torch_git'],
 'source_sha256':inv['source_sha256'],'extension_sha256':inv['extension_sha256'],'device':inv['device'],
 'opinfo_registry_entries':inv['opinfo_entries'],'selected_cases':len(rows),
 'generated_cases_outside_scope':len(inv['excluded']),'excluded_reasons':dict(collections.Counter(x['reason'] for x in inv['excluded'])),
 'raw_outcomes':dict(outcomes),'resolved_outcomes':dict(resolved_outcomes),'failures_by_category':dict(categories),'first_missing_dispatches':dict(fallbacks.most_common()),
 'by_family':{f:dict(collections.Counter(r['status'] for r in rows if r['family']==f)) for f in sorted({r['family'] for r in rows})},
 'by_dtype':{d:dict(collections.Counter(r['status'] for r in rows if r['dtype']==d)) for d in sorted({r['dtype'] for r in rows})},
 'by_operator_variant':{k:dict(v) for k,v in sorted(by_op.items())},
 'timeout_recheck':retry,
 'independent_reproductions':{p.stem:json.loads(p.read_text()) for p in root.glob('*-reproductions.json')},
 'interpretation':[
  'Counts are generated test cases, not distinct supported operators or sample inputs.',
  'Each upstream case can stop on its first failing sample; unvisited samples are not passes.',
  'Upstream skips/xfails are retained. No tolerance or sample-input rewrites were made.',
  'FP32-tagged cases can explicitly request other dtypes. Test-defined cases can exercise several dtypes.',
  'Failures may originate in sample generation, comparison helpers, target ops, or composite infrastructure.',
  'Host-read counters include explicit comparison downloads; this suite does not measure numeric GPU residency.',
  'Suite completion exit code is not a claim that all tests passed.',
  'This is selected test_ops coverage, not all PyTorch tests or a complete finite-difference gradcheck suite.',
 ]}
(root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
(root/'cases.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in case_index))
print(json.dumps({k:summary[k] for k in ('selected_cases','raw_outcomes','resolved_outcomes','failures_by_category')},indent=2))
print('Top first missing dispatches:',fallbacks.most_common(12))
