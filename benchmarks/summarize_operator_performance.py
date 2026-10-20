"""Cross-backend median-of-run-median summary for the 39-case operator performance sweep."""
import argparse,collections,gzip,json,statistics
from pathlib import Path

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('directory',type=Path)
 p.add_argument('--expected-rounds',type=int,default=3)
 p.add_argument('--pack',action='store_true')
 p.add_argument('--allow-incomplete',action='store_true')
 a=p.parse_args();base=a.directory
 records=[]
 for f in sorted(base.glob('ops-r*-*.json')):
  data=json.loads(f.read_text())
  records.append((f,data))
 if not records:raise RuntimeError(f'No timed operator results in {base}')
 grouped=collections.defaultdict(lambda:collections.defaultdict(list))
 for f,data in records:
  if not data['timed_performance'] or data['perf_repeats']<=0:
   raise RuntimeError(f'Not a performance record: {f}')
  for name,row in data['cases'].items():
   grouped[name][data['backend']].append((data,f,row))
 report=dict(metadata=dict(source_commit='1c2d9360376619d90740a1ff533eec98f07d4fcd',
   suite='PyTorch 39-case operator audit with optional matched latency',
   comparison='Same RTX 2080 Ti, CUDA 13.0 and Vulkan, FP32 plus dtype coverage',
   timings='Includes forward+backward for differentiable cases; forward only for others; GPU synchronized after each sample',
   process_runs=len(records),warmup=records[0][1]['perf_warmup'],
   steps=records[0][1]['perf_repeats']),cases=[],paired_ok=0,
   blocked=[],failing=[],cpu_fallback_calls=0)
 torch_git=None
 for name,api in sorted(grouped.items()):
  row=dict(case=name,backends={})
  for backend in ('cuda','vulkan'):
   xs=api.get(backend,[])
   if not a.allow_incomplete and len(xs)!=a.expected_rounds:
    raise RuntimeError((name,backend,len(xs),a.expected_rounds))
   passed=[];errors=[]
   for data,f,entry in xs:
    if torch_git is None:torch_git=data.get('torch_git')
    elif data.get('torch_git')!=torch_git:
     raise RuntimeError((f,'Torch revision differs'))
    if entry.get('status')=='ok' and 'performance_ms' in entry:
     if backend=='vulkan':
      audit=entry['performance_execution']
      if audit['cpu_fallback_calls']:
       raise RuntimeError(f'Unexpected CPU numerical fallback: {name} {f}')
      report['cpu_fallback_calls']+=audit['cpu_fallback_calls']
     passed.append((data,entry))
    else:errors.append(dict(file=f.name,status=entry.get('status'),error_type=entry.get('error_type'),
                            error=str(entry.get('error',''))[:280]))
   d=dict(runs=len(xs),passed=len(passed),failures=errors)
   if passed:
    med=[p['performance_ms']['median'] for _,p in passed]
    d['median_ms']=statistics.median(med)
    d['run_medians_ms']=med
    d['includes_backward']=passed[0][1]['performance_ms']['includes_backward']
    d['samples_per_run']=passed[0][1]['performance_ms']['repeats']
   row['backends'][backend]=d
  if all(row['backends'][backend]['passed']==a.expected_rounds for backend in ('cuda','vulkan')):
   row['vulkan_over_cuda']=row['backends']['vulkan']['median_ms']/row['backends']['cuda']['median_ms']
   report['paired_ok']+=1
  else:
   report['blocked'].append(dict(case=name,cuda=row['backends']['cuda']['failures'],
                                 vulkan=row['backends']['vulkan']['failures']))
  report['cases'].append(row)
 report['metadata']['torch_git']=torch_git
 (base/'summary.json').write_text(json.dumps(report,indent=2)+'\n')
 if a.pack:
  with gzip.open(base/'raw-operator-runs.jsonl.gz','wt') as out:
   for f,data in records:out.write(json.dumps(dict(file=f.name,record=data),separators=(',',':'))+'\n')
 print('PAIRED_COMPLETE',report['paired_ok'],'of',len(report['cases']))
 for x in sorted((x for x in report['cases'] if 'vulkan_over_cuda' in x),key=lambda x:x['vulkan_over_cuda'],reverse=True):
  print(f"{x['case']:24} {x['vulkan_over_cuda']:7.2f}x  CUDA {x['backends']['cuda']['median_ms']:.4f} ms, Vulkan {x['backends']['vulkan']['median_ms']:.4f} ms")
 print('BLOCKED',len(report['blocked']))
 for x in report['blocked']:print('MISSING',x['case'],{'CUDA':x['cuda'][0]['error_type'] if x['cuda'] else None,'Vulkan':x['vulkan'][0]['error_type'] if x['vulkan'] else None})
if __name__=='__main__':main()
