"""Evaluate optimizer selection without altering the standard benchmark baseline.

Vulkan native AdamW is a separate opt-in implementation. CUDA fused=True
provides matched AMSGrad semantics but a different performance baseline.
"""
import argparse,os,sys,json,time,subprocess,statistics
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--out',type=Path,required=True)
p.add_argument('--rounds',type=int,default=2)
p.add_argument('--warmup',type=int,default=8)
p.add_argument('--repeats',type=int,default=20)
p.add_argument('--cases',nargs='+',default=['mlp','unet','transformer'])
p.add_argument('--scale',choices=['medium','reference'],default='medium')
a=p.parse_args()
root=Path(__file__).resolve().parents[1]
out=a.out.resolve();out.mkdir(parents=True,exist_ok=True)
env=os.environ.copy();env.update(
 CUDA_VISIBLE_DEVICES='GPU-30acbf2b-a41a-8b88-e298-fe66c4138b29',
 TORCH_VULKAN_PHYSICAL_DEVICE_INDEX='0',TORCH_VULKAN_GPU_RESIDENT='1',
 TORCH_VULKAN_DEFER_SUBMISSIONS='1',TORCH_VULKAN_FORBID_CPU_FALLBACK='1',
 OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
matrix=[('vulkan','standard'),('vulkan','vulkan_fused_amsgrad'),
        ('cuda','standard'),('cuda','cuda_fused_amsgrad')]
records=[]
for n in range(a.rounds):
 for case in a.cases:
  for backend,implementation in (matrix if n%2==0 else list(reversed(matrix))):
   base=f'r{n}-{case}-{backend}-{implementation}'
   name=out/f'{base}.json'
   cmd=[str(root/('.venv-cuda' if backend=='cuda' else '.venv')/'bin/python'),
        'benchmarks/benchmark_network_matrix.py','--backend',backend,
        '--case',case,'--scale',a.scale,'--warmup',str(a.warmup),
        '--repeats',str(a.repeats),'--optimizer-implementation',implementation,
        '--out',str(name)]
   start=time.monotonic()
   with (out/f'{base}.log').open('w') as log:
    result=subprocess.run(cmd,env=env,cwd=root,stdout=log,stderr=subprocess.STDOUT,timeout=180)
   record={'case':case,'round':n,'backend':backend,'implementation':implementation,
           'exit_code':result.returncode,'duration_s':time.monotonic()-start}
   if name.exists():
    d=json.loads(name.read_text())
    record['status']=d.get('status')
    record['train_median_ms']=d.get('training_ms',{}).get('median')
    record['parity']=d.get('parity',{}).get('passed')
    record['cpu_fallback_calls']=d.get('training_execution',{}).get('cpu_fallback_calls')
   records.append(record)
   (out/'index.json').write_text(json.dumps(records,indent=2)+'\n')
   print(json.dumps(record),flush=True)
   if result.returncode:sys.exit(result.returncode)
aggregate={}
for case in a.cases:
 for backend,impl in matrix:
  rows=[row for row in records if row['case']==case and row['backend']==backend and row['implementation']==impl]
  assert all(row['parity'] and row['status']=='passed' and row['exit_code']==0 for row in rows)
  if backend=='vulkan':assert all(row['cpu_fallback_calls']==0 for row in rows)
  aggregate[f'{case}/{backend}/{impl}']={
   'medians_ms':[r['train_median_ms'] for r in rows],
   'median_of_runs_ms':statistics.median(r['train_median_ms'] for r in rows)}
(out/'summary.json').write_text(json.dumps({'cases':a.cases,'scale':a.scale,'rounds':a.rounds,
 'warmup':a.warmup,'repeats':a.repeats,'optimizer':'AdamW(lr=1e-4,amsgrad=True)',
 'device':'RTX 2080 Ti','source_commit':'3783960 plus uncommitted AMSGrad changes',
 'networks':aggregate},indent=2)+'\n')
for k,v in aggregate.items():print('RESULT',k,round(v['median_of_runs_ms'],4),flush=True)
