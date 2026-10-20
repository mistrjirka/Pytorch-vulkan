"""Compare horizontal AdamW fusion with the exact legacy per-tensor Vulkan path.

Same PyTorch model and AMSGrad math in both configurations. The only A/B
difference is TORCH_VULKAN_ADAMW_DISABLE_MULTITENSOR. RTX 2080 Ti only.
"""
import argparse,json,os,statistics,subprocess,time
from pathlib import Path

p=argparse.ArgumentParser()
p.add_argument('--out',type=Path,required=True)
p.add_argument('--cases',nargs='+',default=['mlp','unet','transformer'])
p.add_argument('--rounds',type=int,default=2)
p.add_argument('--warmup',type=int,default=8)
p.add_argument('--repeats',type=int,default=20)
a=p.parse_args()
root=Path(__file__).resolve().parents[1]
out=a.out.resolve()
out.mkdir(parents=True,exist_ok=True)
env=os.environ.copy()
env.update(CUDA_VISIBLE_DEVICES='GPU-30acbf2b-a41a-8b88-e298-fe66c4138b29',
 TORCH_VULKAN_PHYSICAL_DEVICE_INDEX='0',TORCH_VULKAN_GPU_RESIDENT='1',
 TORCH_VULKAN_DEFER_SUBMISSIONS='1',TORCH_VULKAN_FORBID_CPU_FALLBACK='1',
 OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
records=[]
for round_id in range(a.rounds):
 for case in a.cases:
  for flag in (('1','0') if round_id%2==0 else ('0','1')):
   env['TORCH_VULKAN_ADAMW_DISABLE_MULTITENSOR']=flag
   # All other execution configuration and model weights are identical.
   dest=out/f'{case}-{flag}-r{round_id}.json'
   cmd=[str(root/'.venv/bin/python'),'benchmarks/benchmark_network_matrix.py',
        '--backend','vulkan','--scale','medium','--case',case,
        '--optimizer-implementation','vulkan_fused_amsgrad',
        '--warmup',str(a.warmup),'--repeats',str(a.repeats),
        '--out',str(dest)]
   start=time.monotonic()
   with dest.with_suffix('.log').open('w') as log:
    result=subprocess.run(cmd,cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=200)
   if result.returncode or not dest.exists():
    raise RuntimeError(f'Benchmark failed {case} {flag} {round_id}: {dest.with_suffix(".log").read_text()[-1200:]}')
   data=json.loads(dest.read_text())
   assert data['status']=='passed' and data['parity']['passed'],(case,flag,round_id)
   assert data['training_execution']['cpu_fallback_calls']==0,(case,flag,round_id)
   record={'case':case,'round':round_id,'per_tensor':flag=='1',
      'training_median_ms':data['training_ms']['median'],
      'inference_median_ms':data['inference_ms']['median'],
      'recorded_ops':data.get('training_queue_delta_per_step',{}).get('recorded_ops'),
      'elapsed_s':time.monotonic()-start}
   records.append(record)
   (out/'index.json').write_text(json.dumps(records,indent=2)+'\n')
   print('RESULT',record,flush=True)
grouped={}
for case in a.cases:
 rows={}
 for flag in (True,False):
  subset=[r for r in records if r['case']==case and r['per_tensor']==flag]
  assert len(subset)==a.rounds
  rows['per_tensor' if flag else 'multitensor']={
    'process_medians_ms':[r['training_median_ms'] for r in subset],
    'median_ms':statistics.median(r['training_median_ms'] for r in subset),
    'recorded_ops_per_step':[r['recorded_ops'] for r in subset]
  }
 rows['speedup']=rows['per_tensor']['median_ms']/rows['multitensor']['median_ms']
 grouped[case]=rows
summary={'baseline_commit':'9d5c413','optimizer':'torch_vulkan.optim.AdamW AMSGrad True, batched True',
 'GPU':'NVIDIA GeForce RTX 2080 Ti','rounds':a.rounds,'warmup':a.warmup,
 'repeats':a.repeats,'cases':grouped,'all_rounds_passed_parity':True,'cpu_fallback_calls':0,
 'config':'Only horizontal fusion flag changed; other shader math and optimizer configuration unchanged'}
(out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
for case,r in grouped.items():
 print('SUMMARY',case,'per_tensor',round(r['per_tensor']['median_ms'],3),
       'multitensor',round(r['multitensor']['median_ms'],3),
       'speedup',round(r['speedup'],3),flush=True)
