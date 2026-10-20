"""Run the 39-case PyTorch operator audit as a synchronized performance suite.

Preserves the original correctness gate, runs each backend in its own Python
process, alternates API order across repetitions, and retains all failures.
RTX 2080 Ti GPU UUID is explicit; the RTX 3090 is not used.
"""
import argparse,json,os,subprocess,time
from pathlib import Path

def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--rounds',type=int,default=3)
 p.add_argument('--warmup',type=int,default=5)
 p.add_argument('--repeats',type=int,default=25)
 p.add_argument('--out',type=Path,required=True)
 p.add_argument('--cuda-device',default='GPU-30acbf2b-a41a-8b88-e298-fe66c4138b29')
 p.add_argument('--vulkan-device',default='0')
 a=p.parse_args()
 root=Path(__file__).resolve().parents[1]
 dest=a.out.resolve();dest.mkdir(parents=True,exist_ok=True)
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES=a.cuda_device,TORCH_VULKAN_PHYSICAL_DEVICE_INDEX=a.vulkan_device,TORCH_VULKAN_GPU_RESIDENT='1',TORCH_VULKAN_DEFER_SUBMISSIONS='1',TORCH_VULKAN_FORBID_CPU_FALLBACK='1',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4')
 for k in ['TORCH_VULKAN_PROFILE_TIMESTAMPS','TORCH_VULKAN_DISABLE_DIRECT_UNARY_OUT','TORCH_VULKAN_DISABLE_DIRECT_BMM_OUT','TORCH_VULKAN_DISABLE_DIRECT_MM_OUT']:
  env.pop(k,None)
 runs=[]
 for r in range(a.rounds):
  for backend in (('vulkan','cuda') if r%2 else ('cuda','vulkan')):
   dst=dest/f'ops-r{r}-{backend}.json'
   cmd=[str(root/('.venv-cuda' if backend=='cuda' else '.venv')/'bin/python'),'benchmarks/probe_complete_audit.py','--backend',backend,'--out',str(dst),'--skip-special','--perf-warmup',str(a.warmup),'--perf-repeats',str(a.repeats)]
   start=time.perf_counter()
   with (dest/f'ops-r{r}-{backend}.log').open('w') as f:
    result=subprocess.run(cmd,cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=250)
   record=dict(round=r,backend=backend,returncode=result.returncode,duration_s=time.perf_counter()-start)
   if dst.exists():
    data=json.loads(dst.read_text())
    record.update(passed_cases=sum(q.get('status')=='ok' for q in data['cases'].values()),failed_cases=[k for k,q in data['cases'].items() if q.get('status')!='ok'])
   runs.append(record)
   (dest/'index.json').write_text(json.dumps(runs,indent=2)+'\n')
   print(json.dumps(record),flush=True)

if __name__=='__main__':main()
