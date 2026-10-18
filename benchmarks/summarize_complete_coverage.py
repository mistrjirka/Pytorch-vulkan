"""Verify completion evidence and summarize matched wall/kernel measurements."""
import collections,hashlib,json,re,statistics,subprocess
from pathlib import Path
root=Path(__file__).resolve().parents[1]
results=root/'benchmarks/results/fp32-completion'
def read(name):return json.loads((results/name).read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
tests=read('test-exit-codes.json')
assert len(tests)==8 and all(v['exit_code']==0 for v in tests.values())
for name,row in tests.items():
    lines=(results/(name+'.log')).read_text().splitlines()
    row['summary']=next(line for line in reversed(lines) if re.match(r'\d+ passed',line))
exclude={'mm_fp16','add_fp16','mm_bf16','add_bf16','mm_fp64','add_fp64','mm_complex64'}
audit=read('vulkan-audit.json');cuda=read('cuda-audit.json')
assert len(audit['cases'])==39 and len(cuda['cases'])==39
assert all(v['status']=='ok' for v in cuda['cases'].values())
assert {k for k,v in audit['cases'].items() if v['status']!='ok'}==exclude
for name,row in audit['cases'].items():
    if name not in exclude:
        assert row['execution']['cpu_fallback_calls']==0 and row['execution']['host_read_max_bytes']<=4,(name,row)
assert audit['special']['rng']['result']['changes_cpu_rng_state'] is False
assert audit['special']['device_seed']['result']['device_manual_seed_all_repeats'] is True
assert all(v['status']=='ok' for k,v in audit['special'].items() if k!='autocast')
wall={b:[read(f'{b}-wall-{i}.json') for i in (1,2)] for b in ('vulkan','cuda')}
gpu={b:read(f'{b}-gpu-1.json') for b in ('vulkan','cuda')}
models={}
for name in wall['vulkan'][0]['models']:
    rows=[r['models'][name] for runs in wall.values() for r in runs]+[r['models'][name] for r in gpu.values()]
    assert len({x['input_sha256'] for x in rows})==len({x['weights_sha256'] for x in rows})==1
    assert all('parity_max_abs' in x for x in rows)
    out={}
    for b in ('vulkan','cuda'):
        samples=[x for r in wall[b] for x in r['models'][name]['training_ms']['samples']]
        out[b+'_wall_ms']={'median':statistics.median(samples),'min':min(samples),'max':max(samples),'run_medians':[r['models'][name]['training_ms']['median'] for r in wall[b]]}
    out['wall_ratio']=out['vulkan_wall_ms']['median']/out['cuda_wall_ms']['median']
    timings=gpu['vulkan']['models'][name]['gpu_timing'];count=gpu['vulkan']['repeats']
    out['vulkan_gpu_ms']=sum(t['shader_ms'] for t in timings.values())/count
    out['vulkan_dispatches']=sum(t['samples'] for t in timings.values())/count
    out['vulkan_host_eval_ms']=sum(t['host_eval_ms'] for t in timings.values())/count
    out['vulkan_top_shaders']=sorted([{'name':k,'gpu_ms':v['shader_ms']/count,'calls':v['samples']/count} for k,v in timings.items()],key=lambda x:-x['gpu_ms'])[:8]
    kernels=collections.defaultdict(float);calls=collections.Counter()
    for event in gpu['cuda']['models'][name]['kernels']:
        assert not event['name'].startswith('Optimizer.step#')
        kernels[event['name']]+=event['duration_us']/1000/gpu['cuda']['repeats'];calls[event['name']]+=1
    out['cuda_gpu_ms']=sum(kernels.values());out['cuda_device_activities']=sum(calls.values())/gpu['cuda']['repeats']
    out['cuda_top_kernels']=sorted([{'name':k,'gpu_ms':v,'calls':calls[k]/gpu['cuda']['repeats']} for k,v in kernels.items()],key=lambda x:-x['gpu_ms'])[:8]
    models[name]=out
assert read('cpu-compiler-isolation.json')['status']=='ok'
assert read('cuda-no-vulkan.json')['status']=='ok'
for name in ('baseline','completion'):
    result=read('cuda-'+name+'-coexist.json')
    assert result['status']=='error' and 'opt_ready_stream && opt_parent_stream' in result['error']

cuda_wheel_log=(results/'vulkan-cuda-wheel.log').read_text()
assert '39 passed' in cuda_wheel_log and 'FAILED' not in cuda_wheel_log
files=[]
for directory in ('csrc','torch_vulkan','tests','scripts'):
    files += [p for p in (root/directory).rglob('*') if p.is_file() and p.suffix in ('.cpp','.h','.comp','.glsl','.spv','.py','.sh')]
files += [root/'CMakeLists.txt']+[root/'benchmarks'/name for name in ('feature_networks.py','profile_complete_coverage.py','probe_complete_audit.py','probe_compiler_isolation.py','probe_backend_coexistence.py','summarize_complete_coverage.py')]
extension=next((root/'torch_vulkan').glob('_C*.so'))
summary={'baseline_commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=root,text=True).strip(),'extension_sha256':sha(extension),'source_sha256':{str(p.relative_to(root)):sha(p) for p in sorted(files)},'tests':tests,'vulkan_with_cuda_wheel':'39 passed','original_audit':{'vulkan_in_scope_passed':32,'cuda_all_passed':39,'deferred_cases':sorted(exclude),'deferred_special':['autocast']},'models':models,'compiler_isolation':{'cpu_after_vulkan':read('cpu-compiler-isolation.json'),'cuda_alone':read('cuda-no-vulkan.json'),'cuda_after_vulkan_baseline':read('cuda-baseline-coexist.json'),'cuda_after_vulkan_completion':read('cuda-completion-coexist.json')}}
assert all(r['extension_sha256']==summary['extension_sha256'] for r in wall['vulkan']+[gpu['vulkan']])
(results/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
for name,row in models.items():print(name,round(row['vulkan_wall_ms']['median'],3),round(row['cuda_wall_ms']['median'],3),round(row['wall_ratio'],2),round(row['vulkan_gpu_ms'],3),round(row['cuda_gpu_ms'],3))
