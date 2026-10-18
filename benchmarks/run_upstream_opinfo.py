"""Run pinned, unmodified PyTorch test_ops.py against the Vulkan backend.

Install expecttest, hypothesis and scipy into the project's torch environment.
Run: python benchmarks/run_upstream_opinfo.py --source /path/to/test_ops.py --out DIR
The parent watchdog isolates crashes/timeouts and resumes the remaining tests.
No operator allowlist, tolerance changes, expected-failure additions, or CPU
fallback are installed. See selection() for the explicit first-pass scope.
"""
from __future__ import annotations
import argparse
import collections
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import traceback
import unittest

TORCH_COMMIT = '5c4886908584029761b579af026dcfb627c84070'
SOURCE_SHA256 = '17ffae61a1654f23d7c2c8f73879bee99c56a80482d061c96975c0fc4720020c'
METHODS = sorted([
    'test_compare_cpu', 'test_noncontiguous_samples', 'test_variant_consistency_eager',
    'test_out_requires_grad_error', 'test_out_warning', 'test_out',
    'test_errors_sparse', 'test_errors', 'test_non_standard_bool_values',
    'test_reduction_ops_reduce', 'test_backward', 'test_operator', 'test_forward_ad',
    'test_cow_input', 'test_view_replay', 'test_0d_tensor_with_python_scalar',
], key=len, reverse=True)

def append(path, row):
    with path.open('a') as f:
        f.write(json.dumps(row, default=str) + '\n')
        f.flush()

def read_rows(path):
    if not path.exists(): return []
    return [json.loads(x) for x in path.read_text().splitlines() if x]

def selection(cls, name):
    family = next((m for m in METHODS if name.startswith(m + '_')), None)
    if not cls.startswith(('TestCommon', 'TestCompositeCompliance', 'TestForwardADWithScalars')):
        return None, 'non-core class (FakeTensor, math-bit dtypes, or metadata)'
    if family is None: return None, 'other test family'
    if not (re.search(r'_vulkan_(float32|int32|int64|bool)$', name) or name.endswith('_vulkan')):
        return None, 'other dtype'
    return family, None

def worker(args):
    import torch
    import torch_vulkan as vk
    from torch.testing._internal.common_device_type import PrivateUse1TestBase
    from torch.testing._internal.common_methods_invocations import op_db, ops_and_refs
    assert torch.version.git_version == TORCH_COMMIT, torch.version.git_version
    torch.set_num_threads(4)
    # Upstream's explicit extension hook. This opens built-in-device gates,
    # retaining the test bodies, OpInfo sample sets, skips and tolerances.
    PrivateUse1TestBase.bypass_device_restrictions = True
    spec = importlib.util.spec_from_file_location('upstream_test_ops', args.source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    op_names = {op.formatted_name: op.full_name for op in ops_and_refs}
    selected, excluded = [], []
    for clsname, cls in vars(module).items():
        if not isinstance(cls, type) or not issubclass(cls, unittest.TestCase): continue
        for name in unittest.defaultTestLoader.getTestCaseNames(cls):
            if '_vulkan' not in name: continue
            family, reason = selection(clsname, name)
            node = clsname + '.' + name
            if reason:
                excluded.append({'id': node, 'reason': reason});continue
            op_token = name[len(family)+1:].split('_vulkan')[0]
            selected.append({'id': node, 'family': family, 'operator': op_names.get(op_token, op_token),
                             'dtype': name.split('_vulkan')[-1].lstrip('_') or 'test-defined'})
    # HEAD alone does not identify an in-progress build. Preserve content hashes
    # so intermediate sweeps cannot be mistaken for tests of the previous commit.
    root = Path(__file__).resolve().parents[1]
    source_files = sorted([root/'CMakeLists.txt'] + [p for folder in ('csrc', 'torch_vulkan')
        for p in (root/folder).rglob('*') if p.suffix in ('.cpp', '.h', '.comp', '.spv', '.py', '.glsl')])
    source_hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    manifest = dict(torch=torch.__version__, torch_git=torch.version.git_version,
        backend_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        extension_sha256=hashlib.sha256(Path(vk._C.__file__).read_bytes()).hexdigest(),
        device=vk.device_name(), source_sha256=SOURCE_SHA256, opinfo_entries=len(op_db),
        working_tree_dirty=bool(subprocess.check_output(['git','status','--porcelain'],text=True).strip()),
        backend_source_sha256=hashlib.sha256(json.dumps(source_hashes,sort_keys=True).encode()).hexdigest(),
        backend_source_files=source_hashes,
        bypass_device_restrictions=True, selected=selected, excluded=excluded)
    built = root/'.cmake-build/build-source-manifest.json'
    manifest['source_identity'] = 'runtime_snapshot_unverified_against_build'
    if built.exists():
        build_manifest = json.loads(built.read_text())
        if build_manifest.get('extension_sha256') == manifest['extension_sha256']:
            manifest.update(build_manifest)
            manifest['source_identity'] = 'recorded_after_build'
    (args.out/'inventory.json').write_text(json.dumps(manifest,indent=2)+'\n')
    if args.collect: return
    done = {r['id'] for r in read_rows(args.out/'results.jsonl')}
    chosen = [r for r in selected if r['id'] not in done and (not args.match or args.match in r['id'])]
    info = {r['id']:r for r in chosen}
    current = args.out/'current.json'
    class Result(unittest.TestResult):
        def startTest(self,test):
            super().startTest(test)
            key=test.__class__.__name__+'.'+test._testMethodName
            self.row=dict(info[key],status='passed',details=[],start=time.time())
            current.write_text(json.dumps(self.row))
            vk.reset_execution_stats()
        def mark(self,status,detail):
            self.row['status']=status
            if len(self.row['details'])<4:self.row['details'].append(str(detail)[-14000:])
        def addSuccess(self,test): pass
        def addError(self,test,err):self.mark('error',''.join(traceback.format_exception(*err)))
        def addFailure(self,test,err):self.mark('failed',''.join(traceback.format_exception(*err)))
        def addSkip(self,test,reason):self.mark('skipped',reason)
        def addExpectedFailure(self,test,err):self.mark('expected_failure',''.join(traceback.format_exception(*err)))
        def addUnexpectedSuccess(self,test):self.mark('unexpected_success','Upstream expected failure unexpectedly passed')
        def addSubTest(self,test,subtest,err):
            if err:self.mark('failed',''.join(traceback.format_exception(*err)))
        def stopTest(self,test):
            try:
                self.row['execution']=vk.execution_stats()
                if self.row['status']=='passed' and self.row['execution']['cpu_fallback_calls']:
                    self.mark('fallback_attempt','Upstream test returned successfully after attempting forbidden CPU fallback')
            except Exception as exc:self.mark('error',f'Execution audit failed: {exc}')
            self.row['seconds']=time.time()-self.row.pop('start')
            append(args.out/'results.jsonl',self.row)
            current.unlink(missing_ok=True)
            super().stopTest(test)
    tests=[]
    for row in chosen:
        clsname,name=row['id'].split('.',1)
        tests.append(getattr(module,clsname)(name))
    result=Result()
    unittest.TestSuite(tests).run(result)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--collect',action='store_true');p.add_argument('--worker',action='store_true')
    p.add_argument('--match',default='');p.add_argument('--timeout',type=float,default=30)
    args=p.parse_args();args.source=args.source.resolve();args.out=args.out.resolve();args.out.mkdir(parents=True,exist_ok=True)
    assert hashlib.sha256(args.source.read_bytes()).hexdigest()==SOURCE_SHA256,'Upstream source hash mismatch'
    if args.worker:return worker(args)
    env=dict(os.environ,PYTHONPATH=str(Path(__file__).resolve().parents[1]),
        PYTORCH_TESTING_DEVICE_ONLY_FOR='privateuse1',TORCH_VULKAN_GPU_RESIDENT='1',
        TORCH_VULKAN_DEFER_SUBMISSIONS='1',TORCH_VULKAN_FORBID_CPU_FALLBACK='1',
        PYTORCH_TEST_WITH_SLOW='0',PYTHONUNBUFFERED='1')
    restarts=0
    while True:
        current=args.out/'current.json';current.unlink(missing_ok=True)
        cmd=[sys.executable,str(Path(__file__).resolve()),'--worker','--source',str(args.source),'--out',str(args.out),'--match',args.match]
        if args.collect:cmd.append('--collect')
        started=time.time();last_report=0;reason=None
        with (args.out/'worker.log').open('a') as log:
            proc=subprocess.Popen(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            while proc.poll() is None:
                now=time.time()
                try:active=json.loads(current.read_text())
                except (FileNotFoundError,json.JSONDecodeError):active=None
                if active and now-active['start']>args.timeout:reason='timeout'
                elif not active and now-started>180 and not (args.out/'inventory.json').exists():reason='startup_timeout'
                if reason:
                    os.killpg(proc.pid,signal.SIGKILL);proc.wait();break
                if now-last_report>20:
                    rows=read_rows(args.out/'results.jsonl')
                    print(json.dumps({'completed':len(rows),'outcomes':dict(collections.Counter(r['status'] for r in rows)),'current':active['id'] if active else 'initializing'}),flush=True)
                    last_report=now
                time.sleep(1)
        if proc.returncode==0:break
        if current.exists():
            row=json.loads(current.read_text());row['seconds']=time.time()-row.pop('start');row.update(status=reason or 'worker_crash',details=[f'worker exit {proc.returncode}']);append(args.out/'results.jsonl',row)
        else:raise RuntimeError(f'Worker failed outside a test: {proc.returncode}; see worker.log')
        restarts+=1
        if restarts>=100:raise RuntimeError('100 worker failures; results retained, remaining cases not run')
    rows=read_rows(args.out/'results.jsonl')
    print(json.dumps({'finished':True,'completed':len(rows),'outcomes':dict(collections.Counter(r['status'] for r in rows)),'worker_restarts':restarts}),flush=True)

if __name__=='__main__':main()
