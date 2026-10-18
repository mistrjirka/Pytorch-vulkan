"""Whole-network training attribution; use compare_cuda_vulkan.py for latency.

Uses the same model/data seeds and medium shapes as that benchmark. Profiler
shape recording can retain tensors and alter allocation/reuse, so these CPU
intervals and call counts describe an instrumented run, not normal step time.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch
from compare_cuda_vulkan import CONFIG, data_for_case, synchronize


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--backend', choices=['cuda', 'vulkan'], required=True)
    p.add_argument('--model', choices=CONFIG, required=True)
    p.add_argument('--scale', choices=['tiny', 'small', 'medium', 'reference'], default='medium')
    p.add_argument('--zoo', type=Path, required=True)
    p.add_argument('--warmup', type=int, default=8)
    p.add_argument('--steps', type=int, default=3)
    p.add_argument('--trace', type=Path)
    p.add_argument('--finite-leaky-composite', action='store_true',
                   help='Diagnostic finite-input composition; not a backend implementation.')
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    if args.warmup < 1 or args.steps < 1:
        p.error('warmup and steps must be positive')
    if args.finite_leaky_composite and args.backend != 'vulkan':
        p.error('--finite-leaky-composite requires --backend vulkan')
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = False
    sys.path.insert(0, str((args.zoo / 'benchmarks').resolve()))
    import model_zoo
    torch.manual_seed(42)
    model, _, loss_fn = model_zoo.MODELS[args.model]()
    if args.finite_leaky_composite:
        import types
        from probe_activation_residency import FiniteLeaky
        for module in model.modules():
            if isinstance(module, torch.nn.LeakyReLU):
                assert not module.inplace
                def forward(self, x):
                    return FiniteLeaky.apply(x, self.negative_slope)
                module.forward = types.MethodType(forward, module)
    state_hash = hashlib.sha256()
    for name, value in model.state_dict().items():
        state_hash.update(name.encode())
        state_hash.update(value.contiguous().numpy().tobytes())
    x, label = data_for_case(args.model, CONFIG[args.model][args.scale],
                             torch.Generator().manual_seed(123))
    input_hash = hashlib.sha256(x.numpy().tobytes() + label.numpy().tobytes()).hexdigest()
    if args.backend == 'vulkan':
        import torch_vulkan
        from torch_vulkan.optim import AdamW
        gpu = torch_vulkan.device_name()
        assert torch_vulkan.is_hardware_accelerated()
        def snapshot():
            return dict(cache=torch_vulkan.cache_stats(), queue=torch_vulkan.queue_stats(),
                        dispatch=torch_vulkan.conv_dispatch_stats(), gpu=torch_vulkan.gpu_timing_stats())
    else:
        AdamW = torch.optim.AdamW
        gpu = torch.cuda.get_device_name()
        snapshot = lambda: {}
    model.to(args.backend, dtype=torch.float32).train()
    opt = AdamW(model.parameters(), lr=1e-4,
                **({'foreach': False} if args.backend == 'cuda' else {}))
    x, label = x.to(args.backend), label.to(args.backend)

    def step():
        with torch.profiler.record_function('network/zero_grad'):
            opt.zero_grad(set_to_none=True)
        with torch.profiler.record_function('network/forward'):
            pred = model(x)
        with torch.profiler.record_function('network/loss'):
            loss = loss_fn(pred, label)
        with torch.profiler.record_function('network/backward'):
            loss.backward()
        with torch.profiler.record_function('network/optimizer'):
            opt.step()
        with torch.profiler.record_function('network/final_sync'):
            synchronize(args.backend)
        return loss

    warmup_ms = []
    first_loss = None
    for i in range(args.warmup):
        t = time.perf_counter()
        loss = step()
        warmup_ms.append((time.perf_counter() - t) * 1000)
        if i == 0:
            first_loss = float(loss.detach().cpu())
    before = snapshot()
    activities = [torch.profiler.ProfilerActivity.CPU]
    if args.backend == 'cuda':
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(activities=activities, record_shapes=True) as prof:
        for _ in range(args.steps):
            step()
    after = snapshot()
    events = [dict(name=e.key, shapes=e.input_shapes, calls_per_step=e.count / args.steps,
                   self_cpu_ms=e.self_cpu_time_total / (1000 * args.steps),
                   inclusive_cpu_ms=e.cpu_time_total / (1000 * args.steps),
                   inclusive_device_ms=e.device_time_total / (1000 * args.steps))
              for e in prof.key_averages(group_by_input_shape=True)]
    kernels = defaultdict(lambda: dict(calls_per_step=0, gpu_ms_per_step=0))
    for e in prof.events():
        if getattr(e.device_type, 'name', '') == 'CUDA' and not e.is_user_annotation:
            kernels[e.name]['calls_per_step'] += 1 / args.steps
            kernels[e.name]['gpu_ms_per_step'] += (e.time_range.end - e.time_range.start) / (1000 * args.steps)
    if args.trace:
        args.trace.parent.mkdir(parents=True, exist_ok=True)
        prof.export_chrome_trace(str(args.trace))
    def delta(a, b):
        return {k: delta(a.get(k, {}), v) if isinstance(v, dict) else v-a.get(k, 0)
                for k, v in b.items()}
    report = dict(
        revision=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        profile_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        zoo_revision=subprocess.check_output(['git', '-C', str(args.zoo), 'rev-parse', 'HEAD'], text=True).strip(),
        gpu=gpu, backend=args.backend, model=args.model, dtype='float32',
        finite_leaky_composite=args.finite_leaky_composite,
        shape=CONFIG[args.model][args.scale], torch=torch.__version__,
        initial_state_sha256=state_hash.hexdigest(), input_sha256=input_hash,
        model_seed=42, input_seed=123, first_loss=first_loss,
        optimizer='vulkan_fused' if args.backend == 'vulkan' else 'standard_foreach_false',
        env={k:v for k,v in os.environ.items() if k.startswith('TORCH_VULKAN_') or k=='CUDA_VISIBLE_DEVICES'},
        warmup_ms=warmup_ms, profiled_steps=args.steps, counter_delta=delta(before, after),
        operators=sorted(events, key=lambda r:r['self_cpu_ms'], reverse=True),
        device_events=[dict(name=k, **v) for k,v in sorted(kernels.items(), key=lambda kv:kv[1]['gpu_ms_per_step'], reverse=True)],
        notes=['Instrumented attribution only; shape recording can retain tensors and alter buffer reuse.',
               'Inclusive operator and stage durations overlap; never add them together.',
               'CPU operator times include waits; they are not GPU execution times.',
               'Vulkan queue counters cover deferred submissions, not all Vulkan API submissions.',
               'Vulkan GPU timestamp coverage can be empty under resident/deferred execution.',
               'Device events include CUDA kernels and memory operations; summed durations can overlap.'])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2)+'\n')
    print(args.model, args.backend, 'first_loss', first_loss)
    print([(e['name'], round(e['self_cpu_ms'], 3)) for e in report['operators'][:8]])


if __name__ == '__main__':
    main()
