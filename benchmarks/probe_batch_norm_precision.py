"""Matched BatchNorm FP32 accuracy against rounded-input FP64 and phase timings."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import time

import torch
from probe_conv_dispatch import profile_cuda_kernels, timestamp_delta


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=['vulkan', 'cuda', 'cpu'], required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=30)
    parser.add_argument('--shader-dir', type=Path, help='Diagnostic SPIR-V override for Vulkan A/B runs')
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    torch.set_num_threads(6)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if args.backend == 'vulkan':
        import torch_vulkan
        if args.shader_dir:
            torch_vulkan._C._set_shader_dir(str(args.shader_dir.resolve()))
        sync, stats = torch_vulkan.synchronize, torch_vulkan.gpu_timing_stats
        device_name = torch_vulkan.device_name()
    else:
        sync = torch.cuda.synchronize if args.backend == 'cuda' else lambda: None
        stats = lambda: {}
        device_name = torch.cuda.get_device_name() if args.backend == 'cuda' else 'CPU'
    cases = []
    configs = [(str(m)+'_'+str(offset), (m, 512), offset, noise)
               for m in (2, 3, 8, 128) for offset, noise in ((0., 1.), (10000., .1))]
    configs += [('spatial', (8, 64, 32, 32), 0., 1.)]
    for name, shape, offset, noise in configs:
        gen = torch.Generator().manual_seed(42)
        tensors = (torch.randn(shape, generator=gen)*noise+offset,
                   torch.randn(shape, generator=gen), torch.rand(shape[1], generator=gen)+.5)
        x, g, w = [t.to(args.backend) for t in tensors]
        xd, gd, wd = [t.double() for t in tensors]
        def forward(xx, ww):
            return torch.ops.aten.native_batch_norm(xx, ww, None, None, None, True, .1, 1e-5)
        def backward(gg, xx, ww, mean, inv):
            return torch.ops.aten.native_batch_norm_backward(
                gg, xx, ww, None, None, mean, inv, True, 1e-5, [True]*3)
        yd, md, invd = forward(xd, wd)
        refs = (yd, md, invd, *backward(gd, xd, wd, md, invd))
        y, mean, inv = forward(x, w)
        actual = (y, mean, inv, *backward(g, x, w, mean, inv))
        parity = {key: {'max_abs': (a.cpu().double()-r).abs().max().item(),
                        'reference_max_abs': r.abs().max().item()}
                  for key, a, r in zip(('output','mean','invstd','dinput','dweight','dbias'), actual, refs)}
        row = dict(name=name, shape=shape, input_sha256=hashlib.sha256(
            b''.join(t.numpy().tobytes() for t in tensors)).hexdigest(), parity=parity)
        for phase, fn in [('forward', lambda: forward(x, w)),
                          ('backward', lambda: backward(g, x, w, mean, inv))]:
            for _ in range(8):
                result = fn(); sync(); del result
            before = stats(); times = []
            for _ in range(args.repeats):
                sync(); start = time.perf_counter(); result = fn(); sync()
                times.append((time.perf_counter()-start)*1000); del result
            row[phase] = dict(median_ms=statistics.median(times), samples_ms=times,
                              gpu_timestamps=timestamp_delta(before, stats(), args.repeats))
            if args.backend == 'cuda':
                row[phase]['cuda_kernels'] = profile_cuda_kernels(fn, sync)
        cases.append(row)
    root = Path(__file__).resolve().parents[1]
    shader_names = ('batch_norm_forward', 'batch_norm_backward')
    report = dict(backend=args.backend, device=device_name, torch=torch.__version__,
                  commit=subprocess.check_output(['git','rev-parse','HEAD'], text=True).strip(),
                  shaders={n:hashlib.sha256(((args.shader_dir or root/'csrc/shaders')/f'{n}.spv').read_bytes()).hexdigest()
                           for n in shader_names},
                  host_sources={n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in
                                ('csrc/ops/batch_norm.cpp','csrc/ops/native_utils.h')},
                  environment={k:v for k,v in os.environ.items() if k.startswith('TORCH_VULKAN_')},
                  note='FP32 inputs; FP64 CPU reference; wall includes synchronization; phases separate.',
                  cases=cases)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2)+'\n')
    print(args.out)


if __name__ == '__main__':
    main()
