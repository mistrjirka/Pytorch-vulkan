"""Reproduce the dispatch report from checked-in matched raw measurements."""
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parent / 'results' / 'conv-dispatch'

def read(name):
    return json.loads((ROOT / (name + '.json')).read_text())

def med(values):
    return statistics.median(values)

def main():
    runs = {name: read(name) for name in ('base', 'base_repeat', 'tuned', 'tuned_repeat', 'cuda')}
    assert len({run['convolution_source_sha256'] for run in runs.values()}) == 1
    print('## Measured results\n')
    print('RTX 2080 Ti, FP32, TF32 disabled. Eight warmups; 15 samples per run. '
          'Vulkan base/tuned/tuned/base order with CUDA between the two tuned runs; '
          'Vulkan medians pool 30 samples, CUDA uses 15. Residency and deferred submission enabled. '
          'Base means the current code with tuning disabled and the original tile preference, '
          'not an older checkout. First-use compilation/tuning is excluded.\n')
    print('| Case | Phase | Base ms | Tuned ms | CUDA ms | Base/tuned |')
    print('|---|---|---:|---:|---:|---:|')
    for i, case in enumerate(runs['base']['cases']):
        assert len({run['cases'][i]['input_sha256'] for run in runs.values()}) == 1
        for phase in case['measurements']:
            measurements = {name: run['cases'][i]['measurements'][phase] for name, run in runs.items()}
            pooled = lambda names: med([sample for name in names for sample in measurements[name]['wall']['samples_ms']])
            base, tuned, cuda = pooled(('base', 'base_repeat')), pooled(('tuned', 'tuned_repeat')), pooled(('cuda',))
            print(f"| {case['name']} | {phase} | {base:.4f} | {tuned:.4f} | {cuda:.4f} | {base/tuned:.2f}x |")
    print('\nThese are isolated operator phases. Small/depthwise cases sometimes regress despite unchanged '
          'selected kernels; the cache lookup and runtime/measurement state affect total latency. '
          'A separate same-process depthwise ABBA probe measured roughly 3 microseconds additional '
          'cache-enabled overhead, which does not explain every cross-process difference. '
          'Tuning stays opt-in; these data do not justify a universal default switch.\n')
    print('### Complete models\n')
    print('Same source architectures from model-zoo commit `c7c063101ec5209caf8022f5b4ea6948d1a8d6ed`. '
          'U-Net input [1,3,64,64]; StyleNet [1,3,36,36]. Same interleaved sampling. '
          'Vulkan uses its opt-in fused AdamW for both configurations; CUDA uses standard AdamW '
          'with foreach=False, so training compares these specific optimizer implementations. '
          'This is not a comparison against CUDA fused AdamW. Training includes forward, loss, '
          'backward and optimizer. Run-to-run training variation remains visible in raw samples.\n')
    print('| Model | Mode | Base ms | Tuned ms | CUDA ms | Base/tuned |')
    print('|---|---|---:|---:|---:|---:|')
    for model in ('unet', 'style_net'):
        models = {name: read(model+'_'+name) for name in runs}
        for mode in ('inference', 'training'):
            pooled = lambda names: med([sample for name in names for sample in models[name]['modes'][mode]['samples_ms']])
            base, tuned, cuda = pooled(('base', 'base_repeat')), pooled(('tuned', 'tuned_repeat')), pooled(('cuda',))
            print(f'| {model} | {mode} | {base:.3f} | {tuned:.3f} | {cuda:.3f} | {base/tuned:.2f}x |')
    print('\nThe separate `standard_optimizer/` diagnostic used standard AdamW on Vulkan too; '
          'its CPU fallbacks dominate training and it is not the configuration in this table. '
          'These timings and matching initial losses are not a claim of long-run training equivalence.\n')
    print('### Where CUDA is still faster\n')
    gpu = {case['name']:case for case in read('tall_gpu_profile')['cases']}
    cuda = {case['name']:case for case in read('cuda_kernel_profile')['cases']}
    print('Vulkan GPU timestamps use eager execution with transfers measured separately; '
          'the tall tile matches the selected heavy-workload implementation. CUDA values sum '
          'actual GPU kernel durations from five profiler iterations, excluding launch gaps. '
          'These separate instrumented runs explain computation cost; they are not end-to-end timings. '
          'The split-weight Vulkan interval includes the inter-kernel barrier and finalizer.\n')
    print('| Case | Phase | Vulkan GPU ms | CUDA GPU ms | Vulkan upload/download ms |')
    print('|---|---|---:|---:|---:|')
    for name in gpu:
        assert gpu[name]['input_sha256'] == cuda[name]['input_sha256']
        for phase, measurement in gpu[name]['measurements'].items():
            values = measurement['vulkan_gpu'].values()
            duration = lambda key: sum(value.get(key+'_ms_per_invocation',0) for value in values)
            cuda_ms = cuda[name]['measurements'][phase]['cuda_kernel_profile']['summed_kernel_us_per_iteration']/1000
            print(f'| {name} | {phase} | {duration("shader"):.4f} | {cuda_ms:.4f} | {duration("upload"):.4f}/{duration("download"):.4f} |')
    print('\nThe recorded CUDA kernels show FFT-based forward for both 10x10 and 11x11. '
          'Their input-gradient choices differ: FFT for 10x10, `dgrad_engine` for 11x11. '
          'The tested large 3x3 uses Winograd forward/input gradients and Winograd plus SGEMM '
          'for weight gradients. Our corresponding paths remain tiled implicit GEMM. '
          'Thus algorithm-family breadth, not only tile choice or submission overhead, is a '
          'concrete remaining gap. These choices are specific to the measured device/software/configuration.\n')
    print('### Validation\n')
    for name, result in read('validation').items():
        print(f'- {name}: '+ '; '.join(result['summary']))
    print('\nIncludes 61 new dispatch tests: forced tiles, ragged boundaries, filters9/10/11/12, '
          'groups/stride/dilation/layouts, gradients and masks, pipeline reuse, tuning-cache reuse, '
          'workspace changes, explicit-control bypass, signed-coordinate rejection and two-axis direct dispatch. '
          'All14 shader binaries pass spirv-val. Three pre-existing MSE output-resize warnings remain; '
          'Lavapipe skips the hardware-acceleration assertion. AMD/Intel hardware remains untested. '
          'The previously observed TinyResNet end-to-end numerical mismatch is not resolved by this work.\n')

if __name__ == '__main__':
    main()
