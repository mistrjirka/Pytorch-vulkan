## Measured results

RTX 2080 Ti, FP32, TF32 disabled. Eight warmups; 15 samples per run. Vulkan base/tuned/tuned/base order with CUDA between the two tuned runs; Vulkan medians pool 30 samples, CUDA uses 15. Residency and deferred submission enabled. Base means the current code with tuning disabled and the original tile preference, not an older checkout. First-use compilation/tuning is excluded.

| Case | Phase | Base ms | Tuned ms | CUDA ms | Base/tuned |
|---|---|---:|---:|---:|---:|
| pointwise | forward | 0.0823 | 0.0845 | 0.0408 | 0.97x |
| pointwise | dinput | 0.1082 | 0.1043 | 0.0452 | 1.04x |
| pointwise | dweight | 0.1346 | 0.1212 | 0.0762 | 1.11x |
| pointwise | dbias | 0.0840 | 0.0876 | 0.0430 | 0.96x |
| rect3x5 | forward | 0.0976 | 0.0957 | 0.0430 | 1.02x |
| rect3x5 | dinput | 0.1605 | 0.1329 | 0.0503 | 1.21x |
| rect3x5 | dweight | 0.1271 | 0.1230 | 0.0678 | 1.03x |
| rect3x5 | dbias | 0.0813 | 0.0846 | 0.0388 | 0.96x |
| filter10x10 | forward | 0.3583 | 0.2860 | 0.1213 | 1.25x |
| filter10x10 | dinput | 0.7622 | 0.5446 | 0.1243 | 1.40x |
| filter10x10 | dweight | 0.3889 | 0.3418 | 0.0940 | 1.14x |
| filter10x10 | dbias | 0.0832 | 0.1150 | 0.0394 | 0.72x |
| filter11x11 | forward | 0.4231 | 0.3526 | 0.1222 | 1.20x |
| filter11x11 | dinput | 0.9366 | 0.6757 | 0.1543 | 1.39x |
| filter11x11 | dweight | 0.4675 | 0.4055 | 0.1013 | 1.15x |
| filter11x11 | dbias | 0.1116 | 0.1472 | 0.0387 | 0.76x |
| grouped | forward | 0.1183 | 0.1292 | 0.0668 | 0.92x |
| grouped | dinput | 0.1621 | 0.1814 | 0.0880 | 0.89x |
| grouped | dweight | 0.1880 | 0.1770 | 0.0632 | 1.06x |
| grouped | dbias | 0.1115 | 0.1369 | 0.0403 | 0.81x |
| depthwise | forward | 0.0700 | 0.0961 | 0.0234 | 0.73x |
| depthwise | dinput | 0.0942 | 0.1291 | 0.0353 | 0.73x |
| depthwise | dweight | 0.0919 | 0.1256 | 0.0351 | 0.73x |
| depthwise | dbias | 0.0834 | 0.1181 | 0.0407 | 0.71x |
| conv3x3_large | forward | 0.4543 | 0.3083 | 0.0659 | 1.47x |
| conv3x3_large | dinput | 0.8407 | 0.4836 | 0.0642 | 1.74x |
| conv3x3_large | dweight | 0.6165 | 0.4378 | 0.1326 | 1.41x |
| conv3x3_large | dbias | 0.1267 | 0.1336 | 0.0387 | 0.95x |
| conv3x3_small | forward | 0.1545 | 0.1584 | 0.0520 | 0.98x |
| conv3x3_small | dinput | 0.2677 | 0.2174 | 0.0562 | 1.23x |
| conv3x3_small | dweight | 0.2073 | 0.1844 | 0.0678 | 1.12x |
| conv3x3_small | dbias | 0.1073 | 0.1151 | 0.0392 | 0.93x |

These are isolated operator phases. Small/depthwise cases sometimes regress despite unchanged selected kernels; the cache lookup and runtime/measurement state affect total latency. A separate same-process depthwise ABBA probe measured roughly 3 microseconds additional cache-enabled overhead, which does not explain every cross-process difference. Tuning stays opt-in; these data do not justify a universal default switch.

### Complete models

Same source architectures from model-zoo commit `c7c063101ec5209caf8022f5b4ea6948d1a8d6ed`. U-Net input [1,3,64,64]; StyleNet [1,3,36,36]. Same interleaved sampling. Vulkan uses its opt-in fused AdamW for both configurations; CUDA uses standard AdamW with foreach=False, so training compares these specific optimizer implementations. This is not a comparison against CUDA fused AdamW. Training includes forward, loss, backward and optimizer. Run-to-run training variation remains visible in raw samples.

| Model | Mode | Base ms | Tuned ms | CUDA ms | Base/tuned |
|---|---|---:|---:|---:|---:|
| unet | inference | 2.560 | 2.173 | 1.088 | 1.18x |
| unet | training | 16.059 | 13.398 | 7.631 | 1.20x |
| style_net | inference | 40.843 | 39.983 | 1.539 | 1.02x |
| style_net | training | 189.955 | 145.284 | 12.214 | 1.31x |

The separate `standard_optimizer/` diagnostic used standard AdamW on Vulkan too; its CPU fallbacks dominate training and it is not the configuration in this table. These timings and matching initial losses are not a claim of long-run training equivalence.

### Where CUDA is still faster

Vulkan GPU timestamps use eager execution with transfers measured separately; the tall tile matches the selected heavy-workload implementation. CUDA values sum actual GPU kernel durations from five profiler iterations, excluding launch gaps. These separate instrumented runs explain computation cost; they are not end-to-end timings. The split-weight Vulkan interval includes the inter-kernel barrier and finalizer.

| Case | Phase | Vulkan GPU ms | CUDA GPU ms | Vulkan upload/download ms |
|---|---|---:|---:|---:|
| filter10x10 | forward | 0.2188 | 0.1171 | 0.0336/0.0729 |
| filter10x10 | dinput | 0.4349 | 0.1117 | 0.0378/0.0262 |
| filter10x10 | dweight | 0.2132 | 0.0564 | 0.0348/0.0630 |
| filter10x10 | dbias | 0.0083 | 0.0069 | 0.0222/0.0035 |
| filter11x11 | forward | 0.2592 | 0.1170 | 0.0366/0.0565 |
| filter11x11 | dinput | 0.5239 | 0.1223 | 0.0403/0.0247 |
| filter11x11 | dweight | 0.2414 | 0.0586 | 0.0338/0.0714 |
| filter11x11 | dbias | 0.0077 | 0.0062 | 0.0215/0.0036 |
| conv3x3_large | forward | 0.2808 | 0.0408 | 0.0911/0.6472 |
| conv3x3_large | dinput | 0.4629 | 0.0288 | 0.1709/0.3328 |
| conv3x3_large | dweight | 0.3498 | 0.0931 | 0.2442/0.0154 |
| conv3x3_large | dbias | 0.0230 | 0.0065 | 0.1635/0.0032 |

The recorded CUDA kernels show FFT-based forward for both 10x10 and 11x11. Their input-gradient choices differ: FFT for 10x10, `dgrad_engine` for 11x11. The tested large 3x3 uses Winograd forward/input gradients and Winograd plus SGEMM for weight gradients. Our corresponding paths remain tiled implicit GEMM. Thus algorithm-family breadth, not only tile choice or submission overhead, is a concrete remaining gap. These choices are specific to the measured device/software/configuration.

### Validation

- rtx: 422 passed, 3 warnings in 12.34s
- v100: 422 passed, 3 warnings in 11.34s
- lavapipe: 421 passed, 1 skipped, 3 warnings in 10.64s
- resident: 422 passed, 3 warnings in 12.63s

Includes 61 new dispatch tests: forced tiles, ragged boundaries, filters9/10/11/12, groups/stride/dilation/layouts, gradients and masks, pipeline reuse, tuning-cache reuse, workspace changes, explicit-control bypass, signed-coordinate rejection and two-axis direct dispatch. All14 shader binaries pass spirv-val. Three pre-existing MSE output-resize warnings remain; Lavapipe skips the hardware-acceleration assertion. AMD/Intel hardware remains untested. The previously observed TinyResNet end-to-end numerical mismatch is not resolved by this work.

