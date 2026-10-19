# Native correctness checkpoint 13

A supplemental component-wise boundary probe found complex scalar power errors
at zero, infinity and NaN, after the original 8,696-case selection passed its
scope gate on d43d267. The task was kept open. The original audit is retained
in ../upstream-opinfo-fixes-d43d267; it did not cover these new boundary probes.

The shader now handles zero poles before GLSL's undefined atan(0,0), and follows
the pinned CPU sqrt, rsqrt, reciprocal and -2 specializations. It handles mixed
infinity/NaN components and uses a scaled complex square root. The shared FFT
scale/conjugate shader and C++ runtime remain unchanged. This is original
geometry/dtype math, with no model-specific dispatch or shape list.

The 14 boundary tests cover 49 complex input combinations each, with real and
imaginary components compared independently so one NaN cannot hide a wrong
other component. All 686 combinations pass on RTX, V100 and Lavapipe; the 14
cases also pass in strict eager mode on Lavapipe. All 1,664 native regressions
pass on each device. Ordinary repository: RTX/V100 3,408 passed; Lavapipe 3,407
passed and one skipped. Full strict V100: 3,397 passed plus the same 11 unchanged
explicit FP64 failures. All 31 original complex64-only samples and ten composite
compliance probes pass again. Matched CUDA boundary controls retain individual
raw outcomes and use identical serialized inputs. CUDA matches six cases and
differs from CPU in eight zero/nonfinite component cases; Vulkan matches CPU
in all 14. The earlier 14 finite-value controls pass on both APIs. No speed
claim follows from these checks.
The full immutable upstream run on this updated source remains pending.
