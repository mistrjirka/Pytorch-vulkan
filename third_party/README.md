# Kernel provenance

`csrc/shaders/resize_linear_cubic.comp` adapts the cubic coefficient calculation
from Tencent ncnn `src/layer/vulkan/shader/interp_bicubic_coeffs.comp`, pinned at
`44b02dabdcf97499281a4692ebd65bad73dcf8b6` (Copyright 2019 Tencent,
BSD-3-Clause). The upstream license is retained in `licenses/ncnn-LICENSE.txt`.
This component retains its BSD conditions alongside this project's MIT code.
The package metadata includes this attribution and the license files for binary
and source distributions.

Changes: ordinary FP32 buffer bindings; runtime geometry; Horner evaluation;
fourth coefficient evaluated directly to match ATen rounding; clamped taps
rather than ncnn's shifted edge coefficients. The coordinate search and
input-gather backward implementation are original. No ncnn build/runtime
dependency or third-party SPIR-V binary is included.

Original implementations in this change cover pooling, reduction, predicates,
selection, power/clamp and boolean packing. They are not copies of upstream
training kernels. Burn's MIT/Apache-2.0 licensing and CubeCL approach were also
reviewed as alternatives; no Burn/CubeCL code is included.
