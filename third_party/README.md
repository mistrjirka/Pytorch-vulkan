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

The FP32 completion kernels listed in `scripts/compile_coverage.py` are original
implementations in this project. Philox4x32-10 is implemented from its counter
transformation using core GLSL integer operations; no cuRAND, cuDNN, MIOpen,
cuFFT, rocFFT or other vendor-library code or binary is included. PyTorch's
native metadata helpers are called through the installed PyTorch library.

The LDL shaders adapt LAPACK SSYTF2 Bunch-Kaufman pivot and block formulas at
v3.12.1. The permissive modified BSD notice is retained in
`licenses/lapack-LICENSE.txt` and included by package license metadata.
Pinned source and license hashes are in `lapack/NOTICE.md`. The Vulkan buffer,
parallel update, synchronization and device dispatch code is backend-specific.

## JAMA 1.0.3

The general real eigenvalue shader adapts the public-domain NIST/MathWorks
JAMA Hessenberg/Schur routines. Original source, release metadata and changes
are documented in [jama/NOTICE.md](jama/NOTICE.md). The distribution notice is
in [licenses/jama-PUBLIC-DOMAIN.txt](licenses/jama-PUBLIC-DOMAIN.txt).
