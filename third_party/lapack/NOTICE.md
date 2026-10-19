# LAPACK reference algorithm

The Vulkan LDL factorization uses the Bunch-Kaufman pivot/block formulas from
LAPACK SSYTF2, translated into a parallel Vulkan shader with separate workspace.
The solve retains LAPACK lower-LD interleaved pivot semantics.

Upstream: https://github.com/Reference-LAPACK/lapack/tree/v3.12.1
Source: https://github.com/Reference-LAPACK/lapack/blob/v3.12.1/SRC/ssytf2.f
Source SHA256: a8032797e574d55340ed9cd1e59eabf04c5d1a25e4224c0a2f9fcdf7c2a2dfdc
License SHA256: 978539245c405775bf347b1256c36df4daf8b69a64b099e94b1d53742201621e

LAPACK uses a permissive modified BSD license. The complete upstream notice
and redistribution terms are retained in LICENSE. The C++ host dispatch and
Vulkan storage, synchronization and indexing code are written for this backend.
