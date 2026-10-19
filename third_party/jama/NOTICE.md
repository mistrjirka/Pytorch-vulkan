# JAMA 1.0.3 eigenvalue reference

The real nonsymmetric Hessenberg/Schur algorithm in `eig_general.comp` is
adapted from JAMA 1.0.3 `EigenvalueDecomposition.java` (orthes, cdiv and hqr2).
JAMA is a cooperative MathWorks/NIST product released to the public domain.
The published artifact's POM explicitly declares Public Domain distribution.

- NIST release and public-domain notice: https://math.nist.gov/javanumerics/jama/
- Published sources: https://repo.maven.apache.org/maven2/gov/nist/math/jama/1.0.3/jama-1.0.3-sources.jar
- Source SHA256: c9d8efe5cfd22b7dddfc3d7fb185bb01d1a6ed2130610c0fcfe7195ccee36a81
- The source file is retained as a deterministic gzip archive, with the exact
  upstream bytes and SHA256 above. The published POM is also retained.

The Vulkan port uses FP32 workspaces, exact power-of-two scaling, batched device
buffers, bounded iteration with status reporting and normalized complex64
outputs. It does not execute Java or download matrices to a CPU solver.
