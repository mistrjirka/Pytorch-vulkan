#!/usr/bin/env python3
"""Rebuild and validate convolution tiles and direct fallback (no GPU required).

Runtime convolution parameters stay push constants; the wrappers specialize
only work partitioning. Generated SPIR-V is checked in, like other shaders.
"""
from pathlib import Path
import subprocess


def main():
    shader_dir = Path(__file__).resolve().parents[1] / "csrc" / "shaders"
    for phase in ("conv2d_tiled", "conv2d_backward_input_tiled",
                  "conv2d_backward_weight_tiled"):
        for suffix in ("", "_wide", "_tall", "_register"):
            source = shader_dir / (phase + suffix + ".comp")
            output = source.with_suffix(".spv")
            subprocess.run(["glslangValidator", "-V", str(source), "-o", str(output)],
                           check=True)
            subprocess.run(["spirv-val", str(output)], check=True)

    for phase in ("conv2d_forward", "conv2d_backward_input", "conv2d_backward_weight",
                  "conv2d_backward_bias", "conv2d_depthwise"):
        source = shader_dir / (phase + ".comp")
        output = source.with_suffix(".spv")
        subprocess.run(["glslangValidator", "-V", str(source), "-o", str(output)],
                       check=True)
        subprocess.run(["spirv-val", str(output)], check=True)


if __name__ == "__main__":
    main()
