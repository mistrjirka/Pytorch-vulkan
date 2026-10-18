"""
torch-vulkan: Vulkan compute backend for PyTorch via PrivateUse1

pip install -e .

Requires: PyTorch, Vulkan SDK, Kompute
"""

import os
import subprocess

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        super().__init__(name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cfg = "Release"

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DCMAKE_BUILD_TYPE={cfg}",
            f"-DCMAKE_PREFIX_PATH={os.path.dirname(os.path.abspath(__file__))}",
        ]

        # Point CMake at PyTorch
        try:
            import torch
            cmake_args.append(f"-DCMAKE_PREFIX_PATH={torch.utils.cmake_prefix_path}")
        except ImportError:
            pass

        build_args = ["--config", cfg, "--parallel", os.environ.get("CMAKE_BUILD_PARALLEL_LEVEL", "4")]

        # Keep CMake's dependency downloads/build cache across pip reinstalls.
        self.build_temp = os.path.join(ext.sourcedir, ".cmake-build")
        os.makedirs(self.build_temp, exist_ok=True)
        subprocess.check_call(
            ["cmake", ext.sourcedir] + cmake_args, cwd=self.build_temp
        )
        subprocess.check_call(
            ["cmake", "--build", "."] + build_args, cwd=self.build_temp
        )


setup(
    name="torch-vulkan",
    version="0.1.0",
    author="Pytorch-vulkan contributors",

    description="Vulkan compute backend for PyTorch — runs on any GPU",
    url="https://github.com/mistrjirka/Pytorch-vulkan",
    license_files=["LICENSE", "third_party/README.md", "third_party/licenses/*.txt"],
    packages=["torch_vulkan"],
    ext_modules=[CMakeExtension("torch_vulkan._C")],
    cmdclass={"build_ext": CMakeBuild},
    python_requires=">=3.10",
    install_requires=["torch>=2.14.1,<2.15"],
)
