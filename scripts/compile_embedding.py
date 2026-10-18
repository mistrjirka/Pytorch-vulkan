"""Rebuild and validate exact-integer embedding and int32 packing shaders."""
from pathlib import Path
import subprocess
if __name__ == '__main__':
    root = Path(__file__).resolve().parents[1] / 'csrc/shaders'
    for name in ('embedding', 'embedding_validate', 'native_pack_int32'):
        target = root / f'{name}.spv'
        subprocess.run(['glslangValidator', '-V', str(root / f'{name}.comp'), '-o', str(target)], check=True)
        subprocess.run(['spirv-val', str(target)], check=True)
