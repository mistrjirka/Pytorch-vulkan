"""Rebuild and validate the portable spatial, reduction and predicate shaders."""
from pathlib import Path
import subprocess
SHADERS=('average_pool2d','resize_linear_cubic','reduce_axes','native_pack_bool',
         'predicate','bool_all','bool_logic','where_float','power_clamp')
if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]/'csrc/shaders'
    for name in SHADERS:
        target=root/f'{name}.spv'
        subprocess.run(['glslangValidator','-V',str(root/f'{name}.comp'),'-o',str(target)],check=True)
        subprocess.run(['spirv-val',str(target)],check=True)
