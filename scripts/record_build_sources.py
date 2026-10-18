"""Record source hashes immediately after a completed backend build."""
import argparse
import hashlib
import json
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument('--extension', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
root = Path(__file__).resolve().parents[1]
files = sorted([root / 'CMakeLists.txt'] + [f for folder in ('csrc', 'torch_vulkan')
    for f in (root / folder).rglob('*') if f.suffix in ('.cpp', '.h', '.comp', '.spv', '.py', '.glsl')])
hashes = {str(f.relative_to(root)): hashlib.sha256(f.read_bytes()).hexdigest() for f in files}
a.output.parent.mkdir(parents=True, exist_ok=True)
a.output.write_text(json.dumps(dict(extension_sha256=hashlib.sha256(a.extension.read_bytes()).hexdigest(),
    backend_source_sha256=hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest(),
    backend_source_files=hashes), indent=2) + '\n')
