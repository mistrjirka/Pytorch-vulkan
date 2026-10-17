"""Profile MLP training latency by stage, excluding data/model initialization."""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time
import torch

ap = argparse.ArgumentParser()
ap.add_argument("--backend", choices=("cpu", "cuda", "vulkan"), required=True)
ap.add_argument("--zoo", type=Path, required=True)
ap.add_argument("--out", type=Path)
args = ap.parse_args()
torch.set_num_threads(8)
if args.backend == "vulkan":
    import torch_vulkan
    device_name = torch_vulkan.device_name()
elif args.backend == "cuda":
    device_name = torch.cuda.get_device_name(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
else:
    device_name = "AMD EPYC CPU"
sys.path.insert(0, str((args.zoo / "benchmarks").resolve()))
import model_zoo
torch.manual_seed(42)
model, _, loss_fn = model_zoo.MODELS["mlp"]()
model = model.to(args.backend).train()
generator = torch.Generator().manual_seed(123)
x = torch.randn(128, 784, generator=generator).to(args.backend)
y = torch.randint(10, (128,), generator=generator).to(args.backend)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, foreach=False)

def sync():
    if args.backend == "cuda":
        torch.cuda.synchronize()

samples = []
for i in range(9):
    sync()
    start = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    sync()
    t0 = time.perf_counter()
    result = model(x)
    sync()
    t1 = time.perf_counter()
    loss = loss_fn(result, y)
    sync()
    t2 = time.perf_counter()
    loss.backward()
    sync()
    t3 = time.perf_counter()
    optimizer.step()
    sync()
    t4 = time.perf_counter()
    if i >= 3:
        samples.append({"zero_grad": (t0-start)*1000,
                        "forward": (t1-t0)*1000,
                        "loss": (t2-t1)*1000,
                        "backward": (t3-t2)*1000,
                        "adamw": (t4-t3)*1000,
                        "total": (t4-start)*1000})
report = {"backend": args.backend, "device": device_name, "torch": torch.__version__,
          "batch": 128, "dtype": "float32",
          "phases_ms": {key: round(statistics.median(s[key] for s in samples), 3)
                        for key in samples[0]}}
print(json.dumps(report, indent=2), flush=True)
if args.out:
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
