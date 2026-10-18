"""Matched model training CPU operator profile and CUDA kernel profile."""
import argparse
import json
from pathlib import Path
import sys
import time

import torch
from compare_cuda_vulkan import CONFIG, data_for_case, synchronize

p = argparse.ArgumentParser()
p.add_argument("--backend", choices=["cuda", "vulkan"], required=True)
p.add_argument("--model", choices=CONFIG, default="unet")
p.add_argument("--zoo", type=Path, required=True)
p.add_argument("--out", type=Path, required=True)
args = p.parse_args()
torch.set_num_threads(8)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.backends.cudnn.benchmark = False
sys.path.insert(0, str(args.zoo / "benchmarks"))
import model_zoo
torch.manual_seed(2026)
model, _, loss_fn = model_zoo.MODELS[args.model]()
if args.backend == "vulkan":
    import torch_vulkan
    from torch_vulkan.optim import AdamW
    gpu = torch_vulkan.device_name()
else:
    AdamW = torch.optim.AdamW
    gpu = torch.cuda.get_device_name()
model.to(args.backend, dtype=torch.float32).train()
opt = AdamW(model.parameters(), lr=1e-4, **({"foreach": False} if args.backend == "cuda" else {}))
x, label = data_for_case(args.model, CONFIG[args.model]["medium"],
                         torch.Generator().manual_seed(2026))
x, label = x.to(args.backend), label.to(args.backend)

def step():
    opt.zero_grad(set_to_none=True)
    loss = loss_fn(model(x), label)
    loss.backward()
    opt.step()
    synchronize(args.backend)

warmup_ms = []
for _ in range(8):
    t = time.perf_counter()
    step()
    warmup_ms.append((time.perf_counter() - t) * 1000)
before = torch_vulkan.cache_stats() if args.backend == "vulkan" else {}
activities = [torch.profiler.ProfilerActivity.CPU]
if args.backend == "cuda":
    activities.append(torch.profiler.ProfilerActivity.CUDA)
with torch.profiler.profile(activities=activities) as prof:
    for _ in range(3):
        step()
after = torch_vulkan.cache_stats() if args.backend == "vulkan" else {}
events = [dict(name=e.key, calls_per_step=e.count / 3,
               self_cpu_ms=e.self_cpu_time_total / 3000,
               inclusive_cpu_ms=e.cpu_time_total / 3000,
               device_ms=e.device_time_total / 3000)
          for e in prof.key_averages()]
report = dict(gpu=gpu, backend=args.backend, model=args.model, dtype="float32",
              shape=CONFIG[args.model]["medium"], torch=torch.__version__,
              warmup_ms=warmup_ms, profiled_steps=3,
              cache_delta={k: after[k]-before.get(k, 0) for k in after},
              top_self_cpu=sorted(events, key=lambda r: r["self_cpu_ms"], reverse=True)[:20],
              top_inclusive_cpu=sorted(events, key=lambda r: r["inclusive_cpu_ms"], reverse=True)[:20])
args.out.parent.mkdir(parents=True, exist_ok=True)
args.out.write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
