"""Matched FP32 network latency, correctness, residency and diagnostic profiles.

Run one case/backend per process, with the same torch revision. Wall mode uses
normal deferred Vulkan execution. GPU timestamps require non-deferred execution
and are diagnostic only; phase mode deliberately synchronizes each phase.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, os, statistics, subprocess, sys, time, traceback
from pathlib import Path
import torch
from feature_networks import CASES, make_case
from compare_cuda_vulkan import CONFIG, data_for_case

ZOO_CASES = ("mlp", "unet", "resnet18", "mobilenet_v3", "embedding", "transformer")

def digest(tensors):
    h = hashlib.sha256()
    for tensor in tensors:
        t = tensor.detach().cpu().contiguous()
        h.update(str((str(t.dtype), tuple(t.shape))).encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()

def summary(xs):
    return dict(median=statistics.median(xs), min=min(xs), max=max(xs), samples=xs)

def delta(a, b):
    return {k: v-a.get(k, 0) for k, v in b.items()}

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=["vulkan", "cuda"], required=True)
    p.add_argument("--case", required=True)
    p.add_argument("--mode", choices=["wall", "phases", "gpu"], default="wall")
    p.add_argument("--scale", choices=CONFIG["mlp"], default="medium")
    p.add_argument("--warmup", type=int, default=8)
    p.add_argument("--repeats", type=int, default=25)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--skip-parity", action="store_true")
    p.add_argument("--zoo", type=Path, default=Path("/workspace/pytorch-XDNA2-vulkan-zoo-reference"))
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(a.threads)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    vk = None
    if a.backend == "vulkan":
        import torch_vulkan as vk
        assert vk.is_hardware_accelerated()
        assert os.environ.get("TORCH_VULKAN_FORBID_CPU_FALLBACK") is not None
        sync = vk.synchronize
        device = vk.device_name()
        if a.mode == "gpu":
            assert os.environ.get("TORCH_VULKAN_PROFILE_TIMESTAMPS") == "1"
            assert os.environ.get("TORCH_VULKAN_DEFER_SUBMISSIONS") != "1"
    else:
        sync = torch.cuda.synchronize
        device = torch.cuda.get_device_name()
    row = dict(case=a.case, backend=a.backend, mode=a.mode, scale=a.scale,
               device=device, torch=torch.__version__, torch_git=torch.version.git_version,
               optimizer="torch.optim.AdamW(lr=1e-4,foreach=False,amsgrad=True)",
               dtype="float32", tf32=False, threads=a.threads, warmup=a.warmup,
               repeats=a.repeats, parity_tolerance=dict(rtol=2e-3, atol=8e-5),
               source_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
               harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               environment={k:v for k,v in os.environ.items() if k.startswith("TORCH_VULKAN_") or k in ["CUDA_VISIBLE_DEVICES"]})
    if vk:
        row["extension_sha256"] = hashlib.sha256(Path(vk._C.__file__).read_bytes()).hexdigest()
    def save():
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(json.dumps(row, indent=2)+"\n")
    try:
        torch.manual_seed(42)
        if a.case in ZOO_CASES:
            sys.path.insert(0, str(a.zoo/"benchmarks"))
            import model_zoo
            cpu, _, loss_fn = model_zoo.MODELS[a.case]()
            x, target = data_for_case(a.case, CONFIG[a.case][a.scale],
                                      torch.Generator().manual_seed(123))
            row["zoo_commit"] = subprocess.check_output(["git","-C",str(a.zoo),"rev-parse","HEAD"],text=True).strip()
        else:
            cpu, x = make_case(a.case)
            target = None
            loss_fn = lambda out, _: out.square().mean()
        cpu.train()
        row.update(shape=list(x.shape), parameters=sum(p.numel() for p in cpu.parameters()),
                   input_sha256=digest([x]+([target] if target is not None else [])),
                   weights_sha256=digest(cpu.state_dict().values()))
        # Exact cross-device comparisons cannot share dropout masks. Disable
        # dropout only for parity; restore the original probabilities for timing.
        dropout = {name: module.p for name, module in cpu.named_modules()
                   if isinstance(module, torch.nn.modules.dropout._DropoutNd)}
        if not a.skip_parity:
            for name in dropout:
                cpu.get_submodule(name).p = 0.0
        model = copy.deepcopy(cpu).to(a.backend)
        inp = x.to(a.backend).requires_grad_(x.is_floating_point())
        label = target.to(a.backend) if target is not None else None
        if not a.skip_parity:
            ci = x.clone().requires_grad_(x.is_floating_point())
            co = cpu(ci)
            cl = loss_fn(co, target)
            cl.backward()
            go = model(inp)
            gl = loss_fn(go, label)
            gl.backward()
            sync()
            pairs = [("output", go.detach().cpu(), co.detach()),
                     ("loss", gl.detach().cpu(), cl.detach())]
            if ci.grad is not None:
                pairs.append(("input_gradient", inp.grad.cpu(), ci.grad))
            for (name, gp), (_, cp) in zip(model.named_parameters(), cpu.named_parameters()):
                if gp.grad is None or cp.grad is None:
                    assert gp.grad is None and cp.grad is None, name
                else:
                    pairs.append(("gradient/"+name, gp.grad.cpu(), cp.grad))
            parity = {}
            failures = []
            for name, actual, expected in pairs:
                err = (actual-expected).abs()
                parity[name] = dict(max_abs=float(err.max()),
                                    max_reference=float(expected.abs().max()))
                try:
                    torch.testing.assert_close(actual, expected, rtol=2e-3, atol=8e-5)
                except AssertionError as exc:
                    failures.append(dict(name=name, error=str(exc)))
            row["parity"] = dict(passed=not failures, comparisons=len(pairs), errors=parity, failures=failures, dropout_disabled_for_parity=dropout)
            if failures:
                raise AssertionError("CPU output/gradient parity failed: "+", ".join(f["name"] for f in failures))
            # Reset BN buffers and parameters after the verification pass.
            model.load_state_dict(cpu.state_dict())
            model.zero_grad(set_to_none=True)
            del ci, co, cl, go, gl, pairs
        else:
            row["parity"] = dict(skipped=True)
        for name, probability in dropout.items():
            model.get_submodule(name).p = probability
        row["training_dropout"] = dropout
        del cpu
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, foreach=False, amsgrad=True)
        holder = {}
        def zero():
            opt.zero_grad(set_to_none=True)
            inp.grad = None
        def forward():
            holder["output"] = model(inp)
        def loss():
            holder["loss"] = loss_fn(holder["output"], label)
        def backward():
            holder["loss"].backward()
        phases = dict(zero_grad=zero, forward=forward, loss=loss, backward=backward, optimizer=opt.step)
        def step():
            for fn in phases.values():
                fn()
            holder.clear()
        for _ in range(a.warmup):
            step()
        sync()
        if vk:
            vk.reset_execution_stats()
            qb, cb, gb = vk.queue_stats(), vk.cache_stats(), vk.gpu_timing_stats()
        if a.mode == "wall":
            train = []
            for _ in range(a.repeats):
                sync()
                start = time.perf_counter()
                step()
                sync()
                train.append((time.perf_counter()-start)*1000)
            row["training_ms"] = summary(train)
            # Read audit before result downloads or the separate inference pass.
            if vk:
                row["training_execution"] = vk.execution_stats()
                row["training_queue_delta"] = delta(qb, vk.queue_stats())
                row["training_cache_delta"] = delta(cb, vk.cache_stats())
            model.eval()
            with torch.no_grad():
                for _ in range(a.warmup):
                    model(inp)
                sync()
                if vk:
                    vk.reset_execution_stats()
                inference = []
                for _ in range(a.repeats):
                    sync()
                    start = time.perf_counter()
                    model(inp)
                    sync()
                    inference.append((time.perf_counter()-start)*1000)
            row["inference_ms"] = summary(inference)
            if vk:
                row["inference_execution"] = vk.execution_stats()
        elif a.mode == "phases":
            import nvtx
            values = {k:[] for k in phases}
            for _ in range(a.repeats):
                with nvtx.annotate("measured_step", domain="pv-profiling"):
                    for name, fn in phases.items():
                        sync()
                        with nvtx.annotate(name, domain="pv-profiling"):
                            start = time.perf_counter()
                            fn()
                            sync()
                            values[name].append((time.perf_counter()-start)*1000)
                    holder.clear()
            row["phase_ms"] = {k:summary(v) for k,v in values.items()}
        elif vk:
            for _ in range(a.repeats):
                step()
            sync()
            row["gpu_timing_delta"] = {name:delta(gb.get(name,{}),stats) for name,stats in vk.gpu_timing_stats().items() if stats.get("samples",0)>gb.get(name,{}).get("samples",0)}
        else:
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                for _ in range(a.repeats):
                    with torch.profiler.record_function("training_step"):
                        step()
                sync()
            row["kernels"] = [dict(name=e.name,duration_us=e.time_range.elapsed_us()) for e in prof.events()
                              if e.device_type == torch.autograd.DeviceType.CUDA and not e.is_user_annotation]
            row["operators"] = [dict(name=e.key,calls=e.count,self_cpu_us=e.self_cpu_time_total,self_device_us=e.self_device_time_total,device_us=e.device_time_total)
                                for e in prof.key_averages()]
        if vk and a.mode != "wall":
            row["execution"] = vk.execution_stats()
            row["queue_delta"] = delta(qb, vk.queue_stats())
            row["cache_delta"] = delta(cb, vk.cache_stats())
        row["status"] = "passed"
    except Exception:
        row["status"] = "failed"
        row["error"] = traceback.format_exc()
        if vk:
            row["failure_execution"] = vk.execution_stats()
        save()
        raise
    save()
    print(json.dumps({k:row[k] for k in ["case","backend","mode","status","training_ms","inference_ms"] if k in row}), flush=True)

if __name__ == "__main__":
    main()
