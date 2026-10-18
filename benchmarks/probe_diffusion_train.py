"""Diffusion-style FP32 PyTorch training compatibility and latency smoke test.

A small unconditional denoising U-Net approximating common Diffusers ResNet
blocks: sinusoidal timestep embeddings, SiLU, GroupNorm, convolutional
downsample/upsample, skip concatenation, and an optional self-attention
bottleneck via SDPA. Uses only stock torch.nn operations, not model-specific
Vulkan kernels. This is NOT a full Hugging Face Diffusers UNet2DModel.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import pathlib
import statistics
import subprocess
import time
import traceback

import torch
from torch import nn
from torch.nn import functional as F


class ResBlock(nn.Module):
    def __init__(self,in_channels,out_channels,tdim=16):
        super().__init__()
        self.norm1=nn.GroupNorm(4,in_channels)
        self.act=nn.SiLU()
        self.conv1=nn.Conv2d(in_channels,out_channels,3,padding=1)
        self.time_proj=nn.Linear(tdim,out_channels)
        self.norm2=nn.GroupNorm(4,out_channels)
        self.dropout=nn.Dropout(p=0.0)
        self.conv2=nn.Conv2d(out_channels,out_channels,3,padding=1)
        self.skip=nn.Conv2d(in_channels,out_channels,1) if in_channels!=out_channels else nn.Identity()
    def forward(self,x,temb):
        h=self.conv1(self.act(self.norm1(x)))
        h=h+self.time_proj(temb)[:,:,None,None]
        h=self.conv2(self.dropout(self.act(self.norm2(h))))
        return h+self.skip(x)


class SelfAttention2D(nn.Module):
    def __init__(self,channels=16,heads=2):
        super().__init__()
        self.norm=nn.GroupNorm(4,channels)
        self.qkv=nn.Linear(channels,channels*3)
        self.output=nn.Linear(channels,channels)
        self.heads=heads
    def forward(self,x):
        b,c,h,w=x.shape
        norm=self.norm(x)
        tokens=norm.flatten(2).transpose(1,2)
        q,k,v=self.qkv(tokens).chunk(3,dim=-1)
        q=q.reshape(b,h*w,self.heads,c//self.heads).transpose(1,2)
        k=k.reshape(b,h*w,self.heads,c//self.heads).transpose(1,2)
        v=v.reshape(b,h*w,self.heads,c//self.heads).transpose(1,2)
        attention=F.scaled_dot_product_attention(q,k,v,dropout_p=0.0)
        attn=attention.transpose(1,2).reshape(b,h*w,c)
        out=self.output(attn).transpose(1,2).reshape(b,c,h,w)
        return x+out


class TinyDiffusionUNet(nn.Module):
    def __init__(self,attention=False,interpolation="nearest"):
        super().__init__()
        self.interpolation=interpolation
        self.time_nn=nn.Sequential(nn.Linear(16,16),nn.SiLU(),nn.Linear(16,16))
        self.input=nn.Conv2d(3,8,3,padding=1)
        self.block1=ResBlock(8,8)
        self.down=nn.Conv2d(8,16,3,stride=2,padding=1)
        self.block_mid=ResBlock(16,16)
        self.attn=SelfAttention2D(16) if attention else nn.Identity()
        self.block_up=ResBlock(24,8)
        self.output=nn.Conv2d(8,3,3,padding=1)
    def forward(self,noisy,timesteps):
        freq=torch.arange(8,dtype=torch.float32,device=timesteps.device)
        phases=timesteps[:,None]*torch.exp(freq*-0.15)[None,:]
        time_embedding=torch.cat([torch.sin(phases),torch.cos(phases)],dim=-1)
        time_embedding=self.time_nn(time_embedding)
        h=self.input(noisy)
        skip=self.block1(h,time_embedding)
        h=self.down(skip)
        h=self.block_mid(h,time_embedding)
        h=self.attn(h)
        if self.interpolation=="bilinear":
            h=F.interpolate(h,size=skip.shape[-2:],mode="bilinear",align_corners=False)
        else:
            h=F.interpolate(h,size=skip.shape[-2:],mode="nearest")
        h=torch.cat([h,skip],dim=1)
        h=self.block_up(h,time_embedding)
        return self.output(h)


def sync(backend):
    if backend=="cuda":torch.cuda.synchronize()
    if backend=="vulkan":
        import torch_vulkan
        torch_vulkan.synchronize()


def cpu_parity(model, clean, noise, timesteps, device, inp, noise_dev, time_dev):
    cpu_x=clean.detach().clone().requires_grad_()
    cpu_model=copy.deepcopy(model).train()
    prediction=cpu_model(cpu_x,timesteps)
    ref_loss=F.mse_loss(prediction,noise)
    ref_loss.backward()
    gpu_x=inp.detach().clone().requires_grad_()
    gpu_model=copy.deepcopy(model).to(device).train()
    gpu_pred=gpu_model(gpu_x,time_dev)
    gpu_loss=F.mse_loss(gpu_pred,noise_dev)
    gpu_loss.backward()
    sync(device)
    p_err=(prediction.detach()-gpu_pred.detach().cpu()).abs().max().item()
    x_err=(cpu_x.grad-gpu_x.grad.detach().cpu()).abs().max().item()
    grad_err=0
    param_name=""
    for (n,a),(_,b) in zip(cpu_model.named_parameters(),gpu_model.named_parameters()):
        if a.grad is None or b.grad is None:continue
        d=(a.grad.detach()-b.grad.detach().cpu()).abs().max().item()
        if d>grad_err:grad_err=d;param_name=n
    return dict(pred_max_abs=p_err,input_grad_max_abs=x_err,
                param_grad_max_abs=grad_err,largest_param_grad_error=param_name,
                ref_loss=float(ref_loss.detach()),device_loss=float(gpu_loss.detach().cpu()))


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--backend",choices=["vulkan","cuda","cpu"],required=True)
    p.add_argument("--attention",action="store_true")
    p.add_argument("--interpolation",choices=["nearest","bilinear"],default="nearest")
    p.add_argument("--size",type=int,default=16)
    p.add_argument("--batch",type=int,default=1)
    p.add_argument("--repeats",type=int,default=4)
    p.add_argument("--warmup",type=int,default=2)
    p.add_argument("--out",type=pathlib.Path)
    p.add_argument("--skip-parity",action="store_true")
    args=p.parse_args()
    torch.set_num_threads(6)
    torch.manual_seed(2026)
    start=time.perf_counter()
    config={k:v for k,v in vars(args).items() if k!="out"}
    report=dict(backend=args.backend,config=config,
                vulkan_controls={k:os.environ.get(k,"default") for k in (
                    "TORCH_VULKAN_GPU_RESIDENT", "TORCH_VULKAN_DEFER_SUBMISSIONS",
                    "TORCH_VULKAN_DISABLE_NATIVE_GROUP_NORM",
                    "TORCH_VULKAN_DISABLE_NATIVE_DIFFUSION_UNARY",
                    "TORCH_VULKAN_DISABLE_NATIVE_BMM")},
                git_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),
                torch_version=torch.__version__,status="started",
                activations_note="SiLU/GroupNorm/timestep sin-cos/optional SDPA"
                )
    try:
        gen=torch.Generator().manual_seed(123)
        clean=torch.randn(args.batch,3,args.size,args.size,generator=gen)
        noise=torch.randn(args.batch,3,args.size,args.size,generator=gen)
        t=torch.rand(args.batch,generator=gen)
        # Synthetic diffusion noise corruption; the input is identical on CPU,
        # CUDA and Vulkan, so timing isolates forward/loss/backward/optimizer.
        noisy=0.75*clean+0.25*noise
        model=TinyDiffusionUNet(args.attention,args.interpolation)
        params=sum(p.numel() for p in model.parameters())
        if args.backend=="vulkan":
            import torch_vulkan
            assert torch_vulkan.is_hardware_accelerated()
            gpu_name=torch_vulkan.device_name()
            from torch_vulkan.optim import AdamW as GPUAdamW
        elif args.backend=="cuda":
            gpu_name=torch.cuda.get_device_name()
        else:gpu_name="CPU"
        inp=noisy.to(args.backend)
        target=noise.to(args.backend)
        timesteps=t.to(args.backend)
        if not args.skip_parity and args.backend!="cpu":
            report["parity"]=cpu_parity(model,noisy,noise,t,args.backend,inp,target,timesteps)
            print("PARITY",json.dumps(report["parity"]),flush=True)
        samples={}
        for mode in ["inference","training"]:
            network=copy.deepcopy(model).to(args.backend)
            network.train(mode=="training")
            if mode=="training":
                if args.backend=="vulkan": optimizer=GPUAdamW(network.parameters(),lr=1e-4)
                else:optimizer=torch.optim.AdamW(network.parameters(),lr=1e-4,foreach=False)
            durations=[]
            lossvalue=None
            for iteration in range(args.warmup+args.repeats):
                sync(args.backend)
                tic=time.perf_counter()
                if mode=="inference":
                    with torch.inference_mode():
                        output=network(inp,timesteps)
                else:
                    optimizer.zero_grad(set_to_none=True)
                    output=network(inp,timesteps)
                    loss=F.mse_loss(output,target)
                    loss.backward()
                    optimizer.step()
                    if iteration==0:lossvalue=float(loss.detach().cpu())
                sync(args.backend)
                ms=(time.perf_counter()-tic)*1000
                if iteration>=args.warmup:durations.append(ms)
            samples[mode]=dict(median_ms=round(statistics.median(durations),3),
                               min_ms=round(min(durations),3),
                               max_ms=round(max(durations),3),
                               samples_ms=[round(x,3) for x in durations],
                               first_loss=lossvalue)
            print(mode,json.dumps(samples[mode]),flush=True)
        report.update(status="ok",device_name=gpu_name,parameters=params,modes=samples)
    except BaseException as exc:
        report.update(status="error",message=str(exc)[:1100],type=type(exc).__name__,
                      traceback=traceback.format_exc()[-3500:])
        print("ERROR",report["message"],flush=True)
        traceback.print_exc()
    report["elapsed_s"]=round(time.perf_counter()-start,3)
    if args.out:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(report,indent=2)+"\n")
    print("FINAL_STATUS",report["status"],flush=True)


if __name__=="__main__":main()
