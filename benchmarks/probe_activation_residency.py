"""Diagnostic only: compare CPU-fallback LeakyReLU with finite-input GPU composition.

Preserves the model architecture and slope. The composition uses existing ops:
s*x + (1-s)*relu(x). Extra rounding and inf/NaN behavior mean it is NOT a general
replacement kernel. A custom backward uses the existing threshold kernel to
avoid an unrelated autograd accumulation fallback for the two forward branches.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import types
import torch
import torch_vulkan
from torch_vulkan.optim import AdamW
from compare_cuda_vulkan import CONFIG, data_for_case


class FiniteLeaky(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, slope):
        ctx.save_for_backward(x)
        ctx.slope = slope
        return slope*x + (1-slope)*x.relu()

    @staticmethod
    def backward(ctx, grad):
        x, = ctx.saved_tensors
        positive = torch.ops.aten.threshold_backward(grad, x, 0)
        return ctx.slope*grad + (1-ctx.slope)*positive, None


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--variant',choices=['stock','composite'],required=True)
    p.add_argument('--model',choices=CONFIG,default='style_net')
    p.add_argument('--zoo',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--parity-out',type=Path,required=True)
    p.add_argument('--warmup',type=int,default=8)
    p.add_argument('--repeats',type=int,default=15)
    a=p.parse_args()
    if a.warmup < 1 or a.repeats < 1:
        p.error("warmup and repeats must be positive")
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    sys.path.insert(0,str((a.zoo/'benchmarks').resolve()))
    import model_zoo
    torch.manual_seed(42)
    model,_,loss_fn=model_zoo.MODELS[a.model]()
    count=0
    for m in model.modules():
        if isinstance(m,torch.nn.LeakyReLU):
            assert not m.inplace
            count+=1
            if a.variant=='composite':
                def forward(self,x):return FiniteLeaky.apply(x,self.negative_slope)
                m.forward=types.MethodType(forward,m)
    model.to('vulkan').train()
    opt=AdamW(model.parameters(),lr=1e-4)
    x,y=data_for_case(a.model,CONFIG[a.model]['medium'],torch.Generator().manual_seed(123))
    x,y=x.to('vulkan'),y.to('vulkan')
    # One untouched-weight forward/backward comparison, outside timed iterations.
    output=model(x);loss=loss_fn(output,y);loss.backward();torch_vulkan.synchronize()
    parity=dict(output=output.detach().cpu(),loss=loss.detach().cpu(),
                gradients={n:p.grad.cpu() for n,p in model.named_parameters() if p.grad is not None})
    a.parity_out.parent.mkdir(parents=True,exist_ok=True);torch.save(parity,a.parity_out)
    del output,loss,parity
    samples=[];before={};after={}
    for i in range(a.warmup+a.repeats):
        torch_vulkan.synchronize()
        if i==a.warmup:before=torch_vulkan.queue_stats()
        t=time.perf_counter()
        opt.zero_grad(set_to_none=True)
        loss=loss_fn(model(x),y);loss.backward();opt.step();torch_vulkan.synchronize()
        if i>=a.warmup:samples.append((time.perf_counter()-t)*1000)
    after=torch_vulkan.queue_stats()
    report=dict(revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                model=a.model,shape=CONFIG[a.model]['medium'],variant=a.variant,
                leaky_modules=count,device=torch_vulkan.device_name(),torch=torch.__version__,
                warmup=a.warmup,repeats=a.repeats,median_ms=statistics.median(samples),
                min_ms=min(samples),max_ms=max(samples),samples_ms=samples,
                queue_per_step={k:(after[k]-before.get(k,0))/a.repeats for k in ['submissions','recorded_ops']},
                env={k:v for k,v in os.environ.items() if k.startswith('TORCH_VULKAN_')},
                note='Diagnostic finite-input composition, not a published backend kernel; one parity pass precedes warmup.')
    a.out.write_text(json.dumps(report,indent=2)+'\n');print(a.variant,report['median_ms'],report['queue_per_step'],flush=True)

if __name__=='__main__':main()
