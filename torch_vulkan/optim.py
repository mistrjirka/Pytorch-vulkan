"""Experimental fused Vulkan AdamW: equivalent math, one dispatch per tensor.

This optimizer is opt-in; standard torch.optim.AdamW is not monkey-patched.
Only contiguous float32 Vulkan tensors are supported. Sparse gradients,
maximize, graph capture, per-parameter tensor learning rates and
master-weight mixed precision are not implemented.
"""
from __future__ import annotations

import torch


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9,0.999), eps=1e-8,
                 weight_decay=1e-2, batched=True, amsgrad=False):
        if lr < 0 or eps < 0 or weight_decay < 0 or not all(0 <= x < 1 for x in betas):
            raise ValueError('Invalid AdamW hyperparameters')
        super().__init__(params, dict(lr=lr,betas=betas,eps=eps,weight_decay=weight_decay,
                                      batched=batched,amsgrad=amsgrad))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1,beta2 = group['betas']
            params, grads, moments1, moments2, max_moments, steps = [],[],[],[],[],[]
            for p in group['params']:
                if p.grad is None:continue
                if p.grad.is_sparse:
                    raise RuntimeError('Vulkan AdamW does not support sparse gradients')
                if p.device.type != 'vulkan':
                    raise RuntimeError('Vulkan AdamW only supports Vulkan parameters')
                state = self.state[p]
                if not state:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    if group['amsgrad']:
                        state['max_exp_avg_sq'] = torch.zeros_like(p)
                # A restored checkpoint from native torch.optim.AdamW may
                # encode the step as a scalar Tensor rather than an integer.
                if isinstance(state['step'],torch.Tensor):
                    state['step'] = int(state['step'].item())
                state['step'] += 1
                params.append(p)
                grads.append(p.grad)
                moments1.append(state['exp_avg'])
                moments2.append(state['exp_avg_sq'])
                if group['amsgrad']:
                    max_moments.append(state['max_exp_avg_sq'])
                steps.append(state['step'])
            if not params:
                continue
            if group['amsgrad']:
                if group.get('batched',True):
                    torch.ops.torch_vulkan.adamw_amsgrad_step_batch_(
                        params,grads,moments1,moments2,max_moments,
                        group['lr'],beta1,beta2,group['eps'],group['weight_decay'],steps)
                else:
                    for p,g,m,v,mx,step in zip(params,grads,moments1,moments2,max_moments,steps):
                        torch.ops.torch_vulkan.adamw_amsgrad_step_batch_(
                            [p],[g],[m],[v],[mx],group['lr'],beta1,beta2,
                            group['eps'],group['weight_decay'],[step])
            elif group.get('batched',True):
                torch.ops.torch_vulkan.adamw_step_batch_(
                    params, grads, moments1, moments2,
                    group['lr'], beta1,beta2,group['eps'],group['weight_decay'],steps)
            else:
                for p,g,m,v,step in zip(params,grads,moments1,moments2,steps):
                    torch.ops.torch_vulkan.adamw_step_(
                        p,g,m,v,group['lr'],beta1,beta2,group['eps'],group['weight_decay'],step)
        return loss
