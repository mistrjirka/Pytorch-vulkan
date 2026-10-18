"""Two ordinary AdamW steps; trace setup separately from the update primitives."""
import json
import sys
from pathlib import Path
import torch
import torch_vulkan

torch.manual_seed(191)
cpu = torch.randn(5, 7, requires_grad=True)
vk = cpu.detach().to('vulkan').requires_grad_()
a = torch.optim.AdamW([cpu], lr=.003, foreach=False)
b = torch.optim.AdamW([vk], lr=.003, foreach=False)
for step in range(2):
    grad = torch.randn_like(cpu)
    cpu.grad, vk.grad = grad, grad.to('vulkan')
    torch_vulkan.synchronize()
    print(f'UPDATE_BEGIN {step}', file=sys.stderr, flush=True)
    a.step(); b.step()
    torch_vulkan.synchronize()
    print(f'UPDATE_END {step}', file=sys.stderr, flush=True)
    torch.testing.assert_close(vk.cpu(), cpu, rtol=2e-5, atol=2e-6)
    for key in ('exp_avg', 'exp_avg_sq'):
        torch.testing.assert_close(b.state[vk][key].cpu(), a.state[cpu][key], rtol=2e-5, atol=2e-6)
Path(sys.argv[1]).write_text(json.dumps({'steps':2,'parameter_and_moment_parity':True,'torch':torch.__version__,
                                      'device':torch_vulkan.device_name()},indent=2)+'\n')
