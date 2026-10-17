"""Tiny 2.14.1 training sample. Vulkan mm/backward, CPU fallback for unsupported ops."""
import torch
import torch_vulkan

print('torch:', torch.__version__)
print('driver:', torch_vulkan.device_name())
print('physical GPU:', torch_vulkan.is_hardware_accelerated())

x = torch.tensor([[1., 2.], [2., 1.], [3., 4.]]).to('vulkan')
y = torch.tensor([[1.], [2.], [3.]]).to('vulkan')
w = torch.nn.Parameter(torch.zeros(2, 1, device='vulkan'))
optim = torch.optim.SGD([w], lr=0.01)
for step in range(5):
    optim.zero_grad()
    loss = torch.nn.functional.mse_loss(x @ w, y)
    loss.backward()
    optim.step()
    print(f'step {step}: loss = {loss.detach().cpu().item():.6f}')
