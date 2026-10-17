"""GPU-executed FP32 Conv2D forward and all three backward gradients."""
import pytest
import torch
import torch.nn.functional as F
import torch_vulkan


@pytest.mark.parametrize('shape,channels,kernel,stride,padding,dilation,groups,bias', [
    ((2, 3, 9, 11), 4, 3, 1, 1, 1, 1, True),
    ((1, 2, 11, 12), 3, 3, 2, 1, 1, 1, False),
    ((2, 4, 8, 7), 6, 3, 1, 2, 2, 2, True),
    ((1, 8, 8, 8), 8, 1, 1, 0, 1, 8, True),
])
def test_fp32_conv2d_forward_backward_native(shape,channels,kernel,stride,padding,dilation,groups,bias):
    assert torch._C._dispatch_has_kernel_for_dispatch_key(
        'aten::convolution_overrideable', 'PrivateUse1'
    )
    assert torch._C._dispatch_has_kernel_for_dispatch_key(
        'aten::convolution_backward_overrideable', 'PrivateUse1'
    )
    torch.manual_seed(72)
    inp = torch.randn(*shape, requires_grad=True)
    weight = torch.randn(channels,shape[1]//groups,kernel,kernel,requires_grad=True)
    b = torch.randn(channels,requires_grad=True) if bias else None
    out = F.conv2d(inp,weight,b,stride,padding,dilation,groups)
    grad = torch.randn_like(out)
    refs = (inp,weight,b) if b is not None else (inp,weight)
    cpu_grad = torch.autograd.grad(out,refs,grad_outputs=grad)

    x_vk = inp.detach().to('vulkan').requires_grad_()
    w_vk = weight.detach().to('vulkan').requires_grad_()
    b_vk = b.detach().to('vulkan').requires_grad_() if b is not None else None
    out_vk = F.conv2d(x_vk,w_vk,b_vk,stride,padding,dilation,groups)
    assert out_vk.device.type == 'vulkan'
    torch.testing.assert_close(out_vk.cpu(),out,rtol=2e-4,atol=3e-4)
    vg = (x_vk,w_vk,b_vk) if b_vk is not None else (x_vk,w_vk)
    gradients = torch.autograd.grad(out_vk,vg,grad_outputs=grad.to('vulkan'))
    for actual,expected in zip(gradients,cpu_grad):
        assert actual.device.type == 'vulkan'
        torch.testing.assert_close(actual.cpu(),expected,rtol=3e-4,atol=3e-4)


def test_conv_layer_sgd_decreases_loss():
    torch.manual_seed(7)
    m = torch.nn.Conv2d(1,2,3,padding=1).to('vulkan')
    x = torch.randn(2,1,8,8).to('vulkan')
    target = torch.randn(2,2,8,8).to('vulkan')
    optimizer = torch.optim.SGD(m.parameters(),lr=0.01)
    losses=[]
    for i in range(3):
        optimizer.zero_grad()
        loss = F.mse_loss(m(x),target)
        losses.append(loss.item())
        loss.backward()
        assert m.weight.grad is not None
        optimizer.step()
    assert losses[-1] < losses[0]
