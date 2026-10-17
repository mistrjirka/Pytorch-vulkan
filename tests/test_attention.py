"""Regression tests for safe attention and Transformer training.

Do not re-enable the old Vulkan scaled_dot_product_attention shader without
an explicit autograd implementation and numerical validation.
"""
from __future__ import annotations

import copy

import pytest
import torch
import torch.nn.functional as F
import torch_vulkan


def assert_matching(actual, expected, *, atol=2e-5, rtol=2e-5):
    assert actual.device.type == "vulkan"
    torch.testing.assert_close(actual.detach().cpu(), expected.detach(), atol=atol, rtol=rtol)


@pytest.mark.parametrize("shape", [(1, 2, 8, 16), (2, 4, 24, 32)])
def test_scaled_dot_product_attention_uses_safe_autograd_fallback(shape):
    # The custom fused shader is currently disabled because its forward result
    # was wrong and its gradient implementation was missing.
    assert not torch._C._dispatch_has_kernel_for_dispatch_key(
        "aten::scaled_dot_product_attention", "PrivateUse1"
    )
    torch.manual_seed(102)
    cpu_inputs = tuple(torch.randn(*shape, requires_grad=True) for _ in range(3))
    vk_inputs = tuple(t.detach().to("vulkan").requires_grad_() for t in cpu_inputs)
    expected = F.scaled_dot_product_attention(*cpu_inputs, dropout_p=0.0)
    actual = F.scaled_dot_product_attention(*vk_inputs, dropout_p=0.0)
    assert_matching(actual, expected, atol=6e-6, rtol=6e-6)
    grad = torch.randn_like(expected)
    ref_grads = torch.autograd.grad(expected, cpu_inputs, grad_outputs=grad)
    vk_grads = torch.autograd.grad(actual, vk_inputs, grad_outputs=grad.to("vulkan"))
    for ours, reference in zip(vk_grads, ref_grads):
        assert_matching(ours, reference, atol=8e-6, rtol=8e-6)


def test_transformer_encoder_layer_forward_backward():
    torch.manual_seed(101)
    cpu = torch.nn.TransformerEncoderLayer(
        d_model=32, nhead=4, dim_feedforward=64,
        dropout=0.0, batch_first=True, norm_first=True,
        activation="gelu",
    )
    vk = copy.deepcopy(cpu).to("vulkan")
    x_cpu = torch.randn(2, 12, 32, requires_grad=True)
    x_vk = x_cpu.detach().to("vulkan").requires_grad_()
    expected = cpu(x_cpu)
    actual = vk(x_vk)
    assert_matching(actual, expected, atol=7e-5, rtol=7e-5)
    grad = torch.randn_like(expected)
    ref = torch.autograd.grad(expected, (x_cpu, *cpu.parameters()), grad_outputs=grad)
    got = torch.autograd.grad(actual, (x_vk, *vk.parameters()), grad_outputs=grad.to("vulkan"))
    for ours, reference in zip(got, ref):
        assert_matching(ours, reference, atol=1e-4, rtol=1e-4)
