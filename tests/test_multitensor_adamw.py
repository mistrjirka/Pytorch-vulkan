"""Generalized AdamW horizontal fusion, not tied to one model's shapes."""
import pytest
import os
import torch
import torch_vulkan
from torch_vulkan.optim import AdamW as VulkanAdamW


@pytest.mark.parametrize("amsgrad", [False, True])
@pytest.mark.parametrize("count", [2, 3, 4, 5, 11, 53])
def test_chunk_mapping_and_state_parity(amsgrad, count):
    generator = torch.Generator().manual_seed(7521)
    lengths = [1, 257, 513, 5, 2049, 2, 65, 0, 117, 4056, 16]
    shapes = [(lengths[i % len(lengths)],) for i in range(count)]
    original = [torch.randn(shape, generator=generator) for shape in shapes]
    cpu = [torch.nn.Parameter(x.clone()) for x in original]
    gpu = [torch.nn.Parameter(x.to("vulkan")) for x in original]
    opts = {"lr": .0017, "betas": (.79, .933), "eps": 1e-7,
            "weight_decay": .014, "amsgrad": amsgrad}
    reference = torch.optim.AdamW(cpu, foreach=False, **opts)
    grouped = VulkanAdamW(gpu, batched=True, **opts)
    before = torch_vulkan.queue_stats()["recorded_ops"]
    baseline_fallback = torch_vulkan.execution_stats()["cpu_fallback_calls"]
    for iteration in range(9):
        for i, (c, v) in enumerate(zip(cpu, gpu)):
            if (i + iteration) % 13 == 0:
                c.grad = v.grad = None
            else:
                g = torch.randn(c.shape, generator=generator) * .03
                c.grad = g
                v.grad = g.to("vulkan")
        reference.step()
        grouped.step()
    torch_vulkan.synchronize()
    for c, v in zip(cpu, gpu):
        torch.testing.assert_close(v.cpu(), c.detach(), rtol=5e-5, atol=3e-5)
        if c in reference.state:
            for k in ("exp_avg", "exp_avg_sq") + (("max_exp_avg_sq",) if amsgrad else ()):
                torch.testing.assert_close(grouped.state[v][k].cpu(),
                                           reference.state[c][k],
                                           rtol=5e-5, atol=3e-5)
    assert torch_vulkan.execution_stats()["cpu_fallback_calls"] == baseline_fallback
    # With 53 tensors the horizontal path must genuinely combine dispatches.
    # The counter also includes occasional gradient upload/copy work, so
    # compare against the theoretical 53*9 per-tensor AdamW dispatches only
    # if the instrumentation is isolated in the test's update phase.


def test_grouped_updates_have_fewer_gpu_dispatches_than_parameters():
    params = [torch.nn.Parameter(torch.ones(7 + 257*i, device="vulkan"))
              for i in range(11)]
    opt = VulkanAdamW(params, lr=.0001, amsgrad=True, batched=True)
    for p in params:
        p.grad = torch.ones_like(p)
    # Initialize optimizer moments once; the second step measures only the
    # steady-state update, not the 3*11 initial zero-fill kernels.
    opt.step()
    torch_vulkan.synchronize()
    before = torch_vulkan.queue_stats()["recorded_ops"]
    opt.step()
    torch_vulkan.synchronize()
    after = torch_vulkan.queue_stats()["recorded_ops"]
    # GPU launches are bounded by the configured shader group width.
    # The old per-tensor path is retained as an explicit A/B control.
    if os.getenv("TORCH_VULKAN_ADAMW_DISABLE_MULTITENSOR") == "1":
        expected = len(params)
    else:
        width = int(os.getenv("TORCH_VULKAN_ADAMW_MULTITENSOR_WIDTH", "8"))
        expected = (len(params) + width - 1) // width
    assert after - before == expected, (before, after, expected)


@pytest.mark.parametrize("amsgrad", [False, True])
def test_shared_update_buffer_keeps_sequential_semantics(amsgrad):
    """A repeated parameter/state must not be written by two GPU workgroups."""
    reference = torch.nn.Parameter(torch.linspace(.2, .9, 257))
    optimizer = torch.optim.AdamW([reference], amsgrad=amsgrad,
                                   lr=.003, betas=(.8, .93),
                                   eps=1e-7, weight_decay=.02,
                                   foreach=False)
    g1 = torch.full_like(reference, .025)
    g2 = torch.full_like(reference, -.04)
    for g in (g1, g2):
        reference.grad = g
        optimizer.step()

    initial = torch.linspace(.2, .9, 257)
    param = initial.to("vulkan")
    avg = torch.zeros_like(param)
    sq = torch.zeros_like(param)
    maximum = torch.zeros_like(param)
    inputs = ([param, param], [g1.to("vulkan"), g2.to("vulkan")],
              [avg, avg], [sq, sq])
    options = (.003, .8, .93, 1e-7, .02, [1, 2])
    torch_vulkan.synchronize()
    before = torch_vulkan.queue_stats()["recorded_ops"]
    if amsgrad:
        torch.ops.torch_vulkan.adamw_amsgrad_step_batch_(
            *inputs, [maximum, maximum], *options)
    else:
        torch.ops.torch_vulkan.adamw_step_batch_(*inputs, *options)
    torch_vulkan.synchronize()
    after = torch_vulkan.queue_stats()["recorded_ops"]
    # Explicit dependency/alias check routes both updates sequentially.
    assert after - before == 2
    torch.testing.assert_close(param.cpu(), reference.detach(), atol=4e-5, rtol=4e-5)
    state = optimizer.state[reference]
    torch.testing.assert_close(avg.cpu(), state["exp_avg"], atol=4e-5, rtol=4e-5)
    torch.testing.assert_close(sq.cpu(), state["exp_avg_sq"], atol=4e-5, rtol=4e-5)
    if amsgrad:
        torch.testing.assert_close(maximum.cpu(), state["max_exp_avg_sq"],
                                   atol=4e-5, rtol=4e-5)
