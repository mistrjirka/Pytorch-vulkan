"""Descriptor binding lifetimes under deferred execution and cache eviction.

All tests use actual Vulkan buffers, not a mocked cache. In particular,
queued OpAlgoDispatch objects must keep the owning Algorithm's private
descriptor set alive until the GPU has completed the recorded dispatch.
"""
import os
import torch
import torch_vulkan


def test_descriptor_reuse_with_live_outputs_and_eviction():
    torch.manual_seed(13024)
    left = torch.linspace(-2, 2, 273)
    right = torch.linspace(0.7, 1.3, 273)
    a, b = left.to("vulkan"), right.to("vulkan")
    before = torch_vulkan.execution_stats()["cpu_fallback_calls"]
    results = []
    expected = []
    # More unique output descriptors than the smallest supported cache cap.
    # Results remain live so their VkBuffers must never be rebound incorrectly.
    for i in range(360):
        factor = ((i % 13) - 6) * .035
        t = torch.add(a, b, alpha=factor)
        u = torch.tanh(t)
        results.append(u)
        expected.append(torch.tanh(left + factor * right))
    torch_vulkan.synchronize()
    for i in range(0, len(results), 11):
        torch.testing.assert_close(results[i].cpu(), expected[i],
                                   rtol=3e-5, atol=3e-5)
    assert torch_vulkan.execution_stats()["cpu_fallback_calls"] == before
    stats = torch_vulkan.cache_stats()
    assert stats["bound_algorithms"] <= int(
        os.environ.get("TORCH_VULKAN_BOUND_ALGORITHM_CACHE_CAP", "4096"))
    assert stats["bound_cached_buffer_bytes"] <= int(
        os.environ.get("TORCH_VULKAN_BOUND_ALGORITHM_CACHE_MB", "512")) * 1024 * 1024


def test_descriptor_reuse_with_short_lived_intermediates():
    # Lifetime pressure with fresh temporary outputs in a not-yet-submitted
    # command buffer. No tensor may be rebound to the next operation while
    # a previous command could still read or write the same buffer.
    x = torch.arange(97, dtype=torch.float32)
    expected = x
    actual = x.to("vulkan")
    for i in range(72):
        scale = 0.5 + (i % 5) * .15
        expected = torch.relu(expected * scale + 0.15)
        actual = torch.relu(actual * scale + 0.15)
    torch_vulkan.synchronize()
    torch.testing.assert_close(actual.cpu(), expected, atol=3e-5, rtol=3e-5)


def test_descriptor_bound_cache_honors_live_buffer_budget():
    cap = int(os.environ.get("TORCH_VULKAN_BOUND_ALGORITHM_CACHE_MB", "512"))
    initial = torch_vulkan.cache_stats()
    # Different large live buffers can stress per-byte cache accounting
    # independently of the count of descriptor sets.
    tensors = [torch.full((16384,), i / 100., device="vulkan")
               for i in range(46)]
    outputs = [torch.relu(t) for t in tensors]
    torch_vulkan.synchronize()
    for i in (0, 13, 45):
        torch.testing.assert_close(outputs[i].cpu(),
                                   torch.full((16384,), i / 100.))
    stats = torch_vulkan.cache_stats()
    assert stats["bound_cached_buffer_bytes"] <= cap * 1024 * 1024
    if cap == 0:
        # Note: this is process startup config, not a monkeypatched runtime flag.
        assert stats["bound_algorithms"] == 0
