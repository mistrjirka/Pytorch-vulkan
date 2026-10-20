"""Verify one immutable Vulkan pipeline is shared by many tensor bindings.

The pinned Kompute cache patch must not recompile the same shader for newly
allocated inputs/outputs. Binding-specific descriptors must remain correct,
including when PyTorch's allocator recycles temporary tensor memory.
"""
import os
import pytest
import torch
import torch_vulkan


@pytest.mark.parametrize("allocations", [80, 600])
def test_pipeline_cache_independent_of_tensor_address(allocations):
    before = torch_vulkan.cache_stats()
    xs, ys = [], []
    for i in range(allocations):
        # Hold allocations alive so the distinct Vulkan handles cannot be
        # reused. The operator's shader, dtype, push-constant layout and
        # descriptor count are constant across all iterations.
        tensor = torch.full((32,), float(i) / 10, device="vulkan")
        result = torch.relu(tensor)
        xs.append(tensor)
        ys.append(result)
    after = torch_vulkan.cache_stats()

    if after["pipeline_misses"] == 0 and after["pipeline_hits"] == 0:
        pytest.skip("External unpatched Kompute does not expose pipeline reuse metrics")
    # This test uses two shader families: torch.full() materialization
    # and relu. On a fresh cache both may compile exactly once.
    assert after["pipeline_misses"] - before["pipeline_misses"] <= 2
    assert after["pipeline_hits"] - before["pipeline_hits"] >= allocations - 1
    assert after["pipeline_entries"] <= 256
    max_bound_algorithms = int(
        os.environ.get("TORCH_VULKAN_BOUND_ALGORITHM_CACHE_CAP", "4096"))
    assert after["bound_algorithms"] <= max_bound_algorithms
    assert after["bound_cached_buffer_bytes"] <= int(
        os.environ.get("TORCH_VULKAN_BOUND_ALGORITHM_CACHE_MB", "512")) * 1024 * 1024
    if allocations == 600 and max_bound_algorithms <= 512:
        assert after["algo_evictions"] > before["algo_evictions"]
    for index in [0, allocations // 2, allocations - 1]:
        expected = torch.full((32,), float(index) / 10)
        torch.testing.assert_close(ys[index].cpu(), expected)


def test_shared_pipeline_push_constants_and_live_buffers():
    a = torch.randn(5, 13)
    b = torch.randn(5, 13)
    av, bv = a.to("vulkan"), b.to("vulkan")
    for alpha in (0.0, 0.5, 1.0, 2.25, -0.5):
        result = torch.add(av, bv, alpha=alpha)
        torch.testing.assert_close(result.cpu(), a + alpha * b, rtol=2e-5, atol=2e-5)
    # Reallocate buffers after the old ones go out of scope. Descriptor
    # contents must follow the *current* live VkBuffers.
    for i in range(64):
        x = torch.randn(7, 11)
        y = torch.randn(7, 11)
        actual = torch.add(x.to("vulkan"), y.to("vulkan"), alpha=1.13)
        torch.testing.assert_close(actual.cpu(), x + 1.13 * y, rtol=3e-5, atol=3e-5)


def test_convolution_pipeline_reused_for_different_kernel_sizes():
    import os
    import torch.nn.functional as F
    old=os.environ.get('TORCH_VULKAN_CONV_ALGO')
    os.environ['TORCH_VULKAN_CONV_ALGO']='tiled'
    try:
        before=torch_vulkan.cache_stats()
        live_buffers=[]
        for k in [1,3,5,7,3,5,1,7]:
            torch.manual_seed(100+k)
            inp=torch.randn(1,8,12,13)
            weight=torch.randn(16,8,k,k)
            expected=F.conv2d(inp,weight,padding=k//2)
            inp_vk,weight_vk=inp.to('vulkan'),weight.to('vulkan')
            got=F.conv2d(inp_vk,weight_vk,padding=k//2)
            # Keep bindings distinct: reusing an entire bound algorithm would
            # bypass the shared-pipeline lookup this test is measuring.
            live_buffers.append((inp_vk,weight_vk,got))
            torch.testing.assert_close(got.cpu(),expected,atol=4e-4,rtol=4e-4)
        after=torch_vulkan.cache_stats()
        if after['pipeline_hits'] or after['pipeline_misses']:
            assert after['pipeline_misses']-before['pipeline_misses']<=1
            assert after['pipeline_hits']-before['pipeline_hits']>=7
    finally:
        if old is None:os.environ.pop('TORCH_VULKAN_CONV_ALGO',None)
        else:os.environ['TORCH_VULKAN_CONV_ALGO']=old
