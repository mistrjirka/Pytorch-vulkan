"""Regression coverage for parallel integer scatter, including TorchBench U-Net one_hot."""
import pytest
import torch
import torch.nn.functional as F
import torch_vulkan as vk


@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_integer_scatter_duplicates_and_add(dtype):
    # Scatter overwrite must preserve the last source along each axis.
    x = torch.tensor([[5, 6, 7, 8], [9, 10, 11, 12]], dtype=dtype)
    if dtype == torch.int64:
        x = x * (2**38) - 13
    indices = torch.tensor([[1, 1, 0, 3], [2, 0, 2, 2]], dtype=torch.int64)
    source = torch.tensor([[12, -7, 99, 41], [51, 31, 67, -22]], dtype=dtype)
    xv, iv, sv = x.to("vulkan"), indices.to("vulkan"), source.to("vulkan")
    vk.reset_execution_stats()
    actual = [
        xv.scatter(1, iv, sv),
        xv.scatter_add(1, iv, sv),
        xv.scatter(1, iv, -5),
    ]
    assert vk.execution_stats()["cpu_fallback_calls"] == 0
    expected = [
        x.scatter(1, indices, source),
        x.scatter_add(1, indices, source),
        x.scatter(1, indices, -5),
    ]
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a.cpu(), b)


@pytest.mark.parametrize("classes", [2, 4])
def test_one_hot_spatial_integer_scatter(classes):
    # Representative large image segmentation: no single-thread O(H*W) scatter.
    idx = torch.arange(128 * 191, dtype=torch.long).reshape(1, 128, 191) % classes
    vk.reset_execution_stats()
    actual = F.one_hot(idx.to("vulkan"), num_classes=classes)
    stats = vk.execution_stats()
    assert stats["cpu_fallback_calls"] == 0, stats
    torch.testing.assert_close(actual.cpu(), F.one_hot(idx, num_classes=classes))
