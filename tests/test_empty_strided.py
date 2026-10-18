"""Native allocation must preserve layouts without CPU factory fallback."""
import pytest
import torch
import torch_vulkan


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64, torch.int64, torch.int32, torch.bool])
@pytest.mark.parametrize('shape,stride', [((3, 5), (5, 1)), ((3, 5), (1, 3)),
    ((3, 5), (17, 2)), ((0, 5), (17, 2)), ((), ()), ((3, 5), (0, 1)),
    ((1,) * 8 + (3,), (3,) * 8 + (1,))])
def test_empty_strided_layout_and_storage(dtype, shape, stride):
    cpu = torch.empty_strided(shape, stride, dtype=dtype)
    gpu = torch.empty_strided(shape, stride, dtype=dtype, device='vulkan')
    assert gpu.shape == cpu.shape
    assert gpu.stride() == cpu.stride()
    assert gpu.untyped_storage().nbytes() == cpu.untyped_storage().nbytes()
    # A flat storage view exercises all allocated elements, including holes.
    size = cpu.untyped_storage().nbytes() // cpu.element_size()
    cpu_flat = cpu.as_strided((size,), (1,))
    gpu_flat = gpu.as_strided((size,), (1,))
    cpu_flat.fill_(1)
    gpu_flat.copy_(cpu_flat)
    torch.testing.assert_close(gpu.cpu(), cpu)


def test_empty_strided_default_dtype_and_errors():
    old = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        assert torch.empty_strided((2, 3), (3, 1), device='vulkan').dtype == torch.float64
    finally:
        torch.set_default_dtype(old)
    for shape, stride in [((-1, 3), (3, 1)), ((2, 3), (1,)), ((2**62, 8), (8, 1))]:
        with pytest.raises(RuntimeError):
            torch.empty_strided(shape, stride, device='vulkan')


def test_empty_strided_has_no_cpu_copy():
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        torch.empty_strided((4, 7), (1, 4), device='vulkan')
    keys = {e.key for e in profile.key_averages()}
    assert 'aten::_to_copy' not in keys
    assert 'aten::_to_cpu' not in keys
