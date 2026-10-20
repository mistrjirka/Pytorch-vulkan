"""GPU direct-output unary operations retain full PyTorch out= semantics."""
import pytest
import torch
import torch_vulkan


OPS = {
    "sin": torch.sin,
    "cos": torch.cos,
    "exp": torch.exp,
    "sqrt": torch.sqrt,
    "rsqrt": torch.rsqrt,
    "tanh": torch.tanh,
    "neg": torch.neg,
    "silu": lambda t, out: torch.ops.aten.silu.out(t, out=out),
}


def call(name, x, out):
    if name == "silu":
        return OPS[name](x, out)
    return OPS[name](x, out=out)


@pytest.mark.parametrize("name", list(OPS))
@pytest.mark.parametrize("shape", [(257,), (3, 17), (2, 3, 5), (1, 0)])
def test_unary_out_matches_cpu_across_sizes(name, shape):
    gen = torch.Generator().manual_seed(613)
    x = torch.randn(shape, generator=gen) * .45 + .8
    xg = x.to("vulkan")
    expected = torch.empty_like(x)
    actual = torch.empty_like(xg)
    yc = call(name, x, expected)
    yv = call(name, xg, actual)
    assert yc is expected and yv is actual
    torch.testing.assert_close(actual.cpu(), expected, atol=5e-5, rtol=5e-5, equal_nan=True)
    assert torch_vulkan.execution_stats()["cpu_fallback_calls"] == 0


@pytest.mark.parametrize("name", list(OPS))
def test_unary_out_strided_input_contiguous_output(name):
    x = torch.randn(5, 9, 11).transpose(1, 2)
    cpu = torch.empty_like(x, memory_format=torch.contiguous_format)
    gpu = torch.empty(x.shape, device="vulkan")
    call(name, x, cpu)
    call(name, x.to("vulkan"), gpu)
    torch.testing.assert_close(gpu.cpu(), cpu, atol=5e-5, rtol=5e-5, equal_nan=True)


@pytest.mark.parametrize("name", list(OPS))
def test_unary_out_noncontiguous_output_and_resize(name):
    x = torch.randn(17, 9) * .1 + .75
    base = torch.empty((9, 17))
    dest = base.t()
    vg = torch.empty((9, 17), device="vulkan").t()
    call(name, x, dest)
    call(name, x.to("vulkan"), vg)
    torch.testing.assert_close(vg.cpu(), dest, atol=5e-5, rtol=5e-5, equal_nan=True)

    # An unaliased destination may have a different initial size.
    resized = torch.empty(2, device="vulkan")
    res = call(name, x.to("vulkan"), resized)
    assert res is resized and tuple(resized.shape) == tuple(x.shape)
    expected = torch.empty_like(x)
    call(name, x, expected)
    torch.testing.assert_close(resized.cpu(), expected, atol=5e-5, rtol=5e-5, equal_nan=True)


@pytest.mark.parametrize("name", list(OPS))
def test_unary_out_exact_input_alias(name):
    x = torch.linspace(.3, 1.7, 91)
    v = x.to("vulkan")
    cpu = x.clone()
    call(name, cpu, cpu)
    out = call(name, v, v)
    assert out is v
    torch.testing.assert_close(v.cpu(), cpu, atol=5e-5, rtol=5e-5, equal_nan=True)


@pytest.mark.parametrize("name", ["sin", "exp", "sqrt", "tanh", "neg"])
def test_unary_out_with_views_of_equal_storage(name):
    x = torch.arange(24, dtype=torch.float32).reshape(4, 6) * .02 + 1.
    y = x.to("vulkan")
    cpu = x.clone()
    call(name, cpu[:, :3], cpu[:, :3])
    call(name, y[:, :3], y[:, :3])
    torch.testing.assert_close(y.cpu(), cpu, atol=5e-5, rtol=5e-5, equal_nan=True)


def test_dtype_preserving_and_casting_behavior():
    x = torch.tensor([1., 2., 3.], dtype=torch.float32)
    out = torch.empty((3,), device="vulkan", dtype=torch.float32)
    exp = torch.empty((3,), dtype=torch.float32)
    torch.exp(x, out=exp)
    torch.exp(x.to("vulkan"), out=out)
    torch.testing.assert_close(out.cpu(), exp, atol=5e-5, rtol=5e-5, equal_nan=True)


@pytest.mark.parametrize("name", ["sin", "exp", "sqrt", "tanh", "neg"])
def test_partial_overlap_behavior(name):
    x = torch.arange(20, dtype=torch.float32).reshape(4, 5) * .03 + 0.5
    y = x.to("vulkan")
    # The previous Vulkan adapter materializes the complete input before
    # writing overlapping destinations. Keep that snapshot behavior.
    snapshot = x[:, :4].clone()
    expected = torch.empty_like(snapshot)
    call(name, snapshot, expected)
    x[:, 1:].copy_(expected)
    call(name, y[:, :4], y[:, 1:])
    torch.testing.assert_close(y.cpu(), x, atol=5e-5, rtol=5e-5, equal_nan=True)
