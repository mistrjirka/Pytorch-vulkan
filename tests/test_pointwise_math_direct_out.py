"""Out= direct-output optimization preserves PyTorch mathematical semantics."""
import pytest
import torch
import torch_vulkan

OPS = [
    "reciprocal", "log", "log1p", "expm1", "floor", "ceil", "trunc",
    "round", "sign", "log10", "log2", "exp2", "sinh", "cosh",
    "tan", "asin", "acos", "atan", "asinh", "acosh", "atanh",
    "erf", "erfc",
]

def input_for(name, shape):
    x = torch.linspace(-.35, .35, max(1, int(torch.tensor(shape).prod().item()))).reshape(
        shape if all(shape) else tuple(shape)) if all(shape) else torch.empty(shape)
    if name in {"log", "log10", "log2", "reciprocal", "acosh"}:
        return x + 1.4
    if name == "log1p":
        return x + .4
    return x


@pytest.mark.parametrize("name", OPS)
@pytest.mark.parametrize("shape", [(127,), (3, 5, 7), (1, 0)])
def test_pointwise_math_direct_out_reference(name, shape):
    x = input_for(name,shape)
    vx = x.to("vulkan")
    f = getattr(torch,name)
    expected = torch.empty_like(x)
    actual = torch.empty_like(vx)
    f(x, out=expected)
    result = f(vx, out=actual)
    assert result is actual
    torch.testing.assert_close(actual.cpu(), expected, atol=3e-3, rtol=3e-3, equal_nan=True)
    assert torch_vulkan.execution_stats()["cpu_fallback_calls"] == 0


@pytest.mark.parametrize("name", ["log", "log1p", "expm1", "sinh", "atan", "round"])
def test_pointwise_math_out_inplace_and_strides(name):
    f=getattr(torch,name)
    cpu=torch.linspace(.3,1.2,5*7).reshape(5,7)
    gpu=cpu.to('vulkan')
    f(cpu,out=cpu)
    f(gpu,out=gpu)
    torch.testing.assert_close(gpu.cpu(),cpu,atol=3e-3,rtol=3e-3)

    vcpu=cpu.t()
    vgpu=gpu.t()
    out_cpu=torch.empty_like(vcpu)
    out_gpu=torch.empty_like(vgpu)
    f(vcpu,out=out_cpu)
    f(vgpu,out=out_gpu)
    torch.testing.assert_close(out_gpu.cpu(),out_cpu,atol=3e-3,rtol=3e-3,equal_nan=True)


@pytest.mark.parametrize("name", ["floor", "ceil", "trunc", "round", "sign"])
def test_integer_input_retains_preexisting_fallback(name):
    f=getattr(torch,name)
    cpu=torch.arange(-20,20,dtype=torch.int32)
    gpu=cpu.to('vulkan')
    expected=torch.empty_like(cpu)
    result=torch.empty_like(gpu)
    f(cpu,out=expected)
    f(gpu,out=result)
    torch.testing.assert_close(result.cpu(),expected,rtol=0,atol=0)
