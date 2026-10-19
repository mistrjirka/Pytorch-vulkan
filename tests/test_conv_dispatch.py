"""Convolution plans are reusable operator implementations, not model kernels.

Use FP32 inputs promoted to CPU float64 for the oracle. Force each tile so
shape boundaries exercise the selected shader rather than a silent fallback.
The subprocess test isolates startup-only memory/submission configuration.
"""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
import torch.nn.functional as F
import torch_vulkan


# xshape, output channels, filter, stride, padding, dilation, groups
CASES = [
    pytest.param((1, 7, 1, 15), 7, (1, 1), (1, 1), (0, 0), (1, 1), 1,
                 id="channels7-spatial15"),
    pytest.param((1, 9, 1, 17), 9, (1, 1), (1, 1), (0, 0), (1, 1), 1,
                 id="channels9-spatial17"),
    pytest.param((1, 15, 1, 31), 15, (1, 1), (1, 1), (0, 0), (1, 1), 1,
                 id="channels15-spatial31"),
    pytest.param((1, 17, 1, 33), 17, (1, 1), (1, 1), (0, 0), (1, 1), 1,
                 id="channels17-spatial33"),
    pytest.param((1, 31, 1, 63), 31, (1, 1), (1, 1), (0, 0), (1, 1), 1,
                 id="channels31-spatial63"),
    pytest.param((1, 33, 1, 65), 33, (1, 1), (1, 1), (0, 0), (1, 1), 1,
                 id="channels33-spatial65"),
    *[pytest.param((1, 9, k + 3, k + 5), 17, (k, k), (1, 1), (0, 0), (1, 1), 1,
                   id=f"filter{k}") for k in (9, 10, 11, 12)],
    pytest.param((2, 18, 19, 23), 34, (3, 5), (2, 1), (2, 3), (2, 1), 2,
                 id="grouped-stride-dilation"),
    pytest.param((1, 5, 11, 13), 15, (3, 5), (1, 2), (1, 4), (1, 2), 5,
                 id="depthwise-multiplier3"),
    pytest.param((2, 6, 15, 19), 9, (3, 5), (1, 1), (3, 4), (2, 3), 3,
                 id="unit-stride-dilated-grouped"),
    pytest.param((1, 5, 11, 13), 15, (3, 5), (1, 1), (2, 4), (2, 2), 5,
                 id="unit-stride-dilated-depthwise"),
    pytest.param((1, 9, 17, 23), 17, (3, 5), (2, 3), (3, 2), (2, 1), 1,
                 id="both-strides-nonunit"),
    *[pytest.param((1, 9, 1, positions), 17, (1, 1), (1, 1), (0, 0), (1, 1), 1,
                   id=f"weight-reduction{positions}") for positions in (255, 256, 257)],
]


def _check_case(xshape, cout, kernel, stride, padding, dilation, groups, *, bias=True):
    generator = torch.Generator().manual_seed(813 + sum(xshape) + sum(kernel))
    x = torch.randn(xshape, generator=generator) * 0.2
    w = torch.randn((cout, xshape[1] // groups, *kernel), generator=generator) * 0.2
    tensors = [x, w]
    if bias:
        tensors.append(torch.randn((cout,), generator=generator) * 0.2)
    reference = [t.double().requires_grad_() for t in tensors]
    device = [t.to("vulkan").requires_grad_() for t in tensors]
    args = (stride, padding, dilation, groups)
    cpu = F.conv2d(reference[0], reference[1], reference[2] if bias else None, *args)
    actual = F.conv2d(device[0], device[1], device[2] if bias else None, *args)
    grad = torch.randn(cpu.shape, generator=generator) * 0.3
    expected_grads = torch.autograd.grad(cpu, reference, grad.double())
    actual_grads = torch.autograd.grad(actual, device, grad.to("vulkan"))
    # Synchronize/read only after the entire dependency chain has been queued.
    assert actual.device.type == "vulkan"
    torch.testing.assert_close(actual.cpu().double(), cpu, rtol=4e-4, atol=3e-5)
    for result, expected in zip(actual_grads, expected_grads):
        assert result.device.type == "vulkan"
        torch.testing.assert_close(result.cpu().double(), expected, rtol=5e-4, atol=5e-5)


@pytest.mark.parametrize("tile", ["base", "wide", "tall", "register"])
@pytest.mark.parametrize("xshape,cout,kernel,stride,padding,dilation,groups", CASES)
def test_forced_tiles_forward_and_all_gradients(
    xshape, cout, kernel, stride, padding, dilation, groups, tile, monkeypatch
):
    monkeypatch.setenv("TORCH_VULKAN_CONV_ALGO", "tiled")
    monkeypatch.setenv("TORCH_VULKAN_CONV_BACKWARD_ALGO", "optimized")
    monkeypatch.setenv("TORCH_VULKAN_CONV_TILE", tile)
    before = torch_vulkan.conv_dispatch_stats()
    _check_case(xshape, cout, kernel, stride, padding, dilation, groups)
    after = torch_vulkan.conv_dispatch_stats()
    for phase in ("forward", "dinput", "dweight"):
        key = f"selected_{phase}_tiled_{tile}"
        assert after.get(key, 0) > before.get(key, 0), (phase, tile, before, after)


@pytest.mark.parametrize("tile", ["base", "wide", "tall", "register"])
def test_forced_tiles_without_bias_and_with_strided_gradients(tile, monkeypatch):
    monkeypatch.setenv("TORCH_VULKAN_CONV_ALGO", "tiled")
    monkeypatch.setenv("TORCH_VULKAN_CONV_BACKWARD_ALGO", "optimized")
    monkeypatch.setenv("TORCH_VULKAN_CONV_TILE", tile)
    generator = torch.Generator().manual_seed(225)
    xb = torch.randn((2, 18, 13, 31), generator=generator) * 0.2
    wb = torch.randn((34, 9, 5, 3), generator=generator) * 0.2
    x = xb[..., 1::2]
    w = wb.transpose(2, 3)
    xv = xb.to("vulkan")[..., 1::2].detach().requires_grad_()
    wv = wb.to("vulkan").transpose(2, 3).detach().requires_grad_()
    xd, wd = x.double().requires_grad_(), w.double().requires_grad_()
    args = ((2, 1), (2, 3), (2, 1), 2)
    expected = F.conv2d(xd, wd, None, *args)
    actual = F.conv2d(xv, wv, None, *args)
    gb = torch.randn((*expected.shape[:-1], expected.shape[-1] * 2), generator=generator) * 0.3
    grad = gb[..., 1::2]
    gradv = gb.to("vulkan")[..., 1::2]
    expected_grads = torch.autograd.grad(expected, (xd, wd), grad.double())
    actual_grads = torch.autograd.grad(actual, (xv, wv), gradv)
    torch.testing.assert_close(actual.cpu().double(), expected, rtol=4e-4, atol=3e-5)
    for result, ref in zip(actual_grads, expected_grads):
        torch.testing.assert_close(result.cpu().double(), ref, rtol=5e-4, atol=5e-5)


@pytest.mark.parametrize("deferred", [False, True])
def test_autotune_reuses_plan_for_new_storage_and_keeps_gradients_correct(deferred):
    # A fresh process is needed because resident/deferred configuration is read
    # on context creation. CUDA/Vulkan benchmarking must not run concurrently.
    script = r'''
import json
import sys
sys.path.insert(0, "tests")
from test_conv_dispatch import _check_case
import torch_vulkan

torch_vulkan.clear_conv_dispatch_cache()
args = ((1, 9, 11, 13), 17, (3, 5), (1, 1), (1, 2), (1, 1), 1)
_check_case(*args)
first = torch_vulkan.conv_dispatch_stats()
assert first["tunes"] >= 3, first
assert first["entries"] >= 3, first
assert first["candidates_timed"] >= 3, first
for phase in ("forward", "dinput", "dweight"):
    assert first.get(f"candidates_timed_{phase}_tiled_register", 0) > 0, first
# Identical descriptor, newly allocated tensors: reuse selection, not buffers.
_check_case(*args)
second = torch_vulkan.conv_dispatch_stats()
assert second["tunes"] == first["tunes"], (first, second)
assert second["hits"] >= first["hits"] + 3, (first, second)
# A changed descriptor must not inherit an unchecked cached plan.
changed = ((1, 9, 11, 14), *args[1:])
_check_case(*changed)
third = torch_vulkan.conv_dispatch_stats()
assert third["tunes"] > second["tunes"], (second, third)
assert third["entries"] > second["entries"], (second, third)
print("CONV_REPORT=" + json.dumps({"first": first, "second": second, "third": third}))
'''
    env = os.environ.copy()
    for key in ("TORCH_VULKAN_CONV_ALGO", "TORCH_VULKAN_CONV_BACKWARD_ALGO", "TORCH_VULKAN_CONV_TILE"):
        env.pop(key, None)
    env.update(TORCH_VULKAN_CONV_AUTOTUNE="1", TORCH_VULKAN_GPU_RESIDENT="1",
               TORCH_VULKAN_DEFER_SUBMISSIONS="1" if deferred else "0")
    run = subprocess.run([sys.executable, "-c", script], cwd=Path(__file__).resolve().parents[1],
                         env=env, text=True, capture_output=True, timeout=180)
    assert run.returncode == 0, run.stdout + run.stderr
    reports = [line.removeprefix("CONV_REPORT=") for line in run.stdout.splitlines()
               if line.startswith("CONV_REPORT=")]
    assert len(reports) == 1, run.stdout + run.stderr
    report = json.loads(reports[0])
    assert report["third"]["entries"] <= 256, report


@pytest.mark.parametrize("tile", ["base", "wide", "tall", "register"])
def test_adjacent_filter_sizes_reuse_compiled_tile_pipelines(tile, monkeypatch):
    monkeypatch.setenv("TORCH_VULKAN_CONV_ALGO", "tiled")
    monkeypatch.setenv("TORCH_VULKAN_CONV_BACKWARD_ALGO", "optimized")
    monkeypatch.setenv("TORCH_VULKAN_CONV_TILE", tile)
    _check_case((1, 9, 13, 15), 17, (10, 10), (1, 1), (0, 0), (1, 1), 1)
    before = torch_vulkan.cache_stats()
    _check_case((1, 9, 14, 16), 17, (11, 11), (1, 1), (0, 0), (1, 1), 1)
    after = torch_vulkan.cache_stats()
    if after["pipeline_hits"] == 0 and after["pipeline_misses"] == 0:
        pytest.skip("External Kompute build has no pipeline-cache counters")
    # Changing filter geometry changes runtime arguments, not these binaries.
    assert after["pipeline_misses"] == before["pipeline_misses"], (before, after)
    # Recycled buffers can reuse a bound algorithm directly, bypassing the
    # lower-level compiled-pipeline lookup. Either cache hit proves reuse;
    # neither path may create a new pipeline for the changed filter size.
    assert (after["pipeline_hits"] > before["pipeline_hits"] or
            after["algo_hits"] > before["algo_hits"]), (before, after)


@pytest.mark.parametrize("mask", [(True, False, False), (False, True, False), (False, False, True)])
def test_autotune_only_requested_gradient(mask, monkeypatch):
    for key in ("TORCH_VULKAN_CONV_ALGO", "TORCH_VULKAN_CONV_BACKWARD_ALGO", "TORCH_VULKAN_CONV_TILE"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("TORCH_VULKAN_CONV_AUTOTUNE", "1")
    generator = torch.Generator().manual_seed(479)
    x = torch.randn((1, 9, 7, 11), generator=generator) * 0.2
    w = torch.randn((17, 9, 3, 5), generator=generator) * 0.2
    go = torch.randn((1, 17, 7, 11), generator=generator) * 0.3
    backward = torch.ops.aten.convolution_backward.default
    params = ([17], [1, 1], [1, 2], [1, 1], False, [0, 0], 1, mask)
    expected = backward(go.double(), x.double(), w.double(), *params)
    torch_vulkan.clear_conv_dispatch_cache()
    result = backward(go.to("vulkan"), x.to("vulkan"), w.to("vulkan"), *params)
    for enabled, actual, ref in zip(mask, result, expected):
        if enabled:
            torch.testing.assert_close(actual.cpu().double(), ref, rtol=5e-4, atol=5e-5)
        else:
            assert actual is None
    stats = torch_vulkan.conv_dispatch_stats()
    for enabled, phase in zip(mask, ("dinput", "dweight", "dbias")):
        count = sum(value for key, value in stats.items() if key.startswith(f"selected_{phase}_"))
        assert (count > 0) if enabled else (count == 0), (mask, phase, stats)
        if phase != "dbias":
            timed = stats.get(f"candidates_timed_{phase}_tiled_register", 0)
            assert (timed > 0) if enabled else (timed == 0), (mask, phase, stats)


def test_forced_direct_bypasses_autotuning_and_cached_plans(monkeypatch):
    for key in ("TORCH_VULKAN_CONV_ALGO", "TORCH_VULKAN_CONV_BACKWARD_ALGO", "TORCH_VULKAN_CONV_TILE"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("TORCH_VULKAN_CONV_AUTOTUNE", "1")
    torch_vulkan.clear_conv_dispatch_cache()
    args = ((1, 9, 7, 11), 17, (3, 5), (1, 1), (1, 2), (1, 1), 1)
    _check_case(*args)
    before = torch_vulkan.conv_dispatch_stats()
    monkeypatch.setenv("TORCH_VULKAN_CONV_ALGO", "direct")
    monkeypatch.setenv("TORCH_VULKAN_CONV_BACKWARD_ALGO", "direct")
    _check_case(*args)
    after = torch_vulkan.conv_dispatch_stats()
    for key in ("hits", "misses", "tunes", "entries", "candidates_timed"):
        assert after[key] == before[key], (key, before, after)
    for phase in ("forward", "dinput", "dweight", "dbias"):
        key = f"selected_{phase}_direct"
        assert after.get(key, 0) == before.get(key, 0) + 1, (phase, before, after)


def test_zero_workspace_uses_eligible_plan_and_separate_cache_key(monkeypatch):
    for key in ("TORCH_VULKAN_CONV_ALGO", "TORCH_VULKAN_CONV_BACKWARD_ALGO", "TORCH_VULKAN_CONV_TILE"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("TORCH_VULKAN_CONV_AUTOTUNE", "1")
    monkeypatch.setenv("TORCH_VULKAN_CONV_WORKSPACE_MB", "64")
    torch_vulkan.clear_conv_dispatch_cache()
    # 257 output positions require two partials for every tiled dWeight candidate.
    args = ((1, 9, 1, 257), 17, (1, 1), (1, 1), (0, 0), (1, 1), 1)
    _check_case(*args)
    first = torch_vulkan.conv_dispatch_stats()
    monkeypatch.setenv("TORCH_VULKAN_CONV_WORKSPACE_MB", "0")
    _check_case(*args)
    second = torch_vulkan.conv_dispatch_stats()
    assert second["tunes"] > first["tunes"], (first, second)
    assert second["entries"] > first["entries"], (first, second)
    for tile in ("base", "wide", "tall", "register"):
        key = f"selected_dweight_tiled_{tile}"
        assert second.get(key, 0) == first.get(key, 0), (key, first, second)
        timed_key = f"candidates_timed_dweight_tiled_{tile}"
        assert second.get(timed_key, 0) == first.get(timed_key, 0), (timed_key, first, second)
    _check_case(*args)
    third = torch_vulkan.conv_dispatch_stats()
    assert third["tunes"] == second["tunes"], (second, third)
    assert third["hits"] >= second["hits"] + 3, (second, third)


def test_forced_split_weight_rejects_zero_workspace(monkeypatch):
    monkeypatch.setenv("TORCH_VULKAN_CONV_BACKWARD_ALGO", "optimized")
    monkeypatch.setenv("TORCH_VULKAN_CONV_TILE", "wide")
    monkeypatch.setenv("TORCH_VULKAN_CONV_WORKSPACE_MB", "0")
    x = torch.zeros((1, 9, 1, 257), device="vulkan")
    w = torch.zeros((17, 9, 1, 1), device="vulkan")
    go = torch.zeros((1, 17, 1, 257), device="vulkan")
    with pytest.raises(RuntimeError, match="[Ww]orkspace"):
        torch.ops.aten.convolution_backward.default(
            go, x, w, [17], [1, 1], [0, 0], [1, 1], False, [0, 0], 1,
            [False, True, False])


def test_direct_forward_and_input_gradient_cross_65535_workgroups(monkeypatch):
    monkeypatch.setenv("TORCH_VULKAN_CONV_ALGO", "direct")
    monkeypatch.setenv("TORCH_VULKAN_CONV_BACKWARD_ALGO", "direct")
    monkeypatch.setenv("TORCH_VULKAN_CONV_AUTOTUNE", "0")
    # Beyond a 65535-by-128 linear grid; devices with that X limit must use Y.
    # Only dInput is requested: avoid a huge serial dWeight/dBias reduction.
    count = 65535 * 128 + 17
    x = torch.arange(count, dtype=torch.float32).remainder_(97).div_(97).view(1, 1, 1, count)
    xv = x.to("vulkan")
    wv = torch.full((1, 1, 1, 1), 2.0, device="vulkan")
    bv = torch.ones((1,), device="vulkan")
    actual = F.conv2d(xv, wv, bv)
    torch.testing.assert_close(actual.cpu(), x * 2 + 1, rtol=1e-6, atol=1e-6)
    gradv = torch.full(x.shape, 0.25, device="vulkan")
    dx, dw, db = torch.ops.aten.convolution_backward.default(
        gradv, xv, wv, [1], [1, 1], [0, 0], [1, 1], False, [0, 0], 1,
        [True, False, False])
    torch.testing.assert_close(dx.cpu(), torch.full_like(x, 0.5), rtol=0, atol=0)
    assert dw is None and db is None


def test_spatial_coordinate_overflow_rejected_before_dispatch(monkeypatch):
    # Output is only3x3, but the coordinate arithmetic exceeds signed32-bit.
    monkeypatch.setenv("TORCH_VULKAN_CONV_AUTOTUNE", "0")
    x = torch.ones(1, 1, 1, 1).to("vulkan")
    w = torch.ones(1, 1, 1, 1).to("vulkan")
    with pytest.raises(RuntimeError, match="int32 coordinates"):
        F.conv2d(x, w, stride=2147483647, padding=2147483647)


@pytest.mark.parametrize("xshape,cout,kernel,stride,padding,dilation,groups", [
    pytest.param((2, 32, 7, 11), 32, (3, 5), (1, 2), (1, 2), (1, 1), 1,
                 id="exact-channel32-rectangular"),
    pytest.param((2, 62, 9, 13), 66, (3, 5), (1, 2), (2, 2), (2, 1), 2,
                 id="group-channel31-to33-dilated"),
    pytest.param((2, 126, 7, 11), 130, (5, 1), (2, 1), (2, 0), (1, 2), 2,
                 id="group-channel63-to65-strided"),
    pytest.param((2, 130, 11, 9), 126, (1, 3), (2, 1), (0, 2), (2, 2), 2,
                 id="group-channel65-to63-dilated"),
])
def test_register_tile_batch_group_and_channel_boundaries(
    xshape, cout, kernel, stride, padding, dilation, groups, monkeypatch
):
    monkeypatch.setenv("TORCH_VULKAN_CONV_ALGO", "tiled")
    monkeypatch.setenv("TORCH_VULKAN_CONV_BACKWARD_ALGO", "optimized")
    monkeypatch.setenv("TORCH_VULKAN_CONV_TILE", "register")
    before = torch_vulkan.conv_dispatch_stats()
    _check_case(xshape, cout, kernel, stride, padding, dilation, groups)
    after = torch_vulkan.conv_dispatch_stats()
    for phase in ("forward", "dinput", "dweight"):
        key = f"selected_{phase}_tiled_register"
        assert after.get(key, 0) > before.get(key, 0), (phase, before, after)


@pytest.mark.parametrize("xshape,cout,kernel,padding,groups,expected", [
    pytest.param((1, 32, 64, 64), 32, (3, 3), (1, 1), 1, "register3", id="dense-large-3x3"),
    pytest.param((1, 32, 32, 32), 32, (3, 3), (1, 1), 1, "base", id="dense-small-3x3"),
    pytest.param((1, 64, 32, 32), 64, (3, 3), (1, 1), 1, "register3", id="dense-medium-high-channels"),
    pytest.param((2, 32, 32, 32), 32, (3, 3), (1, 1), 1, "register3", id="dense-medium-batched"),
    pytest.param((1, 32, 64, 64), 32, (3, 5), (1, 2), 1, "base", id="dense-large-3x5"),
    pytest.param((1, 32, 64, 64), 32, (3, 3), (1, 1), 2, "base", id="grouped-large-3x3"),
])
def test_default_large_dense_conv_dispatch_and_gradients(
    xshape, cout, kernel, padding, groups, expected, monkeypatch
):
    # Test the ordinary heuristic rather than forcing a tile or using tuning.
    for key in ("TORCH_VULKAN_CONV_ALGO", "TORCH_VULKAN_CONV_BACKWARD_ALGO",
                "TORCH_VULKAN_CONV_TILE"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("TORCH_VULKAN_CONV_AUTOTUNE", "0")
    before = torch_vulkan.conv_dispatch_stats()
    _check_case(xshape, cout, kernel, (1, 1), padding, (1, 1), groups)
    after = torch_vulkan.conv_dispatch_stats()
    for phase in ("forward", "dinput", "dweight"):
        key = f"selected_{phase}_tiled_{expected}"
        assert after.get(key, 0) == before.get(key, 0) + 1, (phase, before, after)


@pytest.mark.parametrize("shape,kernel,padding,stride,dilation,groups", [
    ((2, 32, 19, 21), (3, 3), (1, 1), (1, 1), (1, 1), 1),
    ((1, 32, 11, 13), (3, 3), (2, 1), (2, 1), (1, 2), 2),
    ((2, 17, 17, 19), (3, 3), (2, 0), (1, 1), (2, 1), 1),
])
def test_fixed_3x3_register_parity(shape, kernel, padding, stride, dilation, groups, monkeypatch):
    monkeypatch.setenv("TORCH_VULKAN_CONV_ALGO", "tiled")
    monkeypatch.setenv("TORCH_VULKAN_CONV_BACKWARD_ALGO", "optimized")
    monkeypatch.setenv("TORCH_VULKAN_CONV_TILE", "register3")
    _check_case(shape, 32, kernel, stride, padding, dilation, groups)


def test_fixed_3x3_rejects_incompatible_filters(monkeypatch):
    monkeypatch.setenv("TORCH_VULKAN_CONV_ALGO", "tiled")
    monkeypatch.setenv("TORCH_VULKAN_CONV_TILE", "register3")
    x = torch.zeros((1, 32, 9, 13), device="vulkan")
    w = torch.zeros((32, 32, 3, 5), device="vulkan")
    with pytest.raises(RuntimeError, match="incompatible|exceeds"):
        F.conv2d(x, w)
