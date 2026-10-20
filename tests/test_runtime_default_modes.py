"""Check optimized execution without needing process-level environment setup."""
import os
import subprocess
import sys

import pytest
import torch_vulkan


@pytest.mark.parametrize("mode,overrides,expected", [
    ("default", {}, (True, True)),
    ("legacy", {"TORCH_VULKAN_GPU_RESIDENT": "0"}, (False, False)),
    ("eager_resident", {"TORCH_VULKAN_DEFER_SUBMISSIONS": "0"}, (True, False)),
    ("explicit_fast", {"TORCH_VULKAN_GPU_RESIDENT": "1",
                       "TORCH_VULKAN_DEFER_SUBMISSIONS": "1"}, (True, True)),
])
def test_execution_mode_in_fresh_process(mode, overrides, expected):
    env = dict(os.environ)
    env.pop("TORCH_VULKAN_GPU_RESIDENT", None)
    env.pop("TORCH_VULKAN_DEFER_SUBMISSIONS", None)
    env.update(overrides)
    script = """
import torch, torch_vulkan
cfg = torch_vulkan.performance_options()
assert (cfg["gpu_resident"],cfg["deferred_submissions"]) == EXPECTED, cfg
assert cfg["adamw_multitensor_requested"]
a=torch.arange(64,dtype=torch.float32).to("vulkan")
b=(a+2).relu()*3
torch.testing.assert_close(b.cpu(),(torch.arange(64,dtype=torch.float32)+2)*3)
""".replace("EXPECTED", repr(expected))
    result = subprocess.run([sys.executable, "-c", script],
                            env=env, capture_output=True, text=True, timeout=70)
    assert result.returncode == 0, (
        f"{mode}: {result.stdout}\n{result.stderr}")


def test_default_mode_is_active_in_main_process():
    config = torch_vulkan.performance_options()
    assert config.keys() >= {"gpu_resident", "deferred_submissions",
                             "adamw_multitensor_requested"}
    assert isinstance(config["gpu_resident"], bool)
    assert isinstance(config["deferred_submissions"], bool)


def test_conflicting_mode_is_rejected():
    env = dict(os.environ)
    env.update(TORCH_VULKAN_GPU_RESIDENT="0",
               TORCH_VULKAN_DEFER_SUBMISSIONS="1")
    result = subprocess.run([sys.executable, "-c",
        "import torch_vulkan; torch_vulkan.performance_options()"],
        env=env, capture_output=True, text=True, timeout=50)
    assert result.returncode != 0
    assert "requires GPU residency" in result.stderr


def test_invalid_flag_rejected():
    env = dict(os.environ)
    env.update(TORCH_VULKAN_GPU_RESIDENT="banana",
               TORCH_VULKAN_DEFER_SUBMISSIONS="1")
    result = subprocess.run([sys.executable, "-c",
        "import torch_vulkan; torch_vulkan.performance_options()"],
        env=env, capture_output=True, text=True, timeout=50)
    assert result.returncode != 0
    assert "must be 0 or 1" in result.stderr
