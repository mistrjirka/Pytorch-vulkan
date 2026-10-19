"""
torch-vulkan: Vulkan compute backend for PyTorch

Usage:
    import torch
    import torch_vulkan

    # Check availability
    print(torch_vulkan.is_available())
    print(torch_vulkan.device_name())

    # Create on CPU, then move to the Vulkan device (the supported path):
    a = torch.randn(64, 64).to("vulkan")
    b = torch.randn(64, 64).to("vulkan")
    c = torch.mm(a, b)  # runs matmul_tiled.spv on GPU
    c.cpu()             # bring the result back

    # torch.empty(..., device="vulkan") and torch.tensor(data, device="vulkan")
    # also work. The torch.randn(..., device="vulkan") constructor and the
    # .vulkan() method are only partially wired -- see the README Limitations
    # section. Prefer .to("vulkan").
"""

import os

import torch

# Load the C++ extension — this registers PrivateUse1 as "vulkan"
from torch_vulkan import _C

# PyTorch already defines `Tensor.is_vulkan` for its legacy Vulkan tensor type,
# which is not `PrivateUse1`. The standard generator therefore fails trying to
# register `is_vulkan` a second time. Do not alias the two device predicates.
# Use tensor.device.type == "vulkan" to check this backend.
def _tensor_vulkan(self, *args, **kwargs):
    if args:
        raise TypeError("tensor.vulkan() takes no positional arguments")
    return self.to(device="vulkan", **kwargs)


torch.Tensor.vulkan = _tensor_vulkan

# PyTorch's standard nn.Module.to('vulkan') also works.

# Point the shader loader at our bundled .spv files.
_shader_dir = os.path.join(os.path.dirname(__file__), "..", "csrc", "shaders")
if os.path.isdir(_shader_dir):
    _abs_shader_dir = os.path.abspath(_shader_dir)
    _C._set_shader_dir(_abs_shader_dir)
    # The raw VulkanEngine (used by the mm_raw path) is constructed lazily and
    # reads its shader directory from this env var, so it resolves to the same
    # bundled shaders instead of a hardcoded default.
    os.environ.setdefault("TORCH_VULKAN_SHADER_DIR", _abs_shader_dir)


def is_available() -> bool:
    """Check if a Vulkan-capable GPU is available."""
    return _C._is_available()


def is_hardware_accelerated() -> bool:
    """True for physical integrated/discrete GPUs; False for Lavapipe and errors."""
    return _C._is_hardware_accelerated()


def synchronize(device=None) -> None:
    """Flush deferred Vulkan commands and wait for GPU completion."""
    _validate_device(device)
    _C._synchronize()


def gpu_timing_stats() -> dict:
    """Read opt-in kernel GPU timestamps (TORCH_VULKAN_PROFILE_TIMESTAMPS=1)."""
    return _C._gpu_timing_stats()


def queue_stats() -> dict:
    """Inspect experimental deferred dispatch/submission counts."""
    return _C._queue_stats()


def device_name() -> str:
    """Return the name of the Vulkan device."""
    return _C._device_name()


def device(index: int = 0) -> torch.device:
    """Return a torch.device for the Vulkan backend."""
    return torch.device("vulkan", index)


def cache_stats() -> dict:
    """Return algorithm cache hit/miss statistics (Phase 4).

    Returns a dict with keys:
      algo_hits:   number of times a cached Algorithm was reused
      algo_misses: number of times a new Algorithm was created
      seq_reuses:  number of times a Sequence was reused from pool
      seq_creates: number of times a new Sequence was created
    """
    return _C._cache_stats()


def clear_algorithm_cache():
    """Clear all cached Vulkan algorithms.

    Call this if you notice memory pressure from too many cached pipelines,
    or after model weights change and old descriptors are stale.
    """
    _C._clear_algorithm_cache()


# PyTorch optimizers and torch.accelerator require a registered device module.
# Registering only sys.modules["torch.vulkan"] is insufficient in PyTorch 2.14.
import sys


def device_count() -> int:
    return 1 if is_available() else 0


def current_device() -> int:
    return 0


def _is_in_bad_fork() -> bool:
    return False


def manual_seed_all(seed: int) -> None:
    _C._default_generator().manual_seed(seed)


torch._register_device_module("vulkan", sys.modules[__name__])

# This is backend arithmetic support, independent of physical-device FP64 features.
has_fp64 = False
from .formatting import install_fp32_formatter as _install_fp32_formatter
_install_fp32_formatter(sys.modules[__name__])


def conv_dispatch_stats() -> dict:
    """Convolution plan-cache counters and selections by phase/implementation."""
    return _C._conv_dispatch_stats()


def clear_conv_dispatch_cache() -> None:
    """Reset optional convolution tuning choices and their counters."""
    _C._clear_conv_dispatch_cache()


def manual_seed(seed: int) -> None:
    manual_seed_all(seed)


def initial_seed() -> int:
    return _C._default_generator().initial_seed()


def seed() -> None:
    _C._default_generator().seed()


def seed_all() -> None:
    seed()


def get_rng_state(device="vulkan"):
    _validate_device(device)
    return _C._default_generator().get_state()


def set_rng_state(state, device="vulkan"):
    _validate_device(device)
    _C._default_generator().set_state(state)

from . import attention as _attention


def _validate_device(value=None):
    if value is None:
        return
    d=torch.device('vulkan',value) if isinstance(value,int) else torch.device(value)
    if d.type!='vulkan' or d.index not in (None,0):
        raise RuntimeError('Vulkan backend supports logical device 0')


def set_device(value):
    _validate_device(value)


def current_stream(device=None):
    _validate_device(device)
    return _C._current_stream()


def default_stream(device=None):
    _validate_device(device)
    return _C._default_stream()


def set_stream(value):
    if value is not None:
        _validate_device(value.device)
        _C._set_stream_id(value.stream_id)


def stream(value):
    import contextlib
    return value if value is not None else contextlib.nullcontext()


class Stream(torch.Stream):
    def __new__(cls,device='vulkan',priority=0):
        _validate_device(device)
        return super().__new__(cls,device=device,priority=priority)


class Event(torch.Event):
    def __new__(cls,enable_timing=False,blocking=False,interprocess=False):
        if interprocess:
            raise RuntimeError('Interprocess Vulkan events are unsupported')
        return torch.Event(device='vulkan',enable_timing=enable_timing,blocking=blocking)

from . import inductor as _inductor


def execution_stats():
    """CPU fallback and host materialization counters, including explicit .cpu()."""
    return _C._execution_stats()


def reset_execution_stats():
    synchronize()
    _C._reset_execution_stats()
