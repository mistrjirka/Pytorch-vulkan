"""Display FP32 tensors on a backend without FP64 arithmetic."""
from functools import wraps

import torch


def install_fp32_formatter(backend):
    # PyTorch 2.14's formatter explicitly handles MPS/XPU/Maia without FP64,
    # but does not consult PrivateUse1 capabilities. Formatting an error can
    # therefore request a Double conversion before the real error is shown.
    # Keep the same FP32 calculation policy for this backend's FP32 tensors.
    import torch._tensor_str as tensor_str

    original = getattr(tensor_str, "tensor_totype", None)
    if original is None or getattr(original, "_vulkan_fp32_formatter", False):
        return

    @wraps(original)
    def tensor_totype(tensor):
        if (tensor.device.type == "vulkan" and tensor.dtype == torch.float32
                and not backend.has_fp64):
            return tensor
        return original(tensor)

    tensor_totype._vulkan_fp32_formatter = True
    tensor_str.tensor_totype = tensor_totype
