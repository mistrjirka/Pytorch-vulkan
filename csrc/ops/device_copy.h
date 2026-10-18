#pragma once
#include <ATen/ATen.h>
namespace torch_vulkan::ops {
bool device_copy_supported(at::ScalarType type);
bool copy_on_device(at::Tensor& out, const at::Tensor& input);
}
