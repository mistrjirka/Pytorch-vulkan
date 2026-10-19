#pragma once
#include <ATen/ATen.h>
namespace torch_vulkan::ops {
at::Tensor to_compressed_native(const at::Tensor&,at::Layout,at::OptionalIntArrayRef,std::optional<int64_t>);
}
