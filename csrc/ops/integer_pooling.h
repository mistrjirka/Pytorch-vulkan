#pragma once
#include <ATen/ATen.h>
#include <optional>
#include <vector>
namespace torch_vulkan::ops {
at::Tensor average_pool_long_native(const at::Tensor&,at::IntArrayRef,const std::vector<uint32_t>&,std::optional<int64_t>,bool,bool);
}
