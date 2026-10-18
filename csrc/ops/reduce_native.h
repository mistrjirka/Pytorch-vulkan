#pragma once
#include <ATen/ATen.h>
namespace torch_vulkan::ops {
at::Tensor reduce_fp32(const at::Tensor&,at::OptionalIntArrayRef,bool,bool);
}
