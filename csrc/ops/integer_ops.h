#pragma once
#include <ATen/Functions.h>
namespace torch_vulkan::ops {
at::Tensor integer_matmul(const at::Tensor&,const at::Tensor&);
bool integer_type(at::ScalarType type);
at::Tensor integer_binary(const at::Tensor&,const at::Tensor&,const at::Scalar&,bool);
at::Tensor integer_sum(const at::Tensor&,at::OptionalIntArrayRef,bool,std::optional<at::ScalarType>);
}
