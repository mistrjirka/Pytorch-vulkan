#pragma once
#include <ATen/ATen.h>
namespace torch_vulkan::ops {
at::Tensor complex_binary(const at::Tensor&,const at::Tensor&,uint32_t,const at::Scalar& alpha=1);
at::Tensor complex_scale(const at::Tensor&,const at::Scalar&);
at::Tensor& complex_fill(at::Tensor&,const at::Scalar&);
}
