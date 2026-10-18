#pragma once
#include <ATen/ATen.h>
#include <array>
namespace torch_vulkan::ops {
at::Tensor convolution_nd(const at::Tensor&,const at::Tensor&,const std::optional<at::Tensor>&,
 c10::SymIntArrayRef,c10::SymIntArrayRef,c10::SymIntArrayRef,bool,c10::SymIntArrayRef,c10::SymInt);
std::tuple<at::Tensor,at::Tensor,at::Tensor> convolution_nd_backward(const at::Tensor&,const at::Tensor&,const at::Tensor&,
 c10::SymIntArrayRef,c10::SymIntArrayRef,c10::SymIntArrayRef,bool,c10::SymIntArrayRef,c10::SymInt,std::array<bool,3>);
}
