// Original FP32 PyTorch GroupNorm training kernels for PrivateUse1 Vulkan.
// Reduction organization informed by the public BSD-3-Clause ncnn GroupNorm
// architecture; no ncnn shader or source code is copied into this file.
// The forward/backward math follows the PyTorch native_group_norm contract.
#include "native_utils.h"
#include <torch/library.h>
#include <ATen/Functions.h>
#include <cmath>
#include <optional>
#include <array>

namespace torch_vulkan::ops {
namespace {
using namespace native;

bool native_enabled() {
  const auto* opt=std::getenv("TORCH_VULKAN_DISABLE_NATIVE_GROUP_NORM");
  return !opt || std::strcmp(opt,"1")!=0;
}

bool eligible(const at::Tensor& x, int64_t n, int64_t c, int64_t hw, int64_t groups) {
  if(x.device().type()!=c10::DeviceType::PrivateUse1 ||
     x.scalar_type()!=at::kFloat ||
     x.dim()<2 || x.dim()>8 || n<=0 || c<=0 || hw<=0 ||
     groups<=0 || c%groups || x.size(0)!=n || x.size(1)!=c ||
     uint64_t(n)*uint64_t(c)*uint64_t(hw)!=uint64_t(x.numel()) ||
     !can_pack_view(x) ||
     uint64_t(n)*uint64_t(groups)>UINT32_MAX ||
     uint64_t(c/groups)*uint64_t(hw)>UINT32_MAX)
    return false;
  const auto& l=device_limits();
  return l.maxComputeWorkGroupSize[0]>=128 &&
         l.maxComputeWorkGroupInvocations>=128 &&
         l.maxComputeSharedMemorySize>=1024 &&
         l.maxPushConstantsSize>=28 &&
         l.maxPerStageDescriptorStorageBuffers>=6 &&
         l.maxDescriptorSetStorageBuffers>=6;
}

std::optional<at::Tensor> cpu_option(const std::optional<at::Tensor>& v) {
  if(!v || !v->defined()) return std::nullopt;
  return native::cpu_fallback_tensor(*v,__func__);
}

std::tuple<at::Tensor,at::Tensor,at::Tensor> cpu_forward(
    const at::Tensor& x, const std::optional<at::Tensor>& weight,
    const std::optional<at::Tensor>& bias,
    c10::SymInt n, c10::SymInt c, c10::SymInt hw,
    int64_t groups, double eps) {
  auto cpu=at::native_group_norm(native::cpu_fallback_tensor(x,__func__),cpu_option(weight),cpu_option(bias),
                                 n.expect_int(),c.expect_int(),hw.expect_int(),groups,eps);
  return {std::get<0>(cpu).to(x.device()),std::get<1>(cpu).to(x.device()),
          std::get<2>(cpu).to(x.device())};
}

std::tuple<at::Tensor,at::Tensor,at::Tensor> cpu_backward(
    const at::Tensor& grad, const at::Tensor& x,
    const at::Tensor& mean, const at::Tensor& rstd,
    const std::optional<at::Tensor>& weight,
    c10::SymInt n, c10::SymInt c, c10::SymInt hw,
    int64_t groups, std::array<bool,3> mask) {
  auto result=at::native_group_norm_backward(
      native::cpu_fallback_tensor(grad,__func__),native::cpu_fallback_tensor(x,__func__),native::cpu_fallback_tensor(mean,__func__),native::cpu_fallback_tensor(rstd,__func__),cpu_option(weight),
      n.expect_int(),c.expect_int(),hw.expect_int(),groups,mask);
  return {std::get<0>(result).defined()?std::get<0>(result).to(x.device()):at::Tensor{},
          std::get<1>(result).defined()?std::get<1>(result).to(x.device()):at::Tensor{},
          std::get<2>(result).defined()?std::get<2>(result).to(x.device()):at::Tensor{}};
}

std::tuple<at::Tensor,at::Tensor,at::Tensor> forward(
    const at::Tensor& x, const std::optional<at::Tensor>& weight,
    const std::optional<at::Tensor>& bias, c10::SymInt N,
    c10::SymInt C, c10::SymInt HxW,int64_t group,double eps) {
  const auto n=N.expect_int(),c=C.expect_int(),hw=HxW.expect_int();
  TORCH_CHECK(group>0 && n>0 && c>0 && hw>0 && c%group==0,
              "native_group_norm: invalid batch, channel, group or spatial dimensions");
  TORCH_CHECK(std::isfinite(eps) && eps>=0,
              "native_group_norm: eps must be finite and nonnegative");
  TORCH_CHECK(x.scalar_type()==at::kFloat,
              "Vulkan native_group_norm currently requires FP32");
  TORCH_CHECK(!weight || !weight->defined() ||
              (weight->device()==x.device() && weight->scalar_type()==at::kFloat &&
               weight->numel()==c),
              "native_group_norm: weight must have C float elements");
  TORCH_CHECK(!bias || !bias->defined() ||
              (bias->device()==x.device() && bias->scalar_type()==at::kFloat &&
               bias->numel()==c),
              "native_group_norm: bias must have C float elements");
  if(!native_enabled() || !eligible(x,n,c,hw,group) ||
     (weight && weight->defined() && !can_pack_view(*weight)) ||
     (bias && bias->defined() && !can_pack_view(*bias)))
    return cpu_forward(x,weight,bias,N,C,HxW,group,eps);

  const auto packed_x=packed(x);
  const auto packed_w=weight && weight->defined() ? packed(*weight) : packed_x;
  const auto packed_b=bias && bias->defined() ? packed(*bias) : packed_x;
  auto out=at::empty(x.sizes(),x.options());
  auto mean=at::empty({n,group},x.options());
  auto rstd=at::empty({n,group},x.options());
  run("group_norm_forward",{base(packed_x),base(packed_w),base(packed_b)},
      {write_base(out),write_base(mean),write_base(rstd)},
      {u32(n),u32(c),u32(hw),u32(group),bits(static_cast<float>(eps)),
       uint32_t(weight && weight->defined()),uint32_t(bias && bias->defined())},
      uint64_t(n)*group,1024);
  return {out,mean,rstd};
}

std::tuple<at::Tensor,at::Tensor,at::Tensor> backward(
    const at::Tensor& grad, const at::Tensor& x,
    const at::Tensor& mean,const at::Tensor& rstd,
    const std::optional<at::Tensor>& weight,c10::SymInt N,
    c10::SymInt C,c10::SymInt HxW,int64_t group,
    std::array<bool,3> mask) {
  const auto n=N.expect_int(),c=C.expect_int(),hw=HxW.expect_int();
  TORCH_CHECK(group>0 && n>0 && c>0 && hw>0 && c%group==0,
              "native_group_norm_backward: invalid geometry");
  TORCH_CHECK(grad.sizes()==x.sizes() && mean.numel()==n*group &&
              rstd.numel()==n*group,
              "native_group_norm_backward: input and saved statistic shape mismatch");
  TORCH_CHECK(x.scalar_type()==at::kFloat && grad.scalar_type()==at::kFloat &&
              mean.scalar_type()==at::kFloat && rstd.scalar_type()==at::kFloat,
              "native_group_norm_backward: FP32 only");
  TORCH_CHECK(!weight || !weight->defined() ||
              (weight->device()==x.device() && weight->numel()==c &&
               weight->scalar_type()==at::kFloat),
              "native_group_norm_backward: invalid gamma");
  const bool fit=native_enabled() && eligible(x,n,c,hw,group) &&
                 can_pack_view(grad) && can_pack_view(mean) && can_pack_view(rstd) &&
                 (!weight || !weight->defined() || can_pack_view(*weight));
  if(!fit)return cpu_backward(grad,x,mean,rstd,weight,N,C,HxW,group,mask);
  if(!mask[0] && !mask[1] && !mask[2])return {};

  // Keep the exact storage lifetimes pinned through Kompute algorithm/sequence.
  const auto xc=packed(x),gc=packed(grad);
  const auto mc=packed(mean),rc=packed(rstd);
  at::Tensor dx,dw,db;
  if(mask[0]){
    auto wc=weight && weight->defined()?packed(*weight):xc;
    dx=at::empty(x.sizes(),x.options());
    run("group_norm_backward_input",
        {base(gc),base(xc),base(mc),base(rc),base(wc)},{write_base(dx)},
        {u32(n),u32(c),u32(hw),u32(group),uint32_t(weight && weight->defined())},
        uint64_t(n)*group,1024);
  }
  if(mask[1] || mask[2]){
    // Compute both parameter gradients in a single reduction.
    auto gw=at::empty({c},x.options());
    auto gb=at::empty({c},x.options());
    run("group_norm_backward_affine",
        {base(gc),base(xc),base(mc),base(rc)},{write_base(gw),write_base(gb)},
        {u32(n),u32(c),u32(hw),u32(group)},(uint64_t(c)+127)/128);
    if(mask[1])dw=gw;
    if(mask[2])db=gb;
  }
  return {dx,dw,db};
}
} // namespace
TORCH_LIBRARY_IMPL(aten,PrivateUse1,m) {
  m.impl("native_group_norm",&forward);
  m.impl("native_group_norm_backward",&backward);
}
} // namespace torch_vulkan::ops
