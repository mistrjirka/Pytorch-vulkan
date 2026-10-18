#include "native_utils.h"
#include <torch/library.h>

namespace torch_vulkan::ops {
namespace {
using namespace native;
bool eligible(const at::Tensor& x,const at::Tensor& target,const std::optional<at::Tensor>& weight) {
  const auto& limits=device_limits();
  if(limits.maxPerStageDescriptorStorageBuffers<5 || limits.maxDescriptorSetStorageBuffers<5 ||
     limits.maxComputeSharedMemorySize<1024 || limits.maxPushConstantsSize<28 ||
     limits.maxComputeWorkGroupInvocations<128 || limits.maxComputeWorkGroupSize[0]<128) return false;
  if(x.storage().nbytes()>limits.maxStorageBufferRange || target.storage().nbytes()>limits.maxStorageBufferRange ||
     (weight && weight->defined() && weight->storage().nbytes()>limits.maxStorageBufferRange)) return false;
  if(!can_pack_view(x) || !can_pack_view(target) ||
     (weight && weight->defined() && !can_pack_view(*weight))) return false;
  return x.scalar_type()==at::kFloat && (x.dim()==1 || x.dim()==2) &&
      x.numel()>0 && x.numel()<=UINT32_MAX-127 && target.scalar_type()==at::kLong &&
      target.device()==x.device() && target.dim()<=1 &&
      (!weight || !weight->defined() || (weight->device()==x.device() &&
       weight->scalar_type()==at::kFloat && weight->dim()==1));
}
void validate(const at::Tensor& x,const at::Tensor& target,
              const std::optional<at::Tensor>& weight,int64_t reduction,int64_t ignore) {
  const int64_t rows=x.dim()==1?1:x.size(0), cols=x.size(-1);
  TORCH_CHECK(reduction>=0 && reduction<=2,"nll_loss: invalid reduction");
  TORCH_CHECK((x.dim()==1 && target.numel()==1) || (x.dim()==2 && target.dim()==1 && target.numel()==rows),
              "nll_loss: input/target shape mismatch");
  TORCH_CHECK(!weight || !weight->defined() || weight->numel()==cols,"nll_loss: weight must match classes");
  if(!rows)return;
  // Keep label validation on GPU, including GPU-produced targets. Only the
  // first invalid position is downloaded; its value is read only on failure.
  auto invalid=at::full({1},-1,target.options().dtype(at::kInt));
  const auto sentinel=uint64_t(ignore);
  run("nll_validate",{base(target,at::kLong)},{write_base(invalid,at::kInt)},
      {u32(rows),u32(cols),u32(target.storage_offset()),u32(target.dim()?target.stride(0):0),
       uint32_t(sentinel),uint32_t(sentinel>>32)},(rows+127)/128);
  const uint32_t bad=uint32_t(invalid.cpu().item<int32_t>());
  if(bad!=UINT32_MAX){
    auto value=(target.dim()?target.select(0,bad):target).cpu().item<int64_t>();
    TORCH_CHECK_INDEX(false,"Target ",value," is out of bounds.");
  }
}
std::vector<uint32_t> params(const at::Tensor& x,const std::optional<at::Tensor>& weight,
                             int64_t reduction,int64_t ignore) {
  const auto v=uint64_t(ignore);
  return {u32(x.dim()==1?1:x.size(0)),u32(x.size(-1)),uint32_t(reduction),
      uint32_t(v),uint32_t(v>>32),uint32_t(weight && weight->defined()),uint32_t(x.dim()==1)};
}
std::optional<at::Tensor> cpu_weight(const std::optional<at::Tensor>& w) {
  return w && w->defined()?std::optional<at::Tensor>(native::cpu_fallback_tensor(*w,__func__)):std::nullopt;
}
std::tuple<at::Tensor,at::Tensor> forward(const at::Tensor& input,const at::Tensor& target,
    const std::optional<at::Tensor>& weight,int64_t reduction,c10::SymInt ignore_index) {
  const auto ignore=ignore_index.expect_int();
  if(!eligible(input,target,weight)) {
    auto [out,total]=at::nll_loss_forward(native::cpu_fallback_tensor(input,__func__),native::cpu_fallback_tensor(target,__func__),cpu_weight(weight),reduction,ignore);
    return {out.to(input.device()),total.to(input.device())};
  }
  validate(input,target,weight,reduction,ignore);
  auto x=packed(input),t=packed(target,at::kLong);
  auto w=weight && weight->defined()?packed(*weight):x;
  auto out=at::empty(reduction==0 && input.dim()==2?at::IntArrayRef({input.size(0)}):at::IntArrayRef({}),input.options());
  auto total=at::empty({},input.options());
  const auto p=params(input,weight,reduction,ignore);
  run("nll_loss_forward",{base(x),base(t,at::kLong),base(w)},
      {write_base(out),write_base(total)},p,reduction==0 && input.dim()==2?(input.size(0)+127)/128:1,1024);
  return {out,total};
}
at::Tensor backward(const at::Tensor& grad,const at::Tensor& input,const at::Tensor& target,
    const std::optional<at::Tensor>& weight,int64_t reduction,c10::SymInt ignore_index,const at::Tensor& total_weight) {
  const auto ignore=ignore_index.expect_int();
  if(!eligible(input,target,weight) || grad.scalar_type()!=at::kFloat || total_weight.scalar_type()!=at::kFloat ||
     !can_pack_view(grad) || !can_pack_view(total_weight))
    return at::nll_loss_backward(native::cpu_fallback_tensor(grad,__func__),native::cpu_fallback_tensor(input,__func__),native::cpu_fallback_tensor(target,__func__),cpu_weight(weight),reduction,ignore,native::cpu_fallback_tensor(total_weight,__func__)).to(input.device());
  validate(input,target,weight,reduction,ignore);
  TORCH_CHECK(total_weight.numel()==1,"nll_loss_backward: total_weight must be scalar");
  TORCH_CHECK(reduction==0 && input.dim()==2 ? (grad.dim()==1 && grad.numel()==input.size(0)) : (grad.dim()<=1 && grad.numel()==1),"nll_loss_backward: gradient shape mismatch");
  auto g=packed(grad),t=packed(target,at::kLong),total=packed(total_weight);
  auto w=weight && weight->defined()?packed(*weight):g;
  auto out=at::empty(input.sizes(),input.options());
  run("nll_loss_backward",{base(g),base(t,at::kLong),base(w),base(total)},
      {write_base(out)},params(input,weight,reduction,ignore),(input.numel()+127)/128);
  return out;
}
} // namespace
TORCH_LIBRARY_IMPL(aten,PrivateUse1,m) {
  m.impl("nll_loss_forward",&forward);
  m.impl("nll_loss_backward",&backward);
}
} // namespace torch_vulkan::ops
