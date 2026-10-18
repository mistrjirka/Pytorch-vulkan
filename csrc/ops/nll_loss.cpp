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
  // Validate indices without downloading logits or blocking unrelated queued work.
  // CPU-created targets already have an authoritative host mirror. GPU-written
  // targets require a real completion/download boundary before this read.
  auto& ctx=VulkanContext::instance();auto& alloc=VulkanAllocator::instance();
  auto target_buffer=base(target,at::kLong);
  if(!ctx.gpu_resident_enabled() || target_buffer->tensorType()==kp::Tensor::TensorTypes::eHost)ctx.synchronize();
  if(ctx.gpu_resident_enabled() && alloc.gpu_dirty(target_buffer))
    ctx.ensure_host_read(target.storage().data_ptr().get());
  const auto* ptr=target.const_data_ptr<int64_t>();
  const auto stride=target.dim()?target.stride(0):0;
  for(int64_t row=0;row<rows;++row) {
    const auto index=ptr[row*stride];
    TORCH_CHECK_INDEX(index==ignore || (index>=0 && index<cols),"Target ",index," is out of bounds.");
  }
}
std::vector<uint32_t> params(const at::Tensor& x,const std::optional<at::Tensor>& weight,
                             int64_t reduction,int64_t ignore) {
  const auto v=uint64_t(ignore);
  return {u32(x.dim()==1?1:x.size(0)),u32(x.size(-1)),uint32_t(reduction),
      uint32_t(v),uint32_t(v>>32),uint32_t(weight && weight->defined()),uint32_t(x.dim()==1)};
}
std::optional<at::Tensor> cpu_weight(const std::optional<at::Tensor>& w) {
  return w && w->defined()?std::optional<at::Tensor>(w->cpu()):std::nullopt;
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
