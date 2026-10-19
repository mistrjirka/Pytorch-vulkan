// General FP32 LeakyReLU, including gradients and alias-preserving mutation.
#include "native_utils.h"
#include <ATen/MemoryOverlap.h>
#include <ATen/ExpandUtils.h>
#include <ATen/native/Resize.h>
#include <ATen/native/CPUFallback.h>
#include <ATen/ops/leaky_relu_ops.h>
#include <ATen/ops/leaky_relu_backward_ops.h>
#include <torch/library.h>

namespace torch_vulkan::ops {
void cpu_fallback(const c10::OperatorHandle&,torch::jit::Stack*);
namespace {
using namespace native;
template<class Op> using Fallback=at::native::call_fallback_fn<&cpu_fallback,Op>;

// Read-only inputs may broadcast/overlap. Writes need an injective stride map:
// sorted strides at least the full span of preceding axes prove this, including
// transposes and sliced views with holes. Ambiguous layouts retain CPU fallback.
bool supported(const at::Tensor& x,bool writing=false) {
  if(x.device().type()!=c10::DeviceType::PrivateUse1 || x.scalar_type()!=at::kFloat ||
      x.dim()>8 || x.numel()>int64_t(UINT32_MAX)-127 || x.storage_offset()<0)return false;
  uint64_t last=x.storage_offset();
  std::vector<std::pair<uint64_t,uint64_t>> axes;
  for(int64_t d=0;d<x.dim();++d) {
    if(x.size(d)>UINT32_MAX || x.stride(d)<0 || x.stride(d)>UINT32_MAX)return false;
    if(x.size(d)>1) {
      const auto stride=uint64_t(x.stride(d)),size=uint64_t(x.size(d));
      const uint64_t extent=(size-1)*stride;
      if(extent>UINT32_MAX || last>UINT32_MAX-extent)return false;
      last+=extent;axes.push_back({stride,size});
    }
  }
  if(last>UINT32_MAX)return false;
  if(writing) {
    std::sort(axes.begin(),axes.end());uint64_t span=1;
    for(auto [stride,size]:axes) {
      if(stride<span)return false;
      span+=(size-1)*stride;
    }
  }
  const auto& l=device_limits();
  return l.maxComputeWorkGroupInvocations>=128 && l.maxComputeWorkGroupSize[0]>=128 &&
      l.maxPushConstantsSize>=80 && l.maxPerStageDescriptorStorageBuffers>=4 &&
      l.maxDescriptorSetStorageBuffers>=4 && x.storage().nbytes()<=l.maxStorageBufferRange;
}
bool safe_alias(const at::Tensor& out,const at::Tensor& input) {
  if(out.storage().unsafeGetStorageImpl()!=input.storage().unsafeGetStorageImpl())return true;
  // Noncontiguous/offset inputs are snapshotted by packed() before writing.
  if(!input.is_contiguous() || input.storage_offset()!=0)return true;
  return out.sizes()==input.sizes() && out.strides()==input.strides() &&
      out.storage_offset()==input.storage_offset();
}
void overlap_check(const at::Tensor& out,const at::Tensor& x) {
  at::assert_no_internal_overlap(out);at::assert_no_partial_overlap(out,x);
}
void dispatch(const at::Tensor& x,const at::Tensor* grad,const at::Scalar& slope,
              const at::Tensor& out) {
  const float value=slope.toFloat(); // Same scalar conversion as FP32 ATen.
  if(!out.numel())return;
  auto input=packed(x);
  auto gradient=grad?packed(*grad):input; // Forward never reads binding 1.
  std::vector<uint32_t> p(20,0);
  p[0]=u32(out.numel());p[1]=u32(out.storage_offset());p[2]=bits(value);p[3]=grad!=nullptr;
  for(int d=0;d<8;++d) {
    p[4+d]=d<out.dim()?u32(out.size(d)):1;
    p[12+d]=d<out.dim()?u32(out.stride(d)):0;
  }
  // Include destination as an input too: writes to a slice must preserve every
  // untouched storage element when an initially host-dirty allocation uploads.
  const bool partial=!out.is_contiguous() || out.storage_offset()!=0 ||
      out.storage().nbytes()!=uint64_t(out.numel())*sizeof(float);
  run("leaky_relu_native",{base(input),base(gradient),partial?base(out):base(input)},
      {write_base(out)},p,(out.numel()+127)/128);
}
at::Tensor forward(const at::Tensor& x,const at::Scalar& slope) {
  if(!supported(x)&&can_iterate_fp32(x))return iterated_unary_fp32(x,13,slope.toFloat());
  if(!supported(x))return Fallback<at::_ops::leaky_relu>::call(x,slope);
  auto out=at::empty(x.sizes(),x.options());dispatch(x,nullptr,slope,out);return out;
}
at::Tensor& forward_(at::Tensor& x,const at::Scalar& slope) {
  at::assert_no_internal_overlap(x);
  if(!supported(x,true)&&can_iterate_fp32(x)){iterate_fp32(x,x,x,13,slope.toFloat());return x;}
  if(!supported(x,true))return Fallback<at::_ops::leaky_relu_>::call(x,slope);
  dispatch(x,nullptr,slope,x);return x;
}
at::Tensor& forward_out(const at::Tensor& x,const at::Scalar& slope,at::Tensor& out) {
  overlap_check(out,x);
  if(can_iterate_fp32(x)&&can_iterate_fp32(out)&&out.device()==x.device()&&(!supported(x)||!supported(out,true)||out.sizes()!=x.sizes()||!safe_alias(out,x))){auto value=forward(x,slope);at::native::resize_output(out,x.sizes());iterate_fp32(value,value,out,0);return out;}
  if(!supported(x) || !supported(out,true) || out.sizes()!=x.sizes() || !safe_alias(out,x))
    return Fallback<at::_ops::leaky_relu_out>::call(x,slope,out);
  dispatch(x,nullptr,slope,out);return out;
}
void check_result(const at::Scalar& slope,bool self_is_result) {
  TORCH_CHECK(!self_is_result || slope.toDouble()>=0,
      "In-place leakyReLu backward calculation is triggered with a negative slope which is not supported. "
      "Please call out-of-place version instead.");
}
at::Tensor backward(const at::Tensor& grad,const at::Tensor& x,
                    const at::Scalar& slope,bool self_is_result) {
  check_result(slope,self_is_result);
  if(can_iterate_fp32(x)&&can_iterate_fp32(grad)&&x.device()==grad.device()&&(!supported(x)||!supported(grad)||x.sizes()!=grad.sizes())){auto shape=at::infer_size(x.sizes(),grad.sizes());auto out=at::empty(shape,x.options());iterate_fp32(x,grad,out,14,slope.toFloat());return out;}
  if(!supported(x) || !supported(grad) || x.sizes()!=grad.sizes())
    return Fallback<at::_ops::leaky_relu_backward>::call(grad,x,slope,self_is_result);
  auto out=at::empty(x.sizes(),x.options());dispatch(x,&grad,slope,out);return out;
}
at::Tensor& backward_out(const at::Tensor& grad,const at::Tensor& x,
    const at::Scalar& slope,bool self_is_result,at::Tensor& out) {
  check_result(slope,self_is_result);overlap_check(out,x);overlap_check(out,grad);
  if(can_iterate_fp32(x)&&can_iterate_fp32(grad)&&can_iterate_fp32(out)&&x.device()==grad.device()&&x.device()==out.device()&&(!supported(x)||!supported(grad)||!supported(out,true)||x.sizes()!=grad.sizes()||x.sizes()!=out.sizes()||!safe_alias(out,x)||!safe_alias(out,grad))){auto value=backward(grad,x,slope,self_is_result);at::native::resize_output(out,value.sizes());iterate_fp32(value,value,out,0);return out;}
  if(!supported(x) || !supported(grad) || !supported(out,true) ||
      x.sizes()!=grad.sizes() || x.sizes()!=out.sizes() ||
      !safe_alias(out,x) || !safe_alias(out,grad))
    return Fallback<at::_ops::leaky_relu_backward_grad_input>::call(grad,x,slope,self_is_result,out);
  dispatch(x,&grad,slope,out);return out;
}
} // namespace
TORCH_LIBRARY_IMPL(aten,PrivateUse1,m) {
  const char* flag=std::getenv("TORCH_VULKAN_DISABLE_NATIVE_LEAKY_RELU");
  if(flag && !std::strcmp(flag,"1"))return;
  m.impl("leaky_relu",&forward);
  m.impl("leaky_relu_",&forward_);
  m.impl("leaky_relu.out",&forward_out);
  m.impl("leaky_relu_backward",&backward);
  m.impl("leaky_relu_backward.grad_input",&backward_out);
}
} // namespace torch_vulkan::ops
