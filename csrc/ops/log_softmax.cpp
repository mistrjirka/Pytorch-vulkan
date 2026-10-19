// General FP32 log-softmax with device-only axis/layout normalization.
#include "native_utils.h"
#include <ATen/WrapDimUtils.h>
#include <torch/library.h>

namespace torch_vulkan::ops {
namespace {
using namespace native;

bool enabled() {
  const auto* flag=std::getenv("TORCH_VULKAN_NATIVE_LOGSOFTMAX");
  return !flag || std::strcmp(flag,"0")!=0;
}
std::vector<int64_t> order(const at::Tensor& input,int64_t axis) {
  std::vector<int64_t> result;
  for(int64_t d=0;d<input.dim();++d)if(d!=axis)result.push_back(d);
  if(input.dim())result.push_back(axis);
  return result;
}
at::Tensor rows(const at::Tensor& input,const std::vector<int64_t>& axes) {
  if(!input.dim())return packed(input.reshape({1}));
  bool identity=true;
  for(size_t d=0;d<axes.size();++d)identity&=axes[d]==int64_t(d);
  return packed(identity?input:input.permute(axes));
}
at::Tensor restore(const at::Tensor& input,const at::Tensor& result,
                   const std::vector<int64_t>& axes) {
  if(!input.dim())return result.reshape({});
  bool identity=true;std::vector<int64_t> inverse(axes.size());
  for(size_t d=0;d<axes.size();++d){inverse[axes[d]]=d;identity&=axes[d]==int64_t(d);}
  return identity?result:packed(result.permute(inverse));
}
bool supported(const at::Tensor& input) {
  return input.scalar_type()==at::kFloat && can_pack_data(input);
}
at::Tensor log_softmax_native(const at::Tensor& input,int64_t dim,bool half_to_float) {
  const auto axis=at::maybe_wrap_dim(dim,input.dim());
  if(!enabled() || !supported(input) || half_to_float)
    return at::_log_softmax(cpu_fallback_tensor(input,__func__),dim,half_to_float).to(input.device());
  if(!input.numel())return at::empty(input.sizes(),input.options());
  auto axes=order(input,axis);auto x=rows(input,axes);
  auto out=at::empty(x.sizes(),input.options());
  const auto cols=u32(x.size(-1)),count=u32(x.numel()/cols);
  run("log_softmax",{base(x)},{write_base(out)},{count,cols},count,1024);
  return restore(input,out,axes);
}
at::Tensor log_softmax_backward_native(const at::Tensor& grad_output,
    const at::Tensor& output,int64_t dim,at::ScalarType input_dtype) {
  TORCH_CHECK(grad_output.sizes()==output.sizes(),
              "log_softmax_backward_data: gradient/output shape mismatch");
  TORCH_CHECK(grad_output.device()==output.device(),
              "log_softmax_backward_data: gradient/output device mismatch");
  const auto axis=at::maybe_wrap_dim(dim,output.dim());
  if(!enabled() || !supported(grad_output) || !supported(output) || input_dtype!=at::kFloat)
    return at::_log_softmax_backward_data(cpu_fallback_tensor(grad_output,__func__),
        cpu_fallback_tensor(output,__func__),dim,input_dtype).to(output.device());
  if(!output.numel())return at::empty(output.sizes(),output.options());
  auto axes=order(output,axis);auto g=rows(grad_output,axes),p=rows(output,axes);
  auto out=at::empty(p.sizes(),output.options());
  const auto cols=u32(p.size(-1)),count=u32(p.numel()/cols);
  run("log_softmax_backward",{base(g),base(p)},{write_base(out)},{count,cols},count,512);
  return restore(output,out,axes);
}
}
TORCH_LIBRARY_IMPL(aten,PrivateUse1,m) {
  m.impl("_log_softmax",&log_softmax_native);
  m.impl("_log_softmax_backward_data",&log_softmax_backward_native);
}
} // namespace torch_vulkan::ops
