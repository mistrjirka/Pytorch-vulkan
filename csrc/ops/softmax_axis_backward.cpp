// Original general FP32 softmax backward for any dimension, not only last axis.
// Uses the same (outer,axis,inner) normalization as the forward shader.
#include "native_utils.h"
#include <torch/library.h>
#include <ATen/Functions.h>
#include <algorithm>

namespace torch_vulkan::ops {
namespace {
using namespace native;
at::Tensor backward(
    const at::Tensor& grad,const at::Tensor& out,
    int64_t dim,at::ScalarType input_dtype) {
  TORCH_CHECK(grad.sizes()==out.sizes() && grad.device()==out.device(),
              "Vulkan softmax backward: grad/output shape or device mismatch");
  TORCH_CHECK(input_dtype==at::kFloat && grad.scalar_type()==at::kFloat &&
              out.scalar_type()==at::kFloat,
              "Vulkan softmax backward currently supports FP32 only");
  const auto ndim=std::max<int64_t>(1,out.dim());
  const auto axis=dim<0?dim+ndim:dim;
  TORCH_CHECK(axis>=0 && axis<ndim,"Vulkan softmax backward: invalid dimension");
  if(!can_pack_view(grad) || !can_pack_view(out))
    return at::_softmax_backward_data(native::cpu_fallback_tensor(grad,__func__),native::cpu_fallback_tensor(out,__func__),dim,input_dtype).to(out.device());

  auto dx=at::empty(out.sizes(),out.options());
  if(!out.numel())return dx;
  uint64_t outer=1,inner=1;
  auto axis_size=out.dim()?out.size(axis):1;
  for(int64_t d=0;d<out.dim();++d){
    if(d<axis) outer*=uint64_t(out.size(d));
    else if(d>axis) inner*=uint64_t(out.size(d));
  }
  TORCH_CHECK(outer*inner*uint64_t(axis_size)<=UINT32_MAX,
              "Vulkan softmax backward: indexing exceeds uint32");
  auto g=packed(grad),y=packed(out);
  run("softmax_axis_backward",{base(g),base(y)},{write_base(dx)},
      {u32(outer),u32(axis_size),u32(inner)},outer*inner,512);
  return dx;
}
} // namespace

TORCH_LIBRARY_IMPL(aten,PrivateUse1,m) {
  m.impl("_softmax_backward_data",&backward);
}
} // namespace torch_vulkan::ops
