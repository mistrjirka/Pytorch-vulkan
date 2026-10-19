#include "native_utils.h"
#include <ATen/native/Pool.h>
#include <torch/library.h>

namespace torch_vulkan::ops {
namespace {
using namespace native;
struct Pool {
  int64_t planes,h,w,oh,ow,kh,kw,sh,sw,ph,pw,dh,dw;
  std::vector<uint32_t> push() const {
    return {u32(planes),u32(h),u32(w),u32(oh),u32(ow),u32(kh),u32(kw),
            u32(sh),u32(sw),u32(ph),u32(pw),u32(dh),u32(dw)};
  }
};
std::array<int64_t,2> pair(at::IntArrayRef values,const char* name) {
  TORCH_CHECK(values.size()==1 || values.size()==2,"Vulkan max_pool2d: ",name," requires one or two values");
  for(auto v:values){if(std::strcmp(name,"padding")==0)TORCH_CHECK(v>=0,"pad must be non-negative");TORCH_CHECK(v>=0 && v<=INT32_MAX,"Vulkan max_pool2d: invalid ",name);}
  return {values[0],values.size()==1?values[0]:values[1]};
}
Pool params(const at::Tensor& input,at::IntArrayRef kernel,at::IntArrayRef stride,
    at::IntArrayRef padding,at::IntArrayRef dilation,bool ceil) {
  TORCH_CHECK(input.device().type()==c10::DeviceType::PrivateUse1 && (input.scalar_type()==at::kFloat||input.scalar_type()==at::kLong),
              "Vulkan max_pool2d requires FP32/int64 Vulkan input");
  TORCH_CHECK(input.dim()==3 || input.dim()==4,"Vulkan max_pool2d requires CHW or NCHW input");
  auto k=pair(kernel,"kernel"),s=stride.empty()?k:pair(stride,"stride"),
       p=pair(padding,"padding"),d=pair(dilation,"dilation");
  TORCH_CHECK(k[0]>0 && k[1]>0 && s[0]>0 && s[1]>0 && d[0]>0 && d[1]>0,
              "Vulkan max_pool2d requires positive kernel, stride and dilation");
  auto h=input.size(-2),w=input.size(-1),c=input.size(-3);
  for(int axis=0;axis<2;++axis) {
    const auto size=axis?w:h;
    TORCH_CHECK(size>=0 && size+p[axis]+d[axis]*(k[axis]-1)+1+s[axis]<=INT32_MAX,
                "Vulkan max_pool2d spatial arithmetic exceeds int32");
  }
  auto oh=at::native::pooling_output_shape<int64_t>(h,k[0],p[0],s[0],d[0],ceil);
  auto ow=at::native::pooling_output_shape<int64_t>(w,k[1],p[1],s[1],d[1],ceil);
  TORCH_CHECK(c>0&&h>0&&w>0,"Expected 3D or 4D (batch mode) tensor with optional 0 dim batch size for input");
  at::native::pool2d_shape_check(input,k[0],k[1],s[0],s[1],p[0],p[1],d[0],d[1],
      c,h,w,oh,ow,at::MemoryFormat::Contiguous);
  u32(input.numel());
  const auto planes=(input.dim()==4?input.size(0):1)*c;
  TORCH_CHECK(uint64_t(h)*w<=INT32_MAX && uint64_t(oh)*ow<=UINT32_MAX &&
      (!planes || uint64_t(planes)<=UINT32_MAX/(uint64_t(oh)*ow)),
      "Vulkan max_pool2d tensor indexing overflow");
  return {planes,h,w,oh,ow,k[0],k[1],s[0],s[1],p[0],p[1],d[0],d[1]};
}
std::vector<int64_t> output_shape(const at::Tensor& input,const Pool& p) {
  auto shape=input.sizes().vec();shape[shape.size()-2]=p.oh;shape.back()=p.ow;return shape;
}
std::tuple<at::Tensor,at::Tensor> forward(const at::Tensor& input,at::IntArrayRef kernel,
    at::IntArrayRef stride,at::IntArrayRef padding,at::IntArrayRef dilation,bool ceil) {
  auto p=params(input,kernel,stride,padding,dilation,ceil);
  auto shape=output_shape(input,p);
  auto out=at::empty(shape,input.options());
  auto indices=at::empty(shape,input.options().dtype(at::kLong));
  if(out.numel()) {
    auto x=packed(input,input.scalar_type());
    run(input.scalar_type()==at::kLong?"max_pool2d_int64":"max_pool2d_forward",{base(x,input.scalar_type())},{write_base(out,input.scalar_type()),write_base(indices,at::kLong)},p.push(),(out.numel()+127)/128);
  }
  return {out,indices};
}
at::Tensor backward(const at::Tensor& grad,const at::Tensor& input,at::IntArrayRef kernel,
    at::IntArrayRef stride,at::IntArrayRef padding,at::IntArrayRef dilation,bool ceil,const at::Tensor& indices) {
  auto p=params(input,kernel,stride,padding,dilation,ceil);
  auto shape=output_shape(input,p);
  TORCH_CHECK(grad.sizes()==at::IntArrayRef(shape) && indices.sizes()==grad.sizes(),
              "Vulkan max_pool2d backward: gradient/indices shape mismatch");
  TORCH_CHECK(grad.device()==input.device() && grad.scalar_type()==at::kFloat &&
      indices.device()==input.device() && indices.scalar_type()==at::kLong,
      "Vulkan max_pool2d backward requires FP32 gradient and int64 indices on the input device");
  auto out=at::empty(input.sizes(),input.options());
  if(out.numel()) {
    // Backward uses the saved indices; it never recomputes maxima from input.
    auto g=packed(grad),idx=packed(indices,at::kLong);
    run("max_pool2d_backward",{base(g),base(idx,at::kLong)},{write_base(out)},p.push(),(out.numel()+127)/128);
  }
  return out;
}
} // namespace
TORCH_LIBRARY_IMPL(aten,PrivateUse1,m) {
  const char* flag=std::getenv("TORCH_VULKAN_DISABLE_NATIVE_POOL");
  if(flag && !std::strcmp(flag,"1"))return;
  m.impl("max_pool2d_with_indices",&forward);
  m.impl("max_pool2d_with_indices_backward",&backward);
}
} // namespace torch_vulkan::ops
