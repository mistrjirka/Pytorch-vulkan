#include "native_utils.h"
#include <torch/library.h>
#include <cmath>
namespace torch_vulkan::ops {
namespace {
using namespace native;
// These two shortcuts are part of ATen's legacy nearest-neighbor mapping.
float scale(int64_t input,int64_t output,std::optional<double> supplied) {
  if(input==output)return 1.0f;
  if(output==input*2)return .5f;
  const float value=supplied && *supplied>0 ? float(1.0/ *supplied):float(input)/float(output);
  TORCH_CHECK(std::isfinite(value) && value>0,"Vulkan nearest resize requires finite positive scales");
  return value;
}
std::vector<uint32_t> params(at::IntArrayRef in,c10::SymIntArrayRef out,
    std::optional<double> sh,std::optional<double> sw,bool backward) {
  TORCH_CHECK(in.size()==4 && out.size()==2,"Vulkan nearest2d requires rank 4 input and two output sizes");
  const auto oh=out[0].expect_int(),ow=out[1].expect_int();
  TORCH_CHECK(in[0]>=0 && in[1]>0 && in[2]>0 && in[3]>0 && oh>0 && ow>0,
              "Vulkan nearest2d requires positive channels and spatial dimensions");
  // Validate the total products before constructing uint32 shader arithmetic.
  uint64_t count=1;
  for(auto d:in){u32(d);TORCH_CHECK(!d || count<=UINT32_MAX/uint64_t(d),"Vulkan nearest input indexing overflow");count*=d;}
  const auto planes=int64_t(in[0])*in[1];
  u32(oh);u32(ow);
  TORCH_CHECK(uint64_t(oh)*ow<=UINT32_MAX && uint64_t(planes)*oh*ow<=UINT32_MAX,
              "Vulkan nearest output indexing overflow");
  return {u32(planes),u32(in[2]),u32(in[3]),u32(oh),u32(ow),uint32_t(backward),
          bits(scale(in[2],oh,sh)),bits(scale(in[3],ow,sw))};
}
at::Tensor upsample(const at::Tensor& input,c10::SymIntArrayRef size,
    std::optional<double> sh,std::optional<double> sw) {
  auto p=params(input.sizes(),size,sh,sw,false);
  TORCH_CHECK(input.device().type()==c10::DeviceType::PrivateUse1 && input.scalar_type()==at::kFloat,"Vulkan nearest resize requires FP32 Vulkan tensors");
  auto out=at::empty({input.size(0),input.size(1),size[0].expect_int(),size[1].expect_int()},input.options());
  if(out.numel()){
    auto x=packed(input);
    run("upsample_nearest2d",{base(x)},{write_base(out)},p,(out.numel()+127)/128);
  }
  return out;
}
at::Tensor backward(const at::Tensor& grad,c10::SymIntArrayRef size,c10::SymIntArrayRef input_size,
    std::optional<double> sh,std::optional<double> sw) {
  std::vector<int64_t> shape;for(const auto& d:input_size)shape.push_back(d.expect_int());
  auto p=params(shape,size,sh,sw,true);
  TORCH_CHECK(grad.device().type()==c10::DeviceType::PrivateUse1 && grad.scalar_type()==at::kFloat,"Vulkan nearest resize requires FP32 Vulkan tensors");
  TORCH_CHECK(grad.dim()==4 && grad.size(0)==shape[0] && grad.size(1)==shape[1] &&
      grad.size(2)==size[0].expect_int() && grad.size(3)==size[1].expect_int(),
      "Vulkan nearest2d gradient shape mismatch");
  auto out=at::empty(shape,grad.options());
  if(out.numel()){
    auto g=packed(grad);
    run("upsample_nearest2d",{base(g)},{write_base(out)},p,(out.numel()+127)/128);
  }
  return out;
}
} // namespace
TORCH_LIBRARY_IMPL(aten,PrivateUse1,m){
  if(!native::batch_norm_resize_enabled())return;
  m.impl("upsample_nearest2d",&upsample);
  m.impl("upsample_nearest2d_backward",&backward);
}
} // namespace torch_vulkan::ops
