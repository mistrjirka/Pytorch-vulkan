// Original FP32 ATen bilinear/bicubic wrappers, including training gradients.
#include "native_utils.h"
#include <torch/library.h>
#include <cmath>
namespace torch_vulkan::ops {
namespace {
using namespace native;
std::vector<uint32_t> params(at::IntArrayRef in,c10::SymIntArrayRef size,bool align,
 std::optional<double> sh,std::optional<double> sw,bool cubic,bool backward){
 TORCH_CHECK(in.size()==4 && size.size()==2,"Vulkan resize requires NCHW input and two output dimensions");
 int64_t oh=size[0].expect_int(),ow=size[1].expect_int();
 TORCH_CHECK(in[0]>=0 && in[1]>0 && in[2]>0 && in[3]>0 && oh>0 && ow>0,"Vulkan resize: invalid size");
 auto scale=[&](int64_t i,int64_t o,std::optional<double> s){
  double v=align?(o>1?double(i-1)/(o-1):0.0):(s && *s>0?1.0/ *s:double(i)/o);
  TORCH_CHECK(std::isfinite(v) && v>=0 && v*double(o)<double(INT32_MAX)-8,"Vulkan resize coordinate overflow");return float(v);};
 uint64_t planes=uint64_t(in[0])*in[1];
 TORCH_CHECK(in[2]<INT32_MAX-4 && in[3]<INT32_MAX-4 && oh<INT32_MAX-4 && ow<INT32_MAX-4,"Vulkan resize dimension overflow");
 TORCH_CHECK(uint64_t(in[2])*in[3]<=UINT32_MAX && planes<=UINT32_MAX/(uint64_t(in[2])*in[3]) && uint64_t(oh)*ow<=UINT32_MAX && planes<=UINT32_MAX/(uint64_t(oh)*ow),"Vulkan resize indexing overflow");
 return {u32(planes),u32(in[2]),u32(in[3]),u32(oh),u32(ow),uint32_t(backward),uint32_t(cubic),uint32_t(align),bits(scale(in[2],oh,sh)),bits(scale(in[3],ow,sw))};
}
template<bool Cubic> at::Tensor forward(const at::Tensor& input,c10::SymIntArrayRef size,bool align,std::optional<double> sh,std::optional<double> sw){
 if(input.scalar_type()!=at::kFloat || !can_pack_view(input))return (Cubic?at::upsample_bicubic2d_symint(input.cpu(),size,align,sh,sw):at::upsample_bilinear2d_symint(input.cpu(),size,align,sh,sw)).to(input.device());
 auto p=params(input.sizes(),size,align,sh,sw,Cubic,false);
 auto out=at::empty({input.size(0),input.size(1),int64_t(p[3]),int64_t(p[4])},input.options());
 if(out.numel()){auto x=packed(input);run("resize_linear_cubic",{base(x)},{base(out)},p,(out.numel()+127)/128);}return out;
}
template<bool Cubic> at::Tensor backward(const at::Tensor& grad,c10::SymIntArrayRef size,c10::SymIntArrayRef input_size,bool align,std::optional<double> sh,std::optional<double> sw){
 if(grad.scalar_type()!=at::kFloat || !can_pack_view(grad))return (Cubic?at::upsample_bicubic2d_backward_symint(grad.cpu(),size,input_size,align,sh,sw):at::upsample_bilinear2d_backward_symint(grad.cpu(),size,input_size,align,sh,sw)).to(grad.device());
 std::vector<int64_t> shape;for(auto d:input_size)shape.push_back(d.expect_int());
 auto p=params(shape,size,align,sh,sw,Cubic,true);
 TORCH_CHECK(grad.sizes()==at::IntArrayRef(std::vector<int64_t>{shape[0],shape[1],p[3],p[4]}),"Vulkan resize gradient shape mismatch");
 auto out=at::empty(shape,grad.options());if(out.numel()){auto g=packed(grad);run("resize_linear_cubic",{base(g)},{base(out)},p,(out.numel()+127)/128);}return out;
}
}
TORCH_LIBRARY_IMPL(aten,PrivateUse1,m){
 if(std::getenv("TORCH_VULKAN_DISABLE_NATIVE_SPATIAL"))return;
 m.impl("upsample_bilinear2d",&forward<false>);m.impl("upsample_bilinear2d_backward",&backward<false>);
 m.impl("upsample_bicubic2d",&forward<true>);m.impl("upsample_bicubic2d_backward",&backward<true>);
}
}
