#include "native_utils.h"
#include <ATen/native/Pool.h>
#include <torch/library.h>
namespace torch_vulkan::ops {
namespace {
using namespace native;
std::array<int64_t,2> pair(at::IntArrayRef v){TORCH_CHECK(v.size()==1||v.size()==2,"Vulkan pooling: expected one or two dimensions");return {v[0],v.size()==1?v[0]:v[1]};}
std::vector<uint32_t> params(const at::Tensor& x,at::IntArrayRef kernel,at::IntArrayRef stride,at::IntArrayRef padding,bool ceil,bool include,std::optional<int64_t> divisor,at::IntArrayRef output,bool adaptive,bool backward){
 TORCH_CHECK(x.dim()==3||x.dim()==4,"Vulkan average pooling requires CHW/NCHW");
 int64_t h=x.size(-2),w=x.size(-1),c=x.size(-3);
 TORCH_CHECK(h<INT32_MAX/4&&w<INT32_MAX/4,"Vulkan pooling: spatial size overflow");
 int64_t planes=x.numel()/std::max<int64_t>(1,h*w);
 TORCH_CHECK(h>0&&w>0&&c>0,"Vulkan pooling: empty spatial/channel dimensions");
 auto k=adaptive?std::array<int64_t,2>{1,1}:pair(kernel),s=adaptive?k:(stride.empty()?k:pair(stride)),p=adaptive?std::array<int64_t,2>{0,0}:pair(padding);
 for(int i=0;i<2;i++)TORCH_CHECK(k[i]>0&&s[i]>0&&p[i]>=0&&k[i]<INT32_MAX/4&&s[i]<INT32_MAX/4&&p[i]<INT32_MAX/4,"Vulkan pooling: invalid geometry");
 TORCH_CHECK(!divisor || *divisor!=0,"divisor must be not zero");
 TORCH_CHECK(!adaptive || (output.size()==2&&output[0]>0&&output[1]>0),"Vulkan adaptive average pooling requires positive output dimensions");
 int64_t oh=adaptive?output[0]:at::native::pooling_output_shape<int64_t>(h,k[0],p[0],s[0],1,ceil),ow=adaptive?output[1]:at::native::pooling_output_shape<int64_t>(w,k[1],p[1],s[1],1,ceil);
 if(!adaptive)at::native::pool2d_shape_check(x,k[0],k[1],s[0],s[1],p[0],p[1],1,1,c,h,w,oh,ow,at::MemoryFormat::Contiguous);
 TORCH_CHECK(h<INT32_MAX/4&&w<INT32_MAX/4&&oh>0&&ow>0&&oh<INT32_MAX/4&&ow<INT32_MAX/4,"Vulkan pooling: size overflow");
 TORCH_CHECK(uint64_t(oh)*ow<=UINT32_MAX&&uint64_t(planes)<=UINT32_MAX/(uint64_t(oh)*ow),"Vulkan pooling: output indexing overflow");
 TORCH_CHECK(uint64_t(h)*oh+std::max(h,oh)<=UINT32_MAX&&uint64_t(w)*ow+std::max(w,ow)<=UINT32_MAX,"Vulkan pooling: adaptive indexing overflow");
 TORCH_CHECK(uint64_t(oh)*s[0]+k[0]<INT32_MAX&&uint64_t(ow)*s[1]+k[1]<INT32_MAX,"Vulkan pooling: window overflow");
 bool parallel=!backward && (adaptive?uint64_t(h)*w/(uint64_t(oh)*ow):uint64_t(k[0])*k[1])>=64;
 return {u32(planes),u32(h),u32(w),u32(oh),u32(ow),u32(k[0]),u32(k[1]),u32(s[0]),u32(s[1]),u32(p[0]),u32(p[1]),uint32_t(adaptive),uint32_t(backward),uint32_t(include),uint32_t(parallel),bits(divisor?float(*divisor):0.f)};
}
at::Tensor dispatch(const at::Tensor& input,const at::Tensor& shape_input,const std::vector<uint32_t>& p){
 auto shape=shape_input.sizes().vec();if(!p[12]){shape[shape.size()-2]=p[3];shape.back()=p[4];}
 auto out=at::empty(shape,input.options());if(out.numel()){auto x=packed(input);run("average_pool2d",{base(x)},{base(out)},p,p[14]?out.numel():(out.numel()+127)/128,512);}return out;
}
at::Tensor forward(const at::Tensor& x,at::IntArrayRef k,at::IntArrayRef s,at::IntArrayRef p,bool ceil,bool include,std::optional<int64_t> d){
 if(x.scalar_type()!=at::kFloat||!can_pack_view(x))return at::avg_pool2d(x.cpu(),k,s,p,ceil,include,d).to(x.device());
 return dispatch(x,x,params(x,k,s,p,ceil,include,d,{},false,false));
}
at::Tensor backward(const at::Tensor& g,const at::Tensor& x,at::IntArrayRef k,at::IntArrayRef s,at::IntArrayRef p,bool ceil,bool include,std::optional<int64_t> d){
 if(x.scalar_type()!=at::kFloat||g.scalar_type()!=at::kFloat||!can_pack_view(g)||!can_pack_view(x))return at::avg_pool2d_backward(g.cpu(),x.cpu(),k,s,p,ceil,include,d).to(x.device());
 auto q=params(x,k,s,p,ceil,include,d,{},false,true);auto shape=x.sizes().vec();shape[shape.size()-2]=q[3];shape.back()=q[4];TORCH_CHECK(g.device()==x.device()&&g.sizes()==at::IntArrayRef(shape),"Vulkan pooling gradient mismatch");return dispatch(g,x,q);
}
at::Tensor adaptive_forward(const at::Tensor& x,at::IntArrayRef size){
 if(x.scalar_type()!=at::kFloat||!can_pack_view(x)||(size.size()==2&&(size[0]==0||size[1]==0)))return at::_adaptive_avg_pool2d(x.cpu(),size).to(x.device());
 return dispatch(x,x,params(x,{},{},{},false,false,{},size,true,false));
}
at::Tensor adaptive_backward(const at::Tensor& g,const at::Tensor& x){
 if(x.scalar_type()!=at::kFloat||g.scalar_type()!=at::kFloat||!can_pack_view(g)||!can_pack_view(x))return at::_adaptive_avg_pool2d_backward(g.cpu(),x.cpu()).to(x.device());
 TORCH_CHECK(g.dim()==x.dim()&&g.device()==x.device()&&g.size(-3)==x.size(-3)&&(x.dim()==3||g.size(0)==x.size(0)),"Vulkan adaptive pooling gradient mismatch");
 return dispatch(g,x,params(x,{},{},{},false,false,{},g.sizes().slice(g.dim()-2),true,true));
}
}
TORCH_LIBRARY_IMPL(aten,PrivateUse1,m){
 if(std::getenv("TORCH_VULKAN_DISABLE_NATIVE_SPATIAL"))return;
 m.impl("avg_pool2d",&forward);m.impl("avg_pool2d_backward",&backward);
 m.impl("_adaptive_avg_pool2d",&adaptive_forward);m.impl("_adaptive_avg_pool2d_backward",&adaptive_backward);
}
}
