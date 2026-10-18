#include "native_utils.h"
#include <torch/library.h>
#include <cmath>
namespace torch_vulkan::ops {
namespace {
using namespace native;
at::Tensor resample_axis(const at::Tensor& x,int64_t axis,int64_t input,int64_t output,bool align,std::optional<double> factor,uint32_t mode,bool backward){
 TORCH_CHECK(x.scalar_type()==at::kFloat&&can_pack_view(x)&&input>0&&output>0,"Vulkan resampling requires positive spatial sizes and FP32 input");
 double scale=align?(output>1?double(input-1)/(output-1):0):(factor&&*factor>0?1/ *factor:double(input)/output);
 TORCH_CHECK(std::isfinite(scale)&&scale>=0&&scale*output<INT32_MAX/4,"resampling coordinate overflow");
 auto shape=x.sizes().vec();shape[axis]=backward?input:output;auto out=at::empty(shape,x.options());if(!out.numel())return out;
 uint64_t inner=1;for(int d=axis+1;d<x.dim();d++)inner*=x.size(d);auto src=packed(x);
 run("resample_axis",{base(src)},{base(out)},{u32(out.numel()),u32(input),u32(output),u32(inner),bits(float(scale)),uint32_t(align),mode,uint32_t(backward)},(out.numel()+127)/128);return out;
}
at::Tensor resize(const at::Tensor& x,c10::SymIntArrayRef output,c10::SymIntArrayRef input,bool align,const std::vector<std::optional<double>>& scales,uint32_t mode,bool backward){
 int spatial=scales.size();TORCH_CHECK(x.dim()==spatial+2&&output.size()==spatial,"resampling input/output rank mismatch");std::vector<int64_t> shape;
 if(backward){TORCH_CHECK(input.size()==x.dim(),"resampling backward input size mismatch");for(auto d:input)shape.push_back(d.expect_int());}else shape=x.sizes().vec();
 TORCH_CHECK(shape[0]>=0&&shape[1]>0,"resampling batch/channel mismatch");for(int d=0;d<spatial;d++)TORCH_CHECK(shape[d+2]>0&&output[d].expect_int()>0,"resampling spatial dimensions must be positive");
 if(backward){TORCH_CHECK(x.size(0)==shape[0]&&x.size(1)==shape[1],"resampling gradient batch mismatch");for(int d=0;d<spatial;d++)TORCH_CHECK(x.size(d+2)==output[d].expect_int(),"resampling gradient spatial mismatch");}
 auto result=x;for(int j=0;j<spatial;j++){int d=backward?j:spatial-1-j;result=resample_axis(result,d+2,shape[d+2],output[d].expect_int(),align,scales[d],mode,backward);}return result;
}
at::Tensor trilinear(const at::Tensor& x,c10::SymIntArrayRef s,bool align,std::optional<double> d,std::optional<double> h,std::optional<double> w){return resize(x,s,{},align,{d,h,w},0,false);}
at::Tensor trilinear_backward(const at::Tensor& g,c10::SymIntArrayRef s,c10::SymIntArrayRef input,bool align,std::optional<double> d,std::optional<double> h,std::optional<double> w){return resize(g,s,input,align,{d,h,w},0,true);}
template<uint32_t Mode> at::Tensor aa_forward(const at::Tensor& x,c10::SymIntArrayRef s,bool align,std::optional<double> h,std::optional<double> w){return resize(x,s,{},align,{h,w},Mode,false);}
template<uint32_t Mode> at::Tensor aa_backward(const at::Tensor& g,c10::SymIntArrayRef s,c10::SymIntArrayRef input,bool align,std::optional<double> h,std::optional<double> w){return resize(g,s,input,align,{h,w},Mode,true);}
}
TORCH_LIBRARY_IMPL(aten,PrivateUse1,m){m.impl("upsample_trilinear3d",&trilinear);m.impl("upsample_trilinear3d_backward",&trilinear_backward);m.impl("_upsample_bilinear2d_aa",&aa_forward<1>);m.impl("_upsample_bilinear2d_aa_backward",&aa_backward<1>);m.impl("_upsample_bicubic2d_aa",&aa_forward<2>);m.impl("_upsample_bicubic2d_aa_backward",&aa_backward<2>);}
}
