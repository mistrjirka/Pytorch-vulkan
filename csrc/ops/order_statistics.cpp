#include "native_utils.h"
#include <torch/library.h>
#include <ATen/native/Resize.h>
#include <ATen/MemoryOverlap.h>
namespace torch_vulkan::ops {
namespace {
using namespace native;
int64_t axis(const at::Tensor& x,int64_t d){if(d<0)d+=std::max<int64_t>(1,x.dim());TORCH_CHECK(d>=0&&d<std::max<int64_t>(1,x.dim()),"dimension out of range");return d;}
std::tuple<at::Tensor,at::Tensor> sort_native(const at::Tensor& x,int64_t raw,bool descending){
 int64_t d=axis(x,raw);TORCH_CHECK(x.scalar_type()==at::kFloat&&can_pack_view(x),"Vulkan sort requires packable FP32 data");
 auto values=at::empty(x.sizes(),x.options()),indices=at::empty(x.sizes(),x.options().dtype(at::kLong));
 if(x.numel()){
  uint64_t inner=1;for(int a=d+1;a<x.dim();a++)inner*=x.size(a);
  run("sort_axis",{base(packed(x))},{write_base(values),write_base(indices,at::kLong)},
   {u32(x.numel()),u32(x.dim()?x.size(d):1),u32(inner),uint32_t(descending)},(x.numel()+127)/128);
 }return {values,indices};
}
std::tuple<at::Tensor,at::Tensor> sort_stable_native(const at::Tensor& x,std::optional<bool>,int64_t d,bool desc){return sort_native(x,d,desc);}
std::tuple<at::Tensor&,at::Tensor&> sort_out_native(const at::Tensor& x,int64_t d,bool desc,at::Tensor& values,at::Tensor& indices){
 TORCH_CHECK(values.device()==x.device()&&indices.device()==x.device()&&values.scalar_type()==x.scalar_type()&&indices.scalar_type()==at::kLong,"sort output mismatch");
 at::assert_no_internal_overlap(values);at::assert_no_internal_overlap(indices);auto [v,i]=sort_native(x,d,desc);at::native::resize_output(values,v.sizes());at::native::resize_output(indices,i.sizes());values.copy_(v);indices.copy_(i);return {values,indices};
}
std::tuple<at::Tensor&,at::Tensor&> sort_stable_out_native(const at::Tensor& x,std::optional<bool>,int64_t d,bool desc,at::Tensor& v,at::Tensor& i){return sort_out_native(x,d,desc,v,i);}
std::tuple<at::Tensor,at::Tensor> topk_native(const at::Tensor& x,c10::SymInt k_,int64_t raw,bool largest,bool){
 int64_t d=axis(x,raw),k=k_.expect_int();TORCH_CHECK(k>=0&&k<=(x.dim()?x.size(d):1),"topk k out of range");auto [v,i]=sort_native(x,d,largest);
 if(x.dim()){v=v.narrow(d,0,k);i=i.narrow(d,0,k);if(k){v=packed(v);i=packed(i,at::kLong);}}
 return {v,i};
}
at::Tensor scan_native(const at::Tensor& x,int64_t raw,std::optional<at::ScalarType> dtype){
 int64_t d=axis(x,raw);TORCH_CHECK(x.scalar_type()==at::kFloat&&dtype.value_or(at::kFloat)==at::kFloat&&can_pack_view(x),"Vulkan cumsum requires packable FP32 input/output");
 auto out=at::empty(x.sizes(),x.options());if(!x.numel())return out;uint64_t inner=1;for(int a=d+1;a<x.dim();a++)inner*=x.size(a);
 int64_t n=x.dim()?x.size(d):1,rows=x.numel()/n;run("scan_axis",{base(packed(x))},{write_base(out)},{u32(rows),u32(n),u32(inner)},rows,516);return out;
}
at::Tensor flip_native(const at::Tensor& x,at::IntArrayRef dims){
 TORCH_CHECK((x.scalar_type()==at::kFloat||x.scalar_type()==at::kLong||x.scalar_type()==at::kInt)&&can_pack_view(x),"Vulkan flip supports FP32/int32/int64");
 uint32_t mask=0;for(auto raw:dims){auto d=axis(x,raw);TORCH_CHECK(!(mask&(1u<<d)),"flip duplicate dim");mask|=1u<<d;}
 auto out=at::empty(x.sizes(),x.options());if(out.numel()){
  std::vector<uint32_t> p={u32(x.numel()),mask,uint32_t(x.element_size()/4)};p.resize(11,1);for(int d=0;d<x.dim();d++)p[3+d]=u32(x.size(d));
  run("flip",{base(packed(x,x.scalar_type()),x.scalar_type())},{write_base(out,out.scalar_type())},p,(out.numel()+127)/128);
 }return out;
}
at::Tensor triangle(const at::Tensor& x,int64_t diagonal,bool upper){
 TORCH_CHECK(x.dim()>=2&&can_pack_view(x),"triangular requires packable rank>=2 tensor");auto out=at::empty(x.sizes(),x.options());if(!out.numel())return out;
 diagonal=std::clamp<int64_t>(diagonal,INT32_MIN,INT32_MAX);
 run("triangular",{base(packed(x,x.scalar_type()),x.scalar_type())},{write_base(out,out.scalar_type())},
  {u32(x.numel()),u32(x.size(-2)),u32(x.size(-1)),uint32_t(x.element_size()/4),uint32_t(x.scalar_type()==at::kBool),uint32_t(int32_t(diagonal)),uint32_t(upper)},(x.numel()+(x.scalar_type()==at::kBool?511:127))/(x.scalar_type()==at::kBool?512:128));return out;
}
at::Tensor tril_native(const at::Tensor& x,int64_t d){return triangle(x,d,false);}
at::Tensor triu_native(const at::Tensor& x,int64_t d){return triangle(x,d,true);}
at::Tensor& triangle_out(const at::Tensor& x,int64_t d,at::Tensor& out,bool upper){TORCH_CHECK(x.device()==out.device()&&x.scalar_type()==out.scalar_type(),"triangular output mismatch");at::assert_no_internal_overlap(out);auto result=triangle(x,d,upper);at::native::resize_output(out,result.sizes());out.copy_(result);return out;}
at::Tensor& tril_out(const at::Tensor& x,int64_t d,at::Tensor& out){return triangle_out(x,d,out,false);}
at::Tensor& triu_out(const at::Tensor& x,int64_t d,at::Tensor& out){return triangle_out(x,d,out,true);}
at::Tensor& tril_inplace(at::Tensor& x,int64_t d){return triangle_out(x,d,x,false);}
at::Tensor& triu_inplace(at::Tensor& x,int64_t d){return triangle_out(x,d,x,true);}
std::tuple<at::Tensor,at::Tensor> var_mean_native(const at::Tensor& x,at::OptionalIntArrayRef dims,const std::optional<at::Scalar>& correction,bool keepdim){
 TORCH_CHECK(x.scalar_type()==at::kFloat&&can_pack_view(x),"Vulkan variance requires packable FP32 data");
 auto mean=at::mean(x,dims,true);auto diff=x.sub(mean);auto squares=diff.mul(diff);
 int64_t n=1;if(!dims||dims->empty())n=x.numel();else for(auto d:*dims)n*=x.dim()?x.size(axis(x,d)):1;
 auto variance=at::sum(squares,dims,keepdim).div(std::max(0.0,double(n)-correction.value_or(at::Scalar(1)).toDouble()));
 return {variance,mean.reshape(variance.sizes())};
}
at::Tensor var_native(const at::Tensor& x,at::OptionalIntArrayRef dims,const std::optional<at::Scalar>& correction,bool keepdim){return std::get<0>(var_mean_native(x,dims,correction,keepdim));}
at::Tensor var_dim_native(const at::Tensor& x,at::OptionalIntArrayRef dims,bool unbiased,bool keepdim){return var_native(x,dims,at::Scalar(unbiased?1:0),keepdim);}
at::Tensor var_all_native(const at::Tensor& x,bool unbiased){return var_dim_native(x,std::nullopt,unbiased,false);}
}
TORCH_LIBRARY_IMPL(aten,PrivateUse1,m){
 m.impl("sort",&sort_native);m.impl("sort.stable",&sort_stable_native);m.impl("sort.values",&sort_out_native);m.impl("sort.values_stable",&sort_stable_out_native);
 m.impl("tril",&tril_native);m.impl("triu",&triu_native);m.impl("tril.out",&tril_out);m.impl("triu.out",&triu_out);m.impl("tril_",&tril_inplace);m.impl("triu_",&triu_inplace);
 m.impl("topk",&topk_native);m.impl("cumsum",&scan_native);m.impl("flip",&flip_native);
 m.impl("var.correction",&var_native);m.impl("var.dim",&var_dim_native);m.impl("var",&var_all_native);m.impl("var_mean.correction",&var_mean_native);
}
}
