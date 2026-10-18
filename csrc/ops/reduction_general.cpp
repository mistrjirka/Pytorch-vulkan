#include "native_utils.h"
#include <ATen/native/Resize.h>
#include <ATen/MemoryOverlap.h>
#include <torch/library.h>
#include <cmath>
namespace torch_vulkan::ops {
namespace {
using namespace native;
at::Tensor reduce_general(const at::Tensor& x,at::OptionalIntArrayRef dims,bool keep,uint32_t mode,float ord){
 TORCH_CHECK(x.scalar_type()==at::kFloat&&can_pack_view(x),"Vulkan reduction requires packable FP32 input");
 int rank=std::max<int64_t>(1,x.dim());uint32_t mask=0;
 if(!dims||dims->empty())mask=(1u<<rank)-1;else for(auto raw:*dims){int64_t d=raw<0?raw+rank:raw;TORCH_CHECK_INDEX(d>=0&&d<rank,"Dimension out of range");TORCH_CHECK(!(mask&(1u<<d)),"dim appears multiple times in the list of dims");mask|=1u<<d;}
 std::vector<int64_t> shape;uint64_t reduced=1;
 for(int d=0;d<x.dim();d++){if(mask&(1u<<d)){reduced*=x.size(d);if(keep)shape.push_back(1);}else shape.push_back(x.size(d));}
 if(!reduced)TORCH_CHECK(mode==0&&ord>=0&&ord!=INFINITY,"reduction over an empty dimension has no identity");
 auto out=at::empty(shape,x.options());if(!out.numel())return out;
 if(!reduced){out.zero_();return out;}
 auto input=packed(x);std::vector<uint32_t> p={u32(out.numel()),u32(reduced),mask,mode,bits(ord)};p.resize(13,1);for(int d=0;d<x.dim();d++)p[5+d]=u32(x.size(d));
 run("reduction_general",{base(input)},{write_base(out)},p,(out.numel()+127)/128);return out;
}
at::Tensor norm(const at::Tensor& x,const at::Scalar& order,at::OptionalIntArrayRef dims,bool keep,std::optional<at::ScalarType> dtype){TORCH_CHECK(dtype.value_or(x.scalar_type())==at::kFloat,"Vulkan norm requires FP32 output");return reduce_general(x,dims,keep,0,order.toFloat());}
at::Tensor& norm_out(const at::Tensor& x,const at::Scalar& order,at::OptionalIntArrayRef dims,bool keep,std::optional<at::ScalarType> dtype,at::Tensor& out){TORCH_CHECK(out.device()==x.device()&&out.scalar_type()==at::kFloat,"norm output device/dtype mismatch");at::assert_no_internal_overlap(out);at::assert_no_partial_overlap(out,x);auto y=norm(x,order,dims,keep,dtype);at::native::resize_output(out,y.sizes());out.copy_(y);return out;}
at::Tensor& legacy_norm_out(const at::Tensor& x,const std::optional<at::Scalar>& order,at::IntArrayRef dims,bool keep,at::Tensor& out){return norm_out(x,order.value_or(2),dims,keep,std::nullopt,out);}
template<bool Max>at::Tensor extrema(const at::Tensor& x,at::IntArrayRef dims,bool keep){return reduce_general(x,dims,keep,Max?1:2,0);}
template<bool Max>at::Tensor& extrema_out(const at::Tensor& x,at::IntArrayRef dims,bool keep,at::Tensor& out){TORCH_CHECK(out.device()==x.device()&&out.scalar_type()==x.scalar_type(),"extrema output device/dtype mismatch");at::assert_no_internal_overlap(out);at::assert_no_partial_overlap(out,x);auto y=extrema<Max>(x,dims,keep);at::native::resize_output(out,y.sizes());out.copy_(y);return out;}
at::Tensor dot(const at::Tensor& x,const at::Tensor& y){TORCH_CHECK(x.dim()==1&&y.dim()==1,"1D tensors expected for dot");TORCH_CHECK(x.numel()==y.numel(),"inconsistent tensor size for dot");TORCH_CHECK(x.device()==y.device()&&x.scalar_type()==y.scalar_type(),"dot input device/dtype mismatch");return at::sum(at::mul(x,y),x.scalar_type());}
}
TORCH_LIBRARY_IMPL(aten,PrivateUse1,m){m.impl("linalg_vector_norm",&norm);m.impl("linalg_vector_norm.out",&norm_out);m.impl("norm.out",&legacy_norm_out);m.impl("amax",&extrema<true>);m.impl("amin",&extrema<false>);m.impl("amax.out",&extrema_out<true>);m.impl("amin.out",&extrema_out<false>);m.impl("dot",&dot);m.impl("vdot",&dot);}
}
