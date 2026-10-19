#include "native_utils.h"
#include "device_copy.h"
#include <torch/library.h>
namespace torch_vulkan::ops {
namespace {
using namespace native;
at::Tensor pad_dispatch(const at::Tensor& data,const at::Tensor& original,c10::SymIntArrayRef padding,int spatial,bool reflect,bool backward){
 TORCH_CHECK(original.dim()==spatial+1||original.dim()==spatial+2,"padding input rank mismatch");TORCH_CHECK(padding.size()==spatial*2&&(data.scalar_type()==at::kFloat||(!backward&&device_copy_supported(data.scalar_type())))&&can_pack_view(data),"Vulkan padding requires supported input and matching padding sizes");
 auto output_shape=original.sizes().vec();std::vector<int64_t> left(spatial);
 for(int a=0;a<spatial;a++){int axis=original.dim()-1-a;int64_t l=padding[2*a].expect_int(),r=padding[2*a+1].expect_int(),n=original.size(axis);TORCH_CHECK(n>0&&n<INT32_MAX/4&&l>-INT32_MAX/4&&r>-INT32_MAX/4&&l<INT32_MAX/4&&r<INT32_MAX/4,"padding dimension overflow");TORCH_CHECK(!reflect||(l<n&&r<n),"reflection padding must be smaller than input dimension");TORCH_CHECK(n+l+r>0,"padding output dimension must be positive");left[a]=l;output_shape[axis]=n+l+r;}
 if(data.scalar_type()!=at::kFloat){
  auto result=data;
  for(int a=0;a<spatial;a++){
   int axis=original.dim()-1-a;auto n=original.size(axis);
   auto indices=at::arange(output_shape[axis],data.options().dtype(at::kLong)).sub(left[a]);
   if(reflect){indices=at::abs(indices);indices=at::where(indices.ge(n),at::rsub(indices,2*n-2),indices);}
   else indices=at::clamp(indices,0,n-1);
   result=at::index_select(result,axis,indices);
  }
  return result;
 }
 if(backward)TORCH_CHECK(data.device()==original.device()&&data.sizes()==at::IntArrayRef(output_shape),"padding gradient mismatch");
 auto x=data;for(int j=0;j<spatial;j++){int a=backward?spatial-1-j:j,axis=original.dim()-1-a;int64_t in=original.size(axis),out=output_shape[axis];auto shape=x.sizes().vec();shape[axis]=backward?in:out;auto result=at::empty(shape,data.options());
  if(result.numel()){uint64_t inner=1;for(int d=axis+1;d<x.dim();d++)inner*=x.size(d);auto packed_x=packed(x);run("pad_axis",{base(packed_x)},{write_base(result)},{u32(result.numel()),u32(in),u32(out),u32(inner),uint32_t(int32_t(left[a])),uint32_t(reflect),uint32_t(backward)},(result.numel()+127)/128);}x=result;
 }return x;
}
template<int Spatial,bool Reflect> at::Tensor forward(const at::Tensor& x,c10::SymIntArrayRef p){return pad_dispatch(x,x,p,Spatial,Reflect,false);}
template<int Spatial,bool Reflect> at::Tensor backward(const at::Tensor& g,const at::Tensor& x,c10::SymIntArrayRef p){return pad_dispatch(g,x,p,Spatial,Reflect,true);}
}
TORCH_LIBRARY_IMPL(aten,PrivateUse1,m){
 m.impl("reflection_pad1d",&forward<1,true>);m.impl("reflection_pad2d",&forward<2,true>);m.impl("reflection_pad3d",&forward<3,true>);
 m.impl("reflection_pad1d_backward",&backward<1,true>);m.impl("reflection_pad2d_backward",&backward<2,true>);m.impl("reflection_pad3d_backward",&backward<3,true>);
 m.impl("replication_pad1d",&forward<1,false>);m.impl("replication_pad2d",&forward<2,false>);m.impl("replication_pad3d",&forward<3,false>);
 m.impl("replication_pad1d_backward",&backward<1,false>);m.impl("replication_pad2d_backward",&backward<2,false>);m.impl("replication_pad3d_backward",&backward<3,false>);
}
}
