#include "native_utils.h"
#include <torch/library.h>
namespace torch_vulkan::ops {
namespace {
using namespace native;
std::vector<uint32_t> parameters(const at::Tensor& x,const at::Tensor& grid,int64_t mode,int64_t padding,bool align,int spatial){
 TORCH_CHECK(x.dim()==spatial+2&&grid.dim()==spatial+2&&grid.size(-1)==spatial&&x.size(0)==grid.size(0)&&x.device()==grid.device()&&x.scalar_type()==at::kFloat&&grid.scalar_type()==at::kFloat&&can_pack_view(x)&&can_pack_view(grid),"Vulkan grid sample requires matching FP32 inputs and grid");
 TORCH_CHECK(mode>=0&&mode<=(spatial==2?2:1)&&padding>=0&&padding<=2,"grid sampler invalid mode");for(int d=2;d<x.dim();d++)TORCH_CHECK(x.size(d)>0&&x.size(d)<INT32_MAX/4,"grid sampler invalid input spatial size");
 return {u32(x.size(0)),u32(x.size(1)),spatial==3?u32(x.size(2)):1u,u32(x.size(-2)),u32(x.size(-1)),spatial==3?u32(grid.size(1)):1u,u32(grid.size(-3)),u32(grid.size(-2)),uint32_t(spatial),uint32_t(mode),uint32_t(padding),uint32_t(align),0};
}
template<int Spatial> at::Tensor forward(const at::Tensor& x,const at::Tensor& grid,int64_t mode,int64_t padding,bool align){
 auto p=parameters(x,grid,mode,padding,align,Spatial);auto shape=x.sizes().vec();for(int a=0;a<Spatial;a++)shape[a+2]=grid.size(a+1);auto out=at::empty(shape,x.options());if(!out.numel())return out;
 auto input=packed(x),g=packed(grid);auto dummy=VulkanAllocator::instance().constant_float_buffer(1,0);run("grid_sample_general",{base(input),base(g),dummy},{base(out)},p,(out.numel()+127)/128);return out;
}
template<int Spatial> std::tuple<at::Tensor,at::Tensor> backward(const at::Tensor& grad,const at::Tensor& x,const at::Tensor& grid,int64_t mode,int64_t padding,bool align,std::array<bool,2> mask){
 auto p=parameters(x,grid,mode,padding,align,Spatial);auto expected=x.sizes().vec();for(int a=0;a<Spatial;a++)expected[a+2]=grid.size(a+1);TORCH_CHECK(grad.device()==x.device()&&grad.scalar_type()==at::kFloat&&grad.sizes()==at::IntArrayRef(expected),"grid sample gradient mismatch");
 auto dx=mask[0]?at::empty(x.sizes(),x.options()):at::Tensor(),dg=at::empty(grid.sizes(),grid.options());
 if(!grad.numel()){if(dx.defined())dx.zero_();dg.zero_();return {dx,dg};}
 auto input=packed(x),g=packed(grid),go=packed(grad);std::vector<Buffer> buffers={base(input),base(g),base(go)};
 if(dx.defined()&&dx.numel()){p[12]=1;run("grid_sample_general",buffers,{base(dx)},p,(dx.numel()+127)/128);}
 if(dg.numel()){p[12]=2;run("grid_sample_general",buffers,{base(dg)},p,(dg.numel()/Spatial+127)/128);}return {dx,dg};
}
}
TORCH_LIBRARY_IMPL(aten,PrivateUse1,m){m.impl("grid_sampler_2d",&forward<2>);m.impl("grid_sampler_3d",&forward<3>);m.impl("grid_sampler_2d_backward",&backward<2>);m.impl("grid_sampler_3d_backward",&backward<3>);}
}
