#include "native_utils.h"
#include <ATen/native/Pool.h>
#include <torch/library.h>
namespace torch_vulkan::ops {
namespace {
using namespace native;
std::array<int64_t,3> triple(at::IntArrayRef a){TORCH_CHECK(a.size()==1||a.size()==3,"3D pooling expects one or three values");return {a[0],a.size()==1?a[0]:a[1],a.size()==1?a[0]:a[2]};}
std::vector<uint32_t> parameters(const at::Tensor& x,at::IntArrayRef kernel,at::IntArrayRef stride,at::IntArrayRef padding,at::IntArrayRef dilation,bool ceil,bool maximum,bool include,std::optional<int64_t> divisor,at::IntArrayRef adaptive,bool backward){
 TORCH_CHECK((x.dim()==4||x.dim()==5)&&x.scalar_type()==at::kFloat&&can_pack_view(x),"Vulkan 3D pooling requires packable FP32 CDHW/NCDHW input");bool ad=!adaptive.empty();
 auto k=ad?std::array<int64_t,3>{1,1,1}:triple(kernel),s=ad?k:stride.empty()?k:triple(stride),p=ad?std::array<int64_t,3>{0,0,0}:triple(padding),d=ad?std::array<int64_t,3>{1,1,1}:triple(dilation);
 TORCH_CHECK(!divisor||*divisor!=0,"pool divisor must not be zero");TORCH_CHECK(!ad||adaptive.size()==3,"adaptive 3D pooling requires three output sizes");
 std::vector<uint32_t> params(24);params[0]=u32(x.dim()==5?x.size(0)*x.size(1):x.size(0));
 for(int a=0;a<3;a++){
  int64_t in=x.size(x.dim()-3+a);TORCH_CHECK(in>0&&in<INT32_MAX/4&&k[a]>0&&s[a]>0&&p[a]>=0&&d[a]>0&&k[a]<INT32_MAX/4&&s[a]<INT32_MAX/4&&p[a]<INT32_MAX/4&&d[a]<INT32_MAX/4,"invalid 3D pooling geometry");
  int64_t out=ad?adaptive[a]:at::native::pooling_output_shape<int64_t>(in,k[a],p[a],s[a],d[a],ceil);
  TORCH_CHECK(out>0&&out<INT32_MAX/4&&uint64_t(k[a])*d[a]+uint64_t(out)*s[a]<INT32_MAX,"3D pooling window overflow");
  TORCH_CHECK(uint64_t(in)*out+std::max(in,out)<=UINT32_MAX,"adaptive pooling coordinate overflow");
  params[1+a]=u32(in);params[4+a]=u32(out);params[7+a]=u32(k[a]);params[10+a]=u32(s[a]);params[13+a]=u32(p[a]);params[16+a]=u32(d[a]);
 }
 TORCH_CHECK(x.size(-4)>0,"pooling channel dimension must be nonempty");
 if(!ad)at::native::pool3d_shape_check(x,x.size(-4),k[0],k[1],k[2],s[0],s[1],s[2],p[0],p[1],p[2],d[0],d[1],d[2],params[1],params[2],params[3],params[4],params[5],params[6],"Vulkan pool3d",!maximum);
 uint64_t count=params[0];for(int a=0;a<3;a++){TORCH_CHECK(count<=UINT32_MAX/params[4+a],"pooling output overflow");count*=params[4+a];}
 params[19]=maximum;params[20]=backward;params[21]=ad;params[22]=include;params[23]=bits(divisor?float(*divisor):0);return params;
}
std::tuple<at::Tensor,at::Tensor> dispatch(const at::Tensor& input,const at::Tensor& shape_input,const at::Tensor& indices,const std::vector<uint32_t>& p){
 if(p[20]){auto expected=shape_input.sizes().vec();for(int a=0;a<3;a++)expected[expected.size()-3+a]=p[4+a];TORCH_CHECK(input.device()==shape_input.device()&&input.scalar_type()==at::kFloat&&input.sizes()==at::IntArrayRef(expected),"pool3d gradient mismatch");if(p[19])TORCH_CHECK(indices.device()==input.device()&&indices.scalar_type()==at::kLong&&indices.sizes()==input.sizes(),"pool3d indices mismatch");}
 auto shape=shape_input.sizes().vec();if(!p[20])for(int a=0;a<3;a++)shape[shape.size()-3+a]=p[4+a];auto out=at::empty(shape,input.options());auto arg=p[19]&&!p[20]?at::empty(shape,input.options().dtype(at::kLong)):at::Tensor();
 if(out.numel()){
  auto x=packed(input),idx=indices.defined()?packed(indices,at::kLong):at::Tensor();auto dummy=VulkanAllocator::instance().constant_float_buffer(2,0);
  run("pool3d_general",{base(x),idx.defined()?base(idx,at::kLong):dummy},{write_base(out),arg.defined()?write_base(arg,at::kLong):dummy},p,(out.numel()+127)/128);
 }return {out,arg};
}
at::Tensor avg_forward(const at::Tensor& x,at::IntArrayRef k,at::IntArrayRef s,at::IntArrayRef p,bool ceil,bool include,std::optional<int64_t> divisor){return std::get<0>(dispatch(x,x,{},parameters(x,k,s,p,{1},ceil,false,include,divisor,{},false)));}
at::Tensor avg_backward(const at::Tensor& g,const at::Tensor& x,at::IntArrayRef k,at::IntArrayRef s,at::IntArrayRef p,bool ceil,bool include,std::optional<int64_t> divisor){return std::get<0>(dispatch(g,x,{},parameters(x,k,s,p,{1},ceil,false,include,divisor,{},true)));}
std::tuple<at::Tensor,at::Tensor> max_forward(const at::Tensor& x,at::IntArrayRef k,at::IntArrayRef s,at::IntArrayRef p,at::IntArrayRef d,bool ceil){return dispatch(x,x,{},parameters(x,k,s,p,d,ceil,true,false,{}, {},false));}
at::Tensor max_backward(const at::Tensor& g,const at::Tensor& x,at::IntArrayRef k,at::IntArrayRef s,at::IntArrayRef p,at::IntArrayRef d,bool ceil,const at::Tensor& indices){return std::get<0>(dispatch(g,x,indices,parameters(x,k,s,p,d,ceil,true,false,{}, {},true)));}
at::Tensor adaptive_avg(const at::Tensor& x,at::IntArrayRef size){return std::get<0>(dispatch(x,x,{},parameters(x,{},{},{},{},false,false,false,{},size,false)));}
at::Tensor adaptive_avg_backward(const at::Tensor& g,const at::Tensor& x){return std::get<0>(dispatch(g,x,{},parameters(x,{},{},{},{},false,false,false,{},g.sizes().slice(g.dim()-3),true)));}
std::tuple<at::Tensor,at::Tensor> adaptive_max(const at::Tensor& x,at::IntArrayRef size){return dispatch(x,x,{},parameters(x,{},{},{},{},false,true,false,{},size,false));}
at::Tensor adaptive_max_backward(const at::Tensor& g,const at::Tensor& x,const at::Tensor& indices){return std::get<0>(dispatch(g,x,indices,parameters(x,{},{},{},{},false,true,false,{},g.sizes().slice(g.dim()-3),true)));}
}
TORCH_LIBRARY_IMPL(aten,PrivateUse1,m){m.impl("avg_pool3d",&avg_forward);m.impl("avg_pool3d_backward",&avg_backward);m.impl("max_pool3d_with_indices",&max_forward);m.impl("max_pool3d_with_indices_backward",&max_backward);m.impl("_adaptive_avg_pool3d",&adaptive_avg);m.impl("_adaptive_avg_pool3d_backward",&adaptive_avg_backward);m.impl("adaptive_max_pool3d",&adaptive_max);m.impl("adaptive_max_pool3d_backward",&adaptive_max_backward);}
}
