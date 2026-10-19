#include "convolution_nd.h"
#include "native_utils.h"
#include <ATen/Functions.h>
namespace torch_vulkan::ops {
namespace {
using namespace native;
struct Geometry {
 std::vector<uint32_t> p=std::vector<uint32_t>(27,0);
 std::vector<int64_t> shape;
};
Geometry geometry(const at::Tensor& x,const at::Tensor& w,c10::SymIntArrayRef stride,c10::SymIntArrayRef pad,
 c10::SymIntArrayRef dilation,bool transposed,c10::SymIntArrayRef output_pad,c10::SymInt groups_) {
 const int rank=x.dim()-2;const int64_t groups=groups_.expect_int();
 TORCH_CHECK(rank>=1&&rank<=3&&w.dim()==x.dim(),"convolution: expected matching rank 3, 4 or 5 tensors");
 TORCH_CHECK((x.scalar_type()==at::kFloat||x.scalar_type()==at::kLong)&&w.scalar_type()==x.scalar_type()&&x.device()==w.device(),"convolution: expected matching FP32/int64 tensors on the same device");
 TORCH_CHECK(stride.size()==rank&&pad.size()==rank&&dilation.size()==rank&&output_pad.size()==rank,"convolution: spatial parameter rank mismatch");
 TORCH_CHECK(groups>0&&x.size(1)>0&&x.size(1)%groups==0,"convolution: invalid groups/input channels");
 for(auto s:w.sizes())TORCH_CHECK(s>0,"convolution: empty weight dimension");
 int64_t co=transposed?w.size(1)*groups:w.size(0);
 TORCH_CHECK(co%groups==0&&(transposed?w.size(0)==x.size(1):w.size(1)*groups==x.size(1)),"convolution: weight/channel/group mismatch");
 TORCH_CHECK(can_pack_view(x)&&can_pack_view(w),"convolution: tensor addressing exceeds Vulkan limits");
 Geometry g;g.p[0]=u32(x.size(0));g.p[1]=u32(x.size(1));g.p[2]=u32(co);g.p[3]=u32(groups);
 g.shape={x.size(0),co};g.p[22]=transposed;
 for(int a=0;a<3;a++) {g.p[4+a]=g.p[7+a]=g.p[10+a]=g.p[13+a]=g.p[19+a]=1;}
 for(int a=0;a<rank;a++) {
  int d=3-rank+a;int64_t in=x.size(a+2),k=w.size(a+2),s=stride[a].expect_int(),p=pad[a].expect_int(),di=dilation[a].expect_int(),op=output_pad[a].expect_int();
  TORCH_CHECK(in>0&&s>0&&di>0&&p>=0&&op>=0,"convolution: invalid spatial parameters");
  TORCH_CHECK(transposed?(op<s||op<di):op==0,"convolution: invalid output padding");
  // Keep all signed coordinate intermediates representable in the shader.
  TORCH_CHECK(in<=INT32_MAX&&k<=INT32_MAX&&s<=INT32_MAX&&p<=INT32_MAX/2&&di<=INT32_MAX&&
   (k-1)<=INT32_MAX/di&&(in-1)<=INT32_MAX/s,"convolution: spatial addressing exceeds int32");
  int64_t effective=(k-1)*di+1;
  int64_t out=transposed?(in-1)*s-2*p+effective+op:(in+2*p-effective)/s+1;
  TORCH_CHECK((transposed||in+2*p>=effective)&&out>0&&out<=INT32_MAX&&out*s+effective+p<=INT32_MAX,"convolution: invalid or oversized output shape");
  g.p[4+d]=u32(in);g.p[7+d]=u32(out);g.p[10+d]=u32(k);g.p[13+d]=u32(s);g.p[16+d]=u32(p);g.p[19+d]=u32(di);g.shape.push_back(out);
 }
 return g;
}
Buffer buf(const at::Tensor& t){return t.defined()&&t.numel()?base(t):VulkanAllocator::instance().constant_float_buffer(1,0);}
void execute(Geometry g,uint32_t mode,const at::Tensor& x,const at::Tensor& w,const at::Tensor& grad,const at::Tensor& bias,at::Tensor& out) {
 if(!out.numel())return;
 g.p[23]=mode;g.p[24]=u32(out.numel());g.p[25]=bias.defined();
 run("convolution_nd",{buf(x),buf(w),buf(grad),buf(bias)},{write_base(out)},g.p,(out.numel()+127)/128);
}
}
at::Tensor convolution_nd(const at::Tensor& x,const at::Tensor& w,const std::optional<at::Tensor>& bias,
 c10::SymIntArrayRef stride,c10::SymIntArrayRef pad,c10::SymIntArrayRef dilation,bool transposed,c10::SymIntArrayRef output_pad,c10::SymInt groups) {
 auto g=geometry(x,w,stride,pad,dilation,transposed,output_pad,groups);auto out=at::empty(g.shape,x.options());
 at::Tensor b;
 if(bias&&bias->defined()) {TORCH_CHECK(bias->dim()==1&&bias->numel()==g.p[2]&&bias->device()==x.device()&&bias->scalar_type()==x.scalar_type(),"convolution: invalid bias");b=packed(*bias,x.scalar_type());}
 if(out.numel()&&x.scalar_type()==at::kLong){
  auto xc=packed(x,at::kLong),wc=packed(w,at::kLong);g.p[24]=u32(out.numel());g.p[25]=b.defined();
  auto bb=b.defined()?base(b,at::kLong):VulkanAllocator::instance().constant_float_buffer(2,0);
  run("convolution_nd_int64",{base(xc,at::kLong),base(wc,at::kLong),bb},{write_base(out,at::kLong)},g.p,(out.numel()+127)/128);
 }else if(out.numel())execute(g,0,packed(x),packed(w),{},b,out);
 return out;
}
std::tuple<at::Tensor,at::Tensor,at::Tensor> convolution_nd_backward(const at::Tensor& grad,const at::Tensor& x,const at::Tensor& w,
 c10::SymIntArrayRef stride,c10::SymIntArrayRef pad,c10::SymIntArrayRef dilation,bool transposed,c10::SymIntArrayRef output_pad,c10::SymInt groups,std::array<bool,3> mask) {
 auto g=geometry(x,w,stride,pad,dilation,transposed,output_pad,groups);
 TORCH_CHECK(grad.sizes()==at::IntArrayRef(g.shape)&&grad.scalar_type()==at::kFloat&&grad.device()==x.device()&&can_pack_view(grad),"convolution backward: invalid gradient");
 auto xc=x.numel()?packed(x):x,wc=packed(w),gc=grad.numel()?packed(grad):grad;at::Tensor dx,dw,db;
 if(mask[0]){dx=at::empty(x.sizes(),x.options());execute(g,1,xc,wc,gc,{},dx);}
 if(mask[1]){dw=at::empty(w.sizes(),w.options());execute(g,2,xc,wc,gc,{},dw);}
 if(mask[2]){db=at::empty({g.p[2]},w.options());execute(g,3,xc,wc,gc,{},db);}
 return {dx,dw,db};
}
}
