#include "native_utils.h"
#include "fft.h"
#include <ATen/ops/view_as_real_native.h>
#include <ATen/ops/view_as_complex_native.h>
#include <torch/library.h>
#include <ATen/ExpandUtils.h>
#include <cmath>
namespace torch_vulkan::ops {
namespace {
using namespace native;
at::Tensor complex_unary(const at::Tensor& x,const at::Scalar& value,uint32_t mode){
 TORCH_CHECK(x.scalar_type()==at::kComplexFloat&&can_pack_view(x),"Vulkan complex operation requires packable complex64");auto out=at::empty(x.sizes(),x.options());if(!out.numel())return out;
 auto z=value.toComplexFloat();auto input=packed(x,at::kComplexFloat);auto in=base(input,at::kComplexFloat);
 run(mode<3?"complex_unary":"complex_math_unary",{in},{write_base(out,at::kComplexFloat)},{u32(x.numel()),mode,bits(z.real()),bits(z.imag())},(out.numel()+127)/128);return out;
}
at::Tensor complex_construct(const at::Tensor& real,const at::Tensor& imag){TORCH_CHECK(real.scalar_type()==imag.scalar_type(),"Expected both inputs to be Half, Float or Double tensors but got ",real.scalar_type()," and ",imag.scalar_type());TORCH_CHECK(real.scalar_type()==at::kFloat&&real.device()==imag.device(),"complex requires FP32 tensors on the same device");auto shape=at::infer_size_dimvector(real.sizes(),imag.sizes());return at::view_as_complex(at::stack({real.expand(shape),imag.expand(shape)},-1));}
at::Tensor polar_construct(const at::Tensor& radius,const at::Tensor& angle){return complex_construct(radius.mul(at::cos(angle)),radius.mul(at::sin(angle)));}
at::Tensor complex_conj(const at::Tensor& x){if(!x.is_complex())return x.clone();return complex_unary(x,0,2);}
std::vector<int64_t> axes(const at::Tensor& x,at::IntArrayRef dim){
 TORCH_CHECK(!dim.empty()&&x.dim()<=8,"FFT requires at least one axis and rank <=8");std::vector<int64_t> result;uint32_t used=0;
 for(auto d:dim){if(d<0)d+=x.dim();TORCH_CHECK(d>=0&&d<x.dim()&&!(used&(1u<<d))&&x.size(d)>0,"invalid FFT axes");used|=1u<<d;result.push_back(d);}return result;
}
at::Tensor finish(const at::Tensor& x,int64_t axis,int64_t length,bool real,float scale){
 auto shape=x.sizes().vec();shape[axis]=length;auto out=at::empty(shape,x.options().dtype(real?at::kFloat:at::kComplexFloat));if(!out.numel())return out;
 uint64_t inner=1;for(int d=axis+1;d<x.dim();d++)inner*=x.size(d);
 run("fft_finish",{base(x,at::kComplexFloat)},{write_base(out,out.scalar_type())},{u32(out.numel()),u32(x.size(axis)),u32(length),u32(inner),uint32_t(real),bits(scale)},(out.numel()+127)/128);return out;
}
at::Tensor transform(const at::Tensor& source,int64_t axis,bool inverse,int64_t real_length=0){
 TORCH_CHECK((source.scalar_type()==at::kFloat||source.scalar_type()==at::kComplexFloat)&&can_pack_view(source),"FFT requires packable FP32/complex64");
 auto x=source;int64_t n=real_length?real_length:x.size(axis);TORCH_CHECK(n>0&&n<=INT32_MAX,"FFT length out of range");
 auto shape=x.sizes().vec();shape[axis]=n;auto out=at::empty(shape,x.options().dtype(at::kComplexFloat));if(!out.numel())return out;
 x=packed(x,x.scalar_type());
 uint64_t inner=1;for(int d=axis+1;d<x.dim();d++)inner*=x.size(d);
 uint32_t mode=real_length?2:(source.is_complex()?1:0);
 if((n&(n-1))==0){
  uint32_t log=0;while((uint64_t(1)<<log)<uint64_t(n))log++;
  run("fft_prepare",{base(x,x.scalar_type())},{write_base(out,at::kComplexFloat)},
   {u32(out.numel()),u32(n),u32(inner),u32(x.size(axis)),mode,log},(out.numel()+127)/128);
  for(uint32_t stage=1;stage<=log;stage++){
   auto next=at::empty_like(out);run("fft_butterfly",{base(out,at::kComplexFloat)},{write_base(next,at::kComplexFloat)},
    {u32(out.numel()),u32(n),u32(inner),1u<<stage,uint32_t(inverse)},(out.numel()+127)/128);out=next;
  }
 }else{
  run("fft_direct",{base(x,x.scalar_type())},{write_base(out,at::kComplexFloat)},
   {u32(out.numel()),u32(n),u32(inner),u32(x.size(axis)),mode,uint32_t(inverse)},(out.numel()+127)/128);
 }
 return out;
}
float normalization(int64_t count,int64_t norm){TORCH_CHECK(norm>=0&&norm<=2,"FFT normalization invalid");return norm==0?1.f:norm==1?float(1/std::sqrt(double(count))):float(1/double(count));}
at::Tensor r2c(const at::Tensor& x,at::IntArrayRef dim,int64_t norm,bool onesided){
 TORCH_CHECK(x.scalar_type()==at::kFloat,"FFT real input must be FP32");auto ds=axes(x,dim);int64_t count=1;auto y=x;
 for(auto d:ds){count*=x.size(d);y=transform(y,d,false);}
 float scale=normalization(count,norm);if(onesided||scale!=1)y=finish(y,ds.back(),onesided?x.size(ds.back())/2+1:x.size(ds.back()),false,scale);return y;
}
at::Tensor c2c(const at::Tensor& x,c10::SymIntArrayRef dim,int64_t norm,bool forward){
 TORCH_CHECK(x.scalar_type()==at::kComplexFloat,"FFT complex input must be complex64");std::vector<int64_t> raw;for(auto d:dim)raw.push_back(d.expect_int());auto ds=axes(x,raw);int64_t count=1;auto y=x;
 for(auto d:ds){count*=x.size(d);y=transform(y,d,!forward);}float scale=normalization(count,norm);if(scale!=1)y=finish(y,ds.back(),y.size(ds.back()),false,scale);return y;
}
at::Tensor c2r(const at::Tensor& x,at::IntArrayRef dim,int64_t norm,c10::SymInt last){
 TORCH_CHECK(x.scalar_type()==at::kComplexFloat,"FFT complex input must be complex64");auto ds=axes(x,dim);int64_t n=last.expect_int(),count=n;TORCH_CHECK(n>0&&x.size(ds.back())==n/2+1,"FFT half spectrum length mismatch");auto y=x;
 for(size_t i=0;i+1<ds.size();i++){count*=x.size(ds[i]);y=transform(y,ds[i],true);}y=transform(y,ds.back(),true,n);return finish(y,ds.back(),n,true,normalization(count,norm));
}
}
at::Tensor complex_binary(const at::Tensor& x,const at::Tensor& y,uint32_t mode,const at::Scalar& alpha){
 using namespace native;TORCH_CHECK((x.scalar_type()==at::kFloat||x.scalar_type()==at::kComplexFloat)&&(y.scalar_type()==at::kFloat||y.scalar_type()==at::kComplexFloat),"complex operations support FP32/complex64");
 auto device=x.is_cpu()?y.device():x.device();TORCH_CHECK(device.type()==c10::DeviceType::PrivateUse1&&(x.device()==device||(x.is_cpu()&&x.dim()==0))&&(y.device()==device||(y.is_cpu()&&y.dim()==0)),"complex operation device mismatch");
 auto shape=at::infer_size_dimvector(x.sizes(),y.sizes());auto out=at::empty(shape,x.options().device(device).dtype(at::kComplexFloat));if(!out.numel())return out;
 auto a=x.is_cpu()?x.to(device):x,b=y.is_cpu()?y.to(device):y;a=packed(a.expand(shape),a.scalar_type());b=packed(b.expand(shape),b.scalar_type());auto scalar=alpha.toComplexFloat();
 run("complex_binary",{base(a,a.scalar_type()),base(b,b.scalar_type())},{write_base(out,at::kComplexFloat)},{u32(out.numel()),uint32_t(a.is_complex()),uint32_t(b.is_complex()),mode,bits(scalar.real()),bits(scalar.imag())},(out.numel()+127)/128);return out;
}
at::Tensor complex_math_unary(const at::Tensor& x,uint32_t mode){return complex_unary(x,0,mode);}
at::Tensor complex_power(const at::Tensor& x,const at::Scalar& exponent){
 auto e=exponent.toComplexFloat();
 if(e.imag()==0){if(e.real()==0)return at::ones_like(x);if(e.real()==1)return x.clone();if(e.real()==2)return complex_binary(x,x,1);if(e.real()==3)return complex_binary(complex_binary(x,x,1),x,1);}
 return complex_unary(x,exponent,6);
}
at::Tensor complex_scale(const at::Tensor& x,const at::Scalar& scale){return complex_unary(x,scale,0);}
at::Tensor& complex_fill(at::Tensor& x,const at::Scalar& value){auto z=value.toComplexFloat();auto view=at::view_as_real(x);view.select(-1,0).fill_(z.real());view.select(-1,1).fill_(z.imag());return x;}
TORCH_LIBRARY_IMPL(aten,PrivateUse1,m){
 m.impl("complex",&complex_construct);m.impl("polar",&polar_construct);
 m.impl("_fft_r2c",&r2c);m.impl("_fft_c2c",&c2c);m.impl("_fft_c2r",&c2r);
 m.impl("view_as_real",&at::native::view_as_real);m.impl("view_as_complex",&at::native::view_as_complex);m.impl("_conj_physical",&complex_conj);
}
}
