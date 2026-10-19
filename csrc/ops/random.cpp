#include "native_utils.h"
#include "device_copy.h"
#include "../vulkan_generator.h"
#include <ATen/MemoryOverlap.h>
#include <ATen/ExpandUtils.h>
#include <torch/library.h>
#include <cmath>
namespace torch_vulkan::ops {
namespace {
using namespace native;
at::Tensor& random_fill(at::Tensor& out,uint32_t mode,double a,double b,const std::optional<at::Generator>& generator,const at::Tensor& probs={}) {
 if(out.scalar_type()==at::kComplexFloat&&(mode==0||mode==1)){auto real=at::view_as_real(out);random_fill(real,mode,a,mode==1?b/std::sqrt(2.):b,generator);return out;}
 if(out.scalar_type()!=at::kFloat&&device_copy_supported(out.scalar_type())&&(mode==2u||mode==3u||mode==7u)){at::assert_no_internal_overlap(out);auto temporary=at::empty(out.sizes(),out.options().dtype(at::kFloat));random_fill(temporary,mode,a,b,generator,probs);out.copy_(temporary);return out;}
 TORCH_CHECK(out.scalar_type()==at::kFloat,"Vulkan random fill requires FP32 output; got ",out.scalar_type());
 if(!out.numel())return out;
 if(out.dim()>8&&out.is_contiguous()){auto flat=out.view({-1});random_fill(flat,mode,a,b,generator,probs);return out;}
 TORCH_CHECK(can_pack_view(out),"Vulkan random fill requires packable FP32 output; got ",out.scalar_type()," with shape ",out.sizes());at::assert_no_internal_overlap(out);
 if(!out.numel())return out;
 auto state=random_state(generator,(out.numel()+3)/4);
 std::vector<uint32_t> p={u32(out.numel()),mode,state[0],state[1],state[2],state[3],bits(float(a)),bits(float(b)),u32(out.storage_offset())};p.resize(25,0);std::fill(p.begin()+9,p.begin()+17,1);
 for(int d=0;d<out.dim();d++){p[9+d]=u32(out.size(d));p[17+d]=u32(out.stride(d));}
 auto probability=probs.defined()?base(probs):VulkanAllocator::instance().constant_float_buffer(1,0);
 bool partial=out.storage_offset()!=0||!out.is_contiguous()||out.storage().nbytes()!=uint64_t(out.numel())*out.element_size();
 auto previous=partial?base(out):VulkanAllocator::instance().constant_float_buffer(1,0);
 run("random_fill",{probability,previous},{write_base(out)},p,(out.numel()+127)/128);return out;
}
at::Tensor& integer_random(at::Tensor& out,int64_t from,std::optional<int64_t> to,std::optional<at::Generator> generator){
 if(out.scalar_type()==at::kUInt64){TORCH_CHECK(to.has_value(),"default random_ is not implemented for UInt64");TORCH_CHECK(from>=0,"from is out of bounds for UInt64");auto storage=out.view(at::kLong);integer_random(storage,from,to,generator);return out;}
 if(out.scalar_type()==at::kBool||out.scalar_type()==at::kFloat||out.scalar_type()==at::kByte||out.scalar_type()==at::kChar||out.scalar_type()==at::kShort){
  at::assert_no_internal_overlap(out);
  if(out.scalar_type()==at::kBool){TORCH_CHECK(from>=0&&from<=1,"from is out of bounds for bool");TORCH_CHECK(!to||(*to>=1&&*to<=2),"to - 1 is out of bounds for bool");}
  int64_t low=0,limit=out.scalar_type()==at::kBool?2:16777217;
  if(out.scalar_type()==at::kByte)limit=256;
  if(out.scalar_type()==at::kChar){low=-128;limit=128;}
  if(out.scalar_type()==at::kShort){low=-32768;limit=32768;}
  if(out.scalar_type()!=at::kFloat){TORCH_CHECK(from>=low&&from<limit,"random_ from is out of bounds");TORCH_CHECK(!to||(*to>low&&*to<=limit),"random_ to - 1 is out of bounds");}
  auto upper=to;
  if(!upper && !(out.scalar_type()==at::kFloat&&from==INT64_MIN))upper=limit;
  auto integers=at::empty(out.sizes(),out.options().dtype(at::kLong));
  integer_random(integers,from,upper,generator);copy_on_device(out,integers);return out;
 }
 TORCH_CHECK((out.scalar_type()==at::kInt||out.scalar_type()==at::kLong)&&can_pack_view(out),"Vulkan integer RNG supports int32/int64 output");at::assert_no_internal_overlap(out);
 __int128 upper=to?__int128(*to):(out.scalar_type()==at::kInt?__int128(INT32_MAX)+1:__int128(INT64_MAX)+1);
 TORCH_CHECK(upper>from,"random_ expects from < to");if(out.scalar_type()==at::kInt)TORCH_CHECK(from>=INT32_MIN&&upper<=__int128(INT32_MAX)+1,"random range exceeds int32");
 if(!out.numel())return out;__uint128_t range=upper-__int128(from),mask=1;while(mask<range)mask<<=1;mask-=1;
 uint64_t span=uint64_t(range),bitsmask=uint64_t(mask),first=uint64_t(from);auto state=random_state(generator,out.numel());
 std::vector<uint32_t> p={u32(out.numel()),uint32_t(out.element_size()/4),u32(out.storage_offset()),state[0],state[1],state[2],state[3],uint32_t(first),uint32_t(first>>32),uint32_t(span),uint32_t(span>>32),uint32_t(bitsmask),uint32_t(bitsmask>>32)};p.resize(29,0);std::fill(p.begin()+13,p.begin()+21,1);for(int d=0;d<out.dim();d++){p[13+d]=u32(out.size(d));p[21+d]=u32(out.stride(d));}
 bool partial=out.storage_offset()!=0||!out.is_contiguous()||out.storage().nbytes()!=uint64_t(out.numel())*out.element_size();auto previous=partial?base(out,out.scalar_type()):VulkanAllocator::instance().constant_float_buffer(1,0);
 run("random_integer",{previous},{write_base(out,out.scalar_type())},p,(out.numel()+127)/128);return out;
}
at::Tensor& integer_random_to(at::Tensor& out,int64_t to,std::optional<at::Generator> generator){return integer_random(out,0,to,generator);}
at::Tensor& integer_random_default(at::Tensor& out,std::optional<at::Generator> generator){return integer_random(out,0,std::nullopt,generator);}
at::Tensor& uniform(at::Tensor& out,double from,double to,std::optional<at::Generator> g){TORCH_CHECK(std::isfinite(from)&&std::isfinite(to)&&from<=to&&std::abs(from)<=std::numeric_limits<float>::max()&&std::abs(to)<=std::numeric_limits<float>::max(),"uniform: invalid range");return random_fill(out,0,from,to,g);}
at::Tensor& normal(at::Tensor& out,double mean,double std,std::optional<at::Generator> g){TORCH_CHECK(std>=0,"normal expects std >= 0.0, but found std ",std);return random_fill(out,1,mean,std,g);}
at::Tensor& bernoulli_scalar(at::Tensor& out,double p,std::optional<at::Generator> g){TORCH_CHECK(p>=0&&p<=1,"bernoulli: probability must be in [0,1]");return random_fill(out,2,p,0,g);}
at::Tensor& bernoulli_tensor(at::Tensor& out,const at::Tensor& probs,std::optional<at::Generator> g){
 TORCH_CHECK(probs.device()==out.device()&&probs.scalar_type()==at::kFloat,"bernoulli: expected FP32 probabilities on same device");
 auto expanded=probs.expand(out.sizes());if(!out.numel())return out;auto p=packed(expanded);TORCH_CHECK(at::all(at::logical_and(p.ge(0),p.le(1))).cpu().item<bool>(),"bernoulli: probability must be in [0,1]");return random_fill(out,3,0,0,g,p);
}
at::Tensor& exponential(at::Tensor& out,double lambda,std::optional<at::Generator> generator){TORCH_CHECK(lambda>0,"exponential_ expects lambda > 0.0, but found lambda=",lambda);return random_fill(out,4,lambda,0,generator);}
at::Tensor& cauchy(at::Tensor& out,double median,double sigma,std::optional<at::Generator> generator){TORCH_CHECK(sigma>0,"cauchy_ expects sigma > 0.0, but found sigma=",sigma);return random_fill(out,5,median,sigma,generator);}
at::Tensor& log_normal(at::Tensor& out,double mean,double std,std::optional<at::Generator> generator){TORCH_CHECK(std>0,"log_normal_ expects std > 0.0, but found std=",std);return random_fill(out,6,mean,std,generator);}
at::Tensor& geometric(at::Tensor& out,double probability,std::optional<at::Generator> generator){TORCH_CHECK(probability>0&&probability<1,"geometric_ expects p to be in (0, 1), but got p=",probability);return random_fill(out,7,probability,0,generator);}
at::Tensor normal_mean(const at::Tensor& mean,double std,std::optional<at::Generator> generator){auto result=at::empty(mean.sizes(),mean.options());normal(result,0,std,generator);return result.add(mean);}
at::Tensor normal_std(double mean,const at::Tensor& std,std::optional<at::Generator> generator){TORCH_CHECK(std.scalar_type()==at::kFloat,"normal expects floating point std");TORCH_CHECK(at::all(std.ge(0)).cpu().item<bool>(),"normal expects all elements of std >= 0.0");auto result=at::empty_like(std);normal(result,0,1,generator);return result.mul(std).add(mean);}
at::Tensor normal_parameters(const at::Tensor& mean,const at::Tensor& std,std::optional<at::Generator> generator){TORCH_CHECK(std.scalar_type()==at::kFloat&&std.device()==mean.device(),"normal expects floating point std on the mean device");TORCH_CHECK(at::all(std.ge(0)).cpu().item<bool>(),"normal expects all elements of std >= 0.0");auto shape=at::infer_size_dimvector(mean.sizes(),std.sizes());auto result=at::empty(shape,mean.options());normal(result,0,1,generator);return result.mul(std).add(mean);}
at::Tensor bernoulli(const at::Tensor& p,std::optional<at::Generator> g){auto out=at::empty(p.sizes(),p.options());return bernoulli_tensor(out,p,g);}
std::tuple<at::Tensor,at::Tensor> dropout(const at::Tensor& x,double p,std::optional<bool> train){
 TORCH_CHECK(p>=0&&p<=1,"dropout probability must be in [0,1]");TORCH_CHECK(x.scalar_type()==at::kFloat&&can_pack_view(x),"Vulkan dropout requires packable FP32 input");
 auto y=at::empty(x.sizes(),x.options()),mask=at::empty(x.sizes(),x.options().dtype(at::kBool));if(!x.numel())return {y,mask};
 double keep=train.value_or(true)?1-p:1;auto state=keep>0&&keep<1?random_state(std::nullopt,(x.numel()+3)/4):std::array<uint32_t,4>{};
 auto a=packed(x);run("dropout",{base(a)},{write_base(y),write_base(mask,at::kBool)},
  {u32(x.numel()),bits(float(keep)),state[0],state[1],state[2],state[3]},(x.numel()+511)/512);return {y,mask};
}
at::Tensor dropout_backward(const at::Tensor& grad,const at::Tensor& mask,double scale){
 if(grad.scalar_type()!=at::kFloat&&device_copy_supported(grad.scalar_type()))return dropout_backward(grad.to(at::kFloat),mask,scale);
 TORCH_CHECK(grad.sizes()==mask.sizes()&&grad.device()==mask.device()&&grad.scalar_type()==at::kFloat&&mask.scalar_type()==at::kBool&&can_pack_view(grad)&&can_pack_view(mask),"dropout backward: invalid tensors");
 auto out=at::empty(grad.sizes(),grad.options());if(out.numel())run("dropout_backward",{base(packed(grad)),base(packed(mask,at::kBool),at::kBool)},{write_base(out)},{u32(out.numel()),bits(float(scale))},(out.numel()+127)/128);return out;
}
}
TORCH_LIBRARY_IMPL(aten,PrivateUse1,m){m.impl("exponential_",&exponential);m.impl("cauchy_",&cauchy);m.impl("log_normal_",&log_normal);m.impl("geometric_",&geometric);m.impl("normal.Tensor_float",&normal_mean);m.impl("normal.float_Tensor",&normal_std);m.impl("normal.Tensor_Tensor",&normal_parameters);m.impl("random_.from",&integer_random);m.impl("random_.to",&integer_random_to);m.impl("random_",&integer_random_default);m.impl("uniform_",&uniform);m.impl("normal_",&normal);m.impl("bernoulli_.float",&bernoulli_scalar);m.impl("bernoulli_.Tensor",&bernoulli_tensor);m.impl("bernoulli",&bernoulli);m.impl("native_dropout",&dropout);m.impl("native_dropout_backward",&dropout_backward);}
}
