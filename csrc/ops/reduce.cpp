#include "native_utils.h"
#include "reduce_native.h"
#include <torch/library.h>
#include <numeric>
namespace torch_vulkan::ops {
at::Tensor reduce_fp32(const at::Tensor& self,at::OptionalIntArrayRef dim,bool keepdim,bool mean){
 using namespace native;
 TORCH_CHECK(self.scalar_type()==at::kFloat && can_pack_view(self),"Vulkan reduction requires packable FP32 input");
 const int64_t rank=std::max<int64_t>(1,self.dim());uint32_t mask=0;
 if(!dim || dim->empty())mask=(1u<<rank)-1;
 else for(auto raw:*dim){int64_t d=raw<0?raw+rank:raw;TORCH_CHECK(d>=0&&d<rank,"reduction dimension out of range");TORCH_CHECK(!(mask&(1u<<d)),"dim appears multiple times in the list of dims");mask|=1u<<d;}
 std::vector<int64_t> shape;uint64_t reduced=1;
 for(int64_t d=0;d<self.dim();d++){if(mask&(1u<<d)){reduced*=self.size(d);if(keepdim)shape.push_back(1);}else shape.push_back(self.size(d));}
 auto out=at::empty(shape,self.options());if(!out.numel())return out;
 if(!reduced){out.fill_(mean?std::numeric_limits<float>::quiet_NaN():0.f);return out;}
 auto x=packed(self);bool parallel=reduced>=64;
 std::vector<uint32_t> p(21,0);p[0]=u32(out.numel());p[1]=u32(reduced);p[2]=mask;p[3]=mean;p[4]=parallel;
 std::fill(p.begin()+5,p.begin()+13,1);
 for(int64_t d=0;d<self.dim();d++){p[5+d]=u32(self.size(d));p[13+d]=u32(x.stride(d));}
 run("reduce_axes",{base(x)},{base(out)},p,parallel?out.numel():(out.numel()+127)/128,512);return out;
}
namespace {
at::Tensor sum_dim(const at::Tensor& x,at::OptionalIntArrayRef dim,bool keepdim,std::optional<at::ScalarType> dtype){
 const auto* flag=std::getenv("TORCH_VULKAN_NATIVE_SUM");
 if((flag&&!std::strcmp(flag,"0"))||x.scalar_type()!=at::kFloat||dtype.value_or(x.scalar_type())!=at::kFloat||!native::can_pack_view(x))return at::sum(x.cpu(),dim,keepdim,dtype).to(x.device());
 return reduce_fp32(x,dim,keepdim,false);
}
at::Tensor sum_all(const at::Tensor& x,std::optional<at::ScalarType> dtype){return sum_dim(x,std::nullopt,false,dtype);}
}
TORCH_LIBRARY_IMPL(aten,PrivateUse1,m){m.impl("sum.dim_IntList",&sum_dim);m.impl("sum",&sum_all);}
}
