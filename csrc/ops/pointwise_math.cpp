#include "native_utils.h"
#include "device_copy.h"
#include <ATen/MemoryOverlap.h>
#include <torch/library.h>
#include <cmath>
namespace torch_vulkan::ops {
namespace {
using namespace native;
template<int Op>at::Tensor math_unary(const at::Tensor& raw){
 TORCH_CHECK(device_copy_supported(raw.scalar_type()),"Vulkan pointwise math supports FP32/int32/int64/bool inputs");
 if constexpr(Op>=4&&Op<=7){TORCH_CHECK(raw.scalar_type()!=at::kBool,"rounding operations are not implemented for bool");if(!raw.is_floating_point())return raw.clone();}
 if constexpr(Op==8){if(raw.scalar_type()==at::kBool)return raw.clone();if(!raw.is_floating_point()){auto negative=raw.lt(0),positive=raw.gt(0);return positive.to(raw.scalar_type()).add(negative.to(raw.scalar_type()),-1);}}
 auto x=raw.to(at::kFloat);TORCH_CHECK(can_pack_view(x),"pointwise math exceeds Vulkan indexing limits");auto out=at::empty(x.sizes(),x.options());if(!x.numel())return out;
 run("pointwise_math",{base(packed(x))},{write_base(out)},{u32(x.numel()),Op,bits(1),0},(x.numel()+127)/128);return out;
}
template<int Op>at::Tensor& math_unary_inplace(at::Tensor& x){at::assert_no_internal_overlap(x);auto value=math_unary<Op>(x);TORCH_CHECK(c10::canCast(value.scalar_type(),x.scalar_type()),"result type cannot be cast to the desired output type");x.copy_(value);return x;}
at::Tensor round_decimals(const at::Tensor& x,int64_t decimals){if(!decimals)return math_unary<7>(x);TORCH_CHECK(x.scalar_type()==at::kFloat,"round decimals requires FP32");auto out=at::empty(x.sizes(),x.options());if(out.numel()){float factor=std::pow(10.,std::abs(double(decimals)));run("pointwise_math",{base(packed(x))},{write_base(out)},{u32(x.numel()),25,bits(factor),uint32_t(decimals<0)},(x.numel()+127)/128);}return out;}
at::Tensor& round_decimals_inplace(at::Tensor& x,int64_t decimals){x.copy_(round_decimals(x,decimals));return x;}
}
TORCH_LIBRARY_IMPL(aten,PrivateUse1,m){
#define MATH(name,op) m.impl(name,&math_unary<op>);m.impl(name "_",&math_unary_inplace<op>);
 MATH("reciprocal",0) MATH("log",1) MATH("log1p",2) MATH("expm1",3)
 MATH("floor",4) MATH("ceil",5) MATH("trunc",6) MATH("round",7) MATH("sign",8)
 MATH("log10",9) MATH("log2",10) MATH("exp2",11) MATH("sinh",12) MATH("cosh",13)
 MATH("tan",14) MATH("asin",15) MATH("acos",16) MATH("atan",17)
 MATH("asinh",18) MATH("acosh",19) MATH("atanh",20) MATH("erf",21) MATH("erfc",22)
#undef MATH
 m.impl("round.decimals",&round_decimals);m.impl("round_.decimals",&round_decimals_inplace);
}
}
