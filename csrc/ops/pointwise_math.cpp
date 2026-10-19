#include "native_utils.h"
#include "device_copy.h"
#include <ATen/ExpandUtils.h>
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
template<int Op>at::Tensor binary_math(const at::Tensor& left,const at::Tensor& right){
 auto device=left.is_cpu()?right.device():left.device();TORCH_CHECK(device.type()==c10::DeviceType::PrivateUse1&&(left.device()==device||(left.is_cpu()&&left.dim()==0))&&(right.device()==device||(right.is_cpu()&&right.dim()==0)),"binary math tensors must be on the same device");
 auto dtype=at::result_type(left,right);if constexpr(Op<=3)TORCH_CHECK(dtype==at::kFloat,"integer remainder/power/rounded division requires an exact native integer implementation");
 TORCH_CHECK(device_copy_supported(dtype),"binary math requires real input");auto shape=at::infer_size_dimvector(left.sizes(),right.sizes());auto out=at::empty(shape,left.options().device(device).dtype(at::kFloat));if(!out.numel())return out;
 auto a=packed(left.to(device,at::kFloat).expand(shape)),b=packed(right.to(device,at::kFloat).expand(shape));run("pointwise_binary",{base(a),base(b)},{write_base(out)},{u32(out.numel()),Op},(out.numel()+127)/128);return out;
}
template<int Op>at::Tensor binary_math_scalar(const at::Tensor& x,const at::Scalar& value){return binary_math<Op>(x,at::scalar_tensor(value,at::TensorOptions().dtype(at::result_type(x,value))));}
template<int Op>at::Tensor& binary_math_inplace(at::Tensor& x,const at::Tensor& y){at::assert_no_internal_overlap(x);auto value=binary_math<Op>(x,y);TORCH_CHECK(x.sizes()==value.sizes()&&c10::canCast(value.scalar_type(),x.scalar_type()),"inplace binary result shape/dtype mismatch");x.copy_(value);return x;}
template<int Op>at::Tensor& binary_math_scalar_inplace(at::Tensor& x,const at::Scalar& value){return binary_math_inplace<Op>(x,at::scalar_tensor(value,at::TensorOptions().dtype(at::result_type(x,value))));}
at::Tensor div_mode(const at::Tensor& a,const at::Tensor& b,std::optional<c10::string_view> mode){if(!mode)return at::div(a,b);TORCH_CHECK(at::result_type(a,b)==at::kFloat,"integer rounded division requires an exact native integer implementation");if(*mode=="trunc")return at::trunc(at::div(a,b));TORCH_CHECK(*mode=="floor","div expected rounding_mode to be one of None, 'trunc', or 'floor'");return binary_math<3>(a,b);}
at::Tensor div_scalar_mode(const at::Tensor& a,const at::Scalar& b,std::optional<c10::string_view> mode){return div_mode(a,at::scalar_tensor(b,at::TensorOptions().dtype(at::result_type(a,b))),mode);}
at::Tensor& div_mode_inplace(at::Tensor& a,const at::Tensor& b,std::optional<c10::string_view> mode){auto value=div_mode(a,b,mode);TORCH_CHECK(a.sizes()==value.sizes()&&c10::canCast(value.scalar_type(),a.scalar_type()),"inplace div shape/dtype mismatch");a.copy_(value);return a;}
at::Tensor& div_scalar_mode_inplace(at::Tensor& a,const at::Scalar& b,std::optional<c10::string_view> mode){return div_mode_inplace(a,at::scalar_tensor(b,at::TensorOptions().dtype(at::result_type(a,b))),mode);}

}
TORCH_LIBRARY_IMPL(aten,PrivateUse1,m){
 m.impl("fmod.Tensor",&binary_math<0>);m.impl("fmod.Scalar",&binary_math_scalar<0>);m.impl("fmod_.Tensor",&binary_math_inplace<0>);m.impl("fmod_.Scalar",&binary_math_scalar_inplace<0>);
 m.impl("remainder.Tensor",&binary_math<1>);m.impl("remainder.Scalar",&binary_math_scalar<1>);m.impl("remainder_.Tensor",&binary_math_inplace<1>);m.impl("remainder_.Scalar",&binary_math_scalar_inplace<1>);
 m.impl("pow.Tensor_Tensor",&binary_math<2>);m.impl("pow_.Tensor",&binary_math_inplace<2>);
 m.impl("floor_divide",&binary_math<3>);m.impl("floor_divide.Scalar",&binary_math_scalar<3>);m.impl("floor_divide_.Tensor",&binary_math_inplace<3>);m.impl("floor_divide_.Scalar",&binary_math_scalar_inplace<3>);
 m.impl("div.Tensor_mode",&div_mode);m.impl("div.Scalar_mode",&div_scalar_mode);m.impl("div_.Tensor_mode",&div_mode_inplace);m.impl("div_.Scalar_mode",&div_scalar_mode_inplace);
 m.impl("hypot",&binary_math<4>);m.impl("hypot_",&binary_math_inplace<4>);m.impl("copysign.Tensor",&binary_math<5>);m.impl("copysign.Scalar",&binary_math_scalar<5>);m.impl("copysign_.Tensor",&binary_math_inplace<5>);m.impl("copysign_.Scalar",&binary_math_scalar_inplace<5>);
 m.impl("logaddexp",&binary_math<6>);m.impl("logaddexp2",&binary_math<7>);m.impl("xlogy.Tensor",&binary_math<8>);m.impl("xlogy.Scalar_Other",&binary_math_scalar<8>);m.impl("xlogy_.Tensor",&binary_math_inplace<8>);m.impl("xlogy_.Scalar_Other",&binary_math_scalar_inplace<8>);m.impl("nextafter",&binary_math<9>);m.impl("nextafter_",&binary_math_inplace<9>);
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
