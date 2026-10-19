#include "native_utils.h"
#include <ATen/ExpandUtils.h>
#include <torch/library.h>
namespace torch_vulkan::ops {
namespace {
using namespace native;
template<int Op>at::Tensor bitwise(const at::Tensor& left,const at::Tensor& right){
 auto dtype=at::result_type(left,right);if constexpr(Op>=3)TORCH_CHECK(dtype!=at::kBool,"bitwise shifts are not implemented for bool");auto dev=left.is_cpu()?right.device():left.device();
 if(dtype==at::kByte||dtype==at::kChar||dtype==at::kShort)return bitwise<Op>(left.to(at::kInt),right.to(at::kInt)).to(dtype);
 TORCH_CHECK(dtype==at::kBool||dtype==at::kInt||dtype==at::kLong,"Vulkan bitwise operations require bool/int32/int64");
 TORCH_CHECK((left.device()==dev||(left.is_cpu()&&left.dim()==0))&&(right.device()==dev||(right.is_cpu()&&right.dim()==0)),"bitwise input device mismatch");
 if(dtype==at::kBool){if constexpr(Op==0)return at::logical_and(left,right);if constexpr(Op==1)return at::logical_or(left,right);if constexpr(Op==2)return at::logical_xor(left,right);}
 auto shape=at::infer_size_dimvector(left.sizes(),right.sizes());auto out=at::empty(shape,left.options().device(dev).dtype(dtype));if(!out.numel())return out;
 auto a=packed(left.to(dev,dtype).expand(shape),dtype),b=packed(right.to(dev,dtype).expand(shape),dtype);
 run("integer_bitwise",{base(a,dtype),base(b,dtype)},{write_base(out,dtype)},{u32(out.numel()),uint32_t(out.element_size()/4),Op},(out.numel()+127)/128);return out;
}
template<int Op>at::Tensor bitwise_scalar(const at::Tensor& x,const at::Scalar& s){return bitwise<Op>(x,at::scalar_tensor(s,at::TensorOptions().dtype(at::result_type(x,s))));}
template<int Op>at::Tensor bitwise_scalar_left(const at::Scalar& s,const at::Tensor& x){return bitwise<Op>(at::scalar_tensor(s,at::TensorOptions().dtype(at::result_type(x,s))),x);}
template<int Op>at::Tensor& bitwise_inplace(at::Tensor& x,const at::Tensor& y){auto result=bitwise<Op>(x,y);TORCH_CHECK(x.sizes()==result.sizes(),"bitwise in-place broadcast mismatch");TORCH_CHECK(c10::canCast(result.scalar_type(),x.scalar_type()),"bitwise in-place dtype mismatch");x.copy_(result);return x;}
template<int Op>at::Tensor& bitwise_scalar_inplace(at::Tensor& x,const at::Scalar& s){return bitwise_inplace<Op>(x,at::scalar_tensor(s,at::TensorOptions().dtype(at::result_type(x,s))));}
at::Tensor bitwise_not(const at::Tensor& x){if(x.scalar_type()==at::kBool)return at::logical_not(x);return bitwise<2>(x,at::scalar_tensor(-1,x.options().device(at::kCPU)));}
at::Tensor& bitwise_not_inplace(at::Tensor& x){x.copy_(at::bitwise_not(x));return x;}
}
TORCH_LIBRARY_IMPL(aten,PrivateUse1,m){
#define BIT(name,op) m.impl(name ".Tensor",&bitwise<op>);m.impl(name ".Scalar",&bitwise_scalar<op>);m.impl(name ".Scalar_Tensor",&bitwise_scalar_left<op>);m.impl(name "_.Tensor",&bitwise_inplace<op>);m.impl(name "_.Scalar",&bitwise_scalar_inplace<op>);
 BIT("bitwise_and",0) BIT("bitwise_or",1) BIT("bitwise_xor",2)
#undef BIT
 m.impl("bitwise_left_shift.Tensor",&bitwise<3>);m.impl("bitwise_left_shift.Tensor_Scalar",&bitwise_scalar<3>);m.impl("bitwise_left_shift.Scalar_Tensor",&bitwise_scalar_left<3>);m.impl("bitwise_left_shift_.Tensor",&bitwise_inplace<3>);m.impl("bitwise_left_shift_.Tensor_Scalar",&bitwise_scalar_inplace<3>);m.impl("bitwise_right_shift.Tensor",&bitwise<4>);m.impl("bitwise_right_shift.Tensor_Scalar",&bitwise_scalar<4>);m.impl("bitwise_right_shift.Scalar_Tensor",&bitwise_scalar_left<4>);m.impl("bitwise_right_shift_.Tensor",&bitwise_inplace<4>);m.impl("bitwise_right_shift_.Tensor_Scalar",&bitwise_scalar_inplace<4>);m.impl("bitwise_not",&bitwise_not);m.impl("bitwise_not_",&bitwise_not_inplace);
}
}
