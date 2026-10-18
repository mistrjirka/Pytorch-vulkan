#include "native_utils.h"
#include "device_copy.h"
#include <ATen/ExpandUtils.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/native/Resize.h>
#include <torch/library.h>
#include <cmath>
namespace torch_vulkan::ops {
namespace {
using namespace native;
uint32_t type_code(at::ScalarType t){return t==at::kBool?0:t==at::kFloat?1:t==at::kInt?2:3;}
template<int Op>at::Tensor comparison(const at::Tensor& left,const at::Tensor& right){
 auto dev=left.is_cpu()?right.device():left.device();
 TORCH_CHECK(dev.type()==c10::DeviceType::PrivateUse1&&(left.device()==dev||(left.is_cpu()&&left.dim()==0))&&(right.device()==dev||(right.is_cpu()&&right.dim()==0)),"comparison: expected tensors on the same device");
 auto dtype=at::result_type(left,right);
 TORCH_CHECK(device_copy_supported(dtype),"Vulkan comparison supports FP32/int32/int64/bool");
 auto shape=at::infer_size_dimvector(left.sizes(),right.sizes());auto out=at::empty(shape,left.options().device(dev).dtype(at::kBool));if(!out.numel())return out;
 auto a=left.to(dev,dtype).expand(shape),b=right.to(dev,dtype).expand(shape);
 TORCH_CHECK(can_pack_view(a)&&can_pack_view(b),"comparison exceeds Vulkan indexing/storage limits");
 a=packed(a,dtype);b=packed(b,dtype);
 run("comparison",{base(a,dtype),base(b,dtype)},{write_base(out,at::kBool)},{u32(out.numel()),type_code(dtype),Op},(out.numel()+511)/512);return out;
}
template<int Op>at::Tensor comparison_scalar(const at::Tensor& x,const at::Scalar& s){
 auto scalar=at::scalar_tensor(s,at::TensorOptions().dtype(at::result_type(x,s)));return comparison<Op>(x,scalar);
}
template<int Op>at::Tensor& comparison_out(const at::Tensor& a,const at::Tensor& b,at::Tensor& out){
 TORCH_CHECK(out.device()==(a.is_cpu()?b.device():a.device()),"comparison output device mismatch");
 at::assert_no_internal_overlap(out);at::assert_no_partial_overlap(out,a);at::assert_no_partial_overlap(out,b);
 auto value=comparison<Op>(a,b);at::native::resize_output(out,value.sizes());out.copy_(value);return out;
}
template<int Op>at::Tensor& comparison_scalar_out(const at::Tensor& a,const at::Scalar& b,at::Tensor& out){
 auto scalar=at::scalar_tensor(b,at::TensorOptions().dtype(at::result_type(a,b)));return comparison_out<Op>(a,scalar,out);
}
template<int Op>at::Tensor& comparison_inplace(at::Tensor& a,const at::Tensor& b){auto value=comparison<Op>(a,b);TORCH_CHECK(a.sizes()==value.sizes(),"comparison in-place broadcast shape mismatch");a.copy_(value);return a;}
template<int Op>at::Tensor& comparison_scalar_inplace(at::Tensor& a,const at::Scalar& b){a.copy_(comparison_scalar<Op>(a,b));return a;}
template<int Op>at::Tensor unary_predicate(const at::Tensor& x){
 if constexpr(Op==0)return comparison<1>(x,x);
 if constexpr(Op==1)return comparison_scalar<0>(x,0);
 if constexpr(Op==2)return comparison_scalar<0>(x,INFINITY);
 if constexpr(Op==3)return comparison_scalar<0>(x,-INFINITY);
 if constexpr(Op==4)return at::logical_or(comparison_scalar<0>(x,INFINITY),comparison_scalar<0>(x,-INFINITY));
}
template<int Op>at::Tensor& unary_predicate_out(const at::Tensor& x,at::Tensor& out){
 TORCH_CHECK(out.device()==x.device(),"predicate output device mismatch");
 at::assert_no_internal_overlap(out);at::assert_no_partial_overlap(out,x);
 auto value=unary_predicate<Op>(x);at::native::resize_output(out,value.sizes());out.copy_(value);return out;
}
at::Tensor& logical_not_inplace(at::Tensor& x){return unary_predicate_out<1>(x,x);}
bool equal_native(const at::Tensor& a,const at::Tensor& b){if(a.sizes()!=b.sizes())return false;return comparison<0>(a,b).all().item<bool>();}

}
TORCH_LIBRARY_IMPL(aten,PrivateUse1,m){
#define COMPARE(name, op) m.impl(name ".Tensor", &comparison<op>);m.impl(name ".Scalar", &comparison_scalar<op>);m.impl(name ".Tensor_out", &comparison_out<op>);m.impl(name ".Scalar_out", &comparison_scalar_out<op>);m.impl(name "_.Tensor", &comparison_inplace<op>);m.impl(name "_.Scalar", &comparison_scalar_inplace<op>);
 COMPARE("eq",0) COMPARE("ne",1) COMPARE("lt",2) COMPARE("le",3) COMPARE("gt",4) COMPARE("ge",5)
#undef COMPARE
 m.impl("equal",&equal_native);
#define LOGIC(name, op) m.impl(name, &comparison<op>);m.impl(name ".out", &comparison_out<op>);m.impl(name "_", &comparison_inplace<op>);
 LOGIC("logical_and",6) LOGIC("logical_or",7) LOGIC("logical_xor",8)
#undef LOGIC
#define PRED(name,op) m.impl(name,&unary_predicate<op>);m.impl(name ".out",&unary_predicate_out<op>);
 PRED("isnan",0) PRED("logical_not",1) PRED("isposinf",2) PRED("isneginf",3) PRED("isinf",4)
#undef PRED
 m.impl("logical_not_",&logical_not_inplace);
}
}
