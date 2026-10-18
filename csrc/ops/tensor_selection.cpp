// Shape-independent compositions of native device selection primitives.
#include "native_utils.h"
#include "device_copy.h"
#include <ATen/ExpandUtils.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/WrapDimUtils.h>
#include <torch/library.h>
namespace torch_vulkan::ops {
namespace {
at::Tensor masked_fill_tensor(const at::Tensor& x,const at::Tensor& mask,const at::Tensor& value){
 TORCH_CHECK(value.dim()==0,"masked_fill_ only supports a 0-dimensional value tensor");
 TORCH_CHECK(mask.scalar_type()==at::kBool,"masked_fill_ only supports boolean masks");
 TORCH_CHECK(x.device()==mask.device(),"masked_fill device mismatch");
 TORCH_CHECK(value.device()==x.device()||value.is_cpu(),"masked_fill value device mismatch");
 return at::where(mask,value.to(x.device(),x.scalar_type()),x);
}
at::Tensor masked_fill_scalar(const at::Tensor& x,const at::Tensor& mask,const at::Scalar& value){return masked_fill_tensor(x,mask,at::scalar_tensor(value,at::TensorOptions().dtype(x.scalar_type())));}
at::Tensor& assign(at::Tensor& x,const at::Tensor& result){
 TORCH_CHECK(x.sizes()==result.sizes(),"output with shape ",x.sizes()," doesn't match the broadcast shape ",result.sizes());
 TORCH_CHECK(c10::canCast(result.scalar_type(),x.scalar_type()),"result type cannot be cast to output type");
 at::assert_no_internal_overlap(x);x.copy_(result);return x;
}
at::Tensor& masked_fill_tensor_(at::Tensor& x,const at::Tensor& mask,const at::Tensor& value){return assign(x,masked_fill_tensor(x,mask,value));}
at::Tensor& masked_fill_scalar_(at::Tensor& x,const at::Tensor& mask,const at::Scalar& value){return assign(x,masked_fill_scalar(x,mask,value));}
template<bool Minimum>at::Tensor bound_tensor(const at::Tensor& x,const at::Tensor& bound){
 auto dtype=at::result_type(x,bound);TORCH_CHECK(device_copy_supported(dtype),"clamp requires supported real dtype");
 auto a=x.to(dtype),b=bound.to(dtype);auto select=Minimum?at::lt(a,b):at::gt(a,b);
 if(dtype==at::kFloat)select=at::logical_or(select,at::isnan(b));
 return at::where(select,b,a);
}
template<bool Minimum>at::Tensor bound_scalar(const at::Tensor& x,const at::Scalar& bound){return bound_tensor<Minimum>(x,at::scalar_tensor(bound,at::TensorOptions().dtype(at::result_type(x,bound))));}
template<bool Minimum>at::Tensor& bound_tensor_(at::Tensor& x,const at::Tensor& bound){return assign(x,bound_tensor<Minimum>(x,bound));}
template<bool Minimum>at::Tensor& bound_scalar_(at::Tensor& x,const at::Scalar& bound){return assign(x,bound_scalar<Minimum>(x,bound));}
at::Tensor clamp_tensor(const at::Tensor& x,const std::optional<at::Tensor>& lo,const std::optional<at::Tensor>& hi){
 TORCH_CHECK(lo||hi,"clamp requires at least one bound");auto dtype=x.scalar_type();if(lo)dtype=at::promote_types(dtype,at::result_type(x,*lo));if(hi)dtype=at::promote_types(dtype,at::result_type(x,*hi));
 auto result=x.to(dtype);if(lo)result=bound_tensor<true>(result,lo->to(dtype));if(hi)result=bound_tensor<false>(result,hi->to(dtype));return result;
}
at::Tensor& clamp_tensor_(at::Tensor& x,const std::optional<at::Tensor>& lo,const std::optional<at::Tensor>& hi){return assign(x,clamp_tensor(x,lo,hi));}
at::Tensor& clamp_scalar_(at::Tensor& x,const std::optional<at::Scalar>& lo,const std::optional<at::Scalar>& hi){return assign(x,at::clamp(x,lo,hi));}
at::Tensor hardtanh_native(const at::Tensor& x,const at::Scalar& lo,const at::Scalar& hi){TORCH_CHECK(x.scalar_type()!=at::kBool,"hardtanh is not supported for bool");return at::clamp(x,lo,hi);}
at::Tensor& hardtanh_(at::Tensor& x,const at::Scalar& lo,const at::Scalar& hi){return assign(x,hardtanh_native(x,lo,hi));}
at::Tensor any_all(const at::Tensor& x){auto result=at::logical_not(at::all(at::logical_not(x)));return x.scalar_type()==at::kByte?result.to(at::kByte):result;}
at::Tensor any_dim(const at::Tensor& x,int64_t d,bool keep){auto result=at::logical_not(at::all(at::logical_not(x),d,keep));return x.scalar_type()==at::kByte?result.to(at::kByte):result;}
template<bool Any>at::Tensor boolean_dims(const at::Tensor& x,at::OptionalIntArrayRef dims,bool keep){
 if(!dims){auto r=Any?any_all(x):at::all(x);return keep?r.reshape(std::vector<int64_t>(x.dim(),1)):r;}
 std::vector<int64_t> axes;for(auto d:*dims){d=at::maybe_wrap_dim(d,x.dim());TORCH_CHECK(std::find(axes.begin(),axes.end(),d)==axes.end(),"dim appears multiple times");axes.push_back(d);}std::sort(axes.rbegin(),axes.rend());
 auto out=x.ne(0);for(auto d:axes)out=Any?any_dim(out,d,keep):at::all(out,d,keep);return x.scalar_type()==at::kByte?out.to(at::kByte):out;
}
at::Tensor reverse_sub(const at::Tensor& x,const at::Tensor& other,const at::Scalar& alpha){return at::sub(other,x,alpha);}
at::Tensor reverse_sub_scalar(const at::Tensor& x,const at::Scalar& other,const at::Scalar& alpha){return at::sub(at::scalar_tensor(other,at::TensorOptions().dtype(at::result_type(x,other))),x,alpha);}
at::Tensor addmv(const at::Tensor& bias,const at::Tensor& matrix,const at::Tensor& vector,const at::Scalar& beta,const at::Scalar& alpha){
 TORCH_CHECK(matrix.dim()==2&&vector.dim()==1,"addmv expects a matrix and a vector");
 TORCH_CHECK(bias.dim()<=1,"addmv bias must be scalar or vector");
 auto b=bias.dim()?bias.unsqueeze(1):bias;
 return at::addmm(b,matrix,vector.unsqueeze(1),beta,alpha).squeeze(1);
}
}
TORCH_LIBRARY_IMPL(aten,PrivateUse1,m){
 m.impl("masked_fill.Scalar",&masked_fill_scalar);m.impl("masked_fill.Tensor",&masked_fill_tensor);m.impl("masked_fill_.Scalar",&masked_fill_scalar_);m.impl("masked_fill_.Tensor",&masked_fill_tensor_);
 m.impl("clamp_min",&bound_scalar<true>);m.impl("clamp_max",&bound_scalar<false>);m.impl("clamp_min.Tensor",&bound_tensor<true>);m.impl("clamp_max.Tensor",&bound_tensor<false>);
 m.impl("clamp_min_",&bound_scalar_<true>);m.impl("clamp_max_",&bound_scalar_<false>);m.impl("clamp_min_.Tensor",&bound_tensor_<true>);m.impl("clamp_max_.Tensor",&bound_tensor_<false>);
 m.impl("clamp.Tensor",&clamp_tensor);m.impl("clamp_.Tensor",&clamp_tensor_);m.impl("clamp_",&clamp_scalar_);m.impl("hardtanh",&hardtanh_native);m.impl("hardtanh_",&hardtanh_);
 m.impl("any",&any_all);m.impl("any.dim",&any_dim);m.impl("any.dims",&boolean_dims<true>);m.impl("all.dims",&boolean_dims<false>);
 m.impl("rsub.Tensor",&reverse_sub);m.impl("rsub.Scalar",&reverse_sub_scalar);m.impl("addmv",&addmv);
}
}
