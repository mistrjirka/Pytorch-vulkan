// Shape-independent compositions of native device selection primitives.
#include "native_utils.h"
#include "device_copy.h"
#include <ATen/ExpandUtils.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/native/Resize.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/WrapDimUtils.h>
#include <torch/library.h>
namespace torch_vulkan::ops {
namespace {
at::Tensor masked_fill_tensor(const at::Tensor& x,const at::Tensor& mask,const at::Tensor& value){
 TORCH_CHECK(value.dim()==0,"masked_fill_ only supports a 0-dimensional value tensor, but got tensor with ",value.dim()," dimension");
 auto converted=value;
 if(value.is_complex()&&!x.is_complex()){TORCH_CHECK(at::imag(value).eq(0).cpu().item<bool>(),"value cannot be converted to type ",x.scalar_type()," without overflow");converted=at::real(value);}
 TORCH_CHECK(mask.scalar_type()==at::kBool,"masked_fill_ only supports boolean masks");
 TORCH_CHECK(x.device()==mask.device(),"masked_fill device mismatch");
 TORCH_CHECK(value.device()==x.device()||value.is_cpu(),"masked_fill value device mismatch");
 return at::where(mask,converted.to(x.device(),x.scalar_type()),x);
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
// General tensor compositions; all arithmetic dispatches to the device backend.
at::Tensor roll_native(const at::Tensor& x,c10::SymIntArrayRef shifts,at::IntArrayRef dims){
 TORCH_CHECK(!shifts.empty(),"`shifts` required");
 if(dims.empty()){TORCH_CHECK(shifts.size()==1,"shifts and dimensions must align");return roll_native(x.reshape({-1}),shifts,{0}).reshape(x.sizes());}
 TORCH_CHECK(shifts.size()==dims.size(),"shifts and dimensions must align");auto result=x;
 for(size_t i=0;i<dims.size();i++){auto d=at::maybe_wrap_dim(dims[i],x.dim());auto n=x.size(d);if(!n)continue;auto shift=shifts[i].expect_int()%n;if(shift<0)shift+=n;if(shift)result=at::cat({result.narrow(d,n-shift,shift),result.narrow(d,0,n-shift)},d);}
 return result.is_same(x)?result.clone():result;
}
at::Tensor nansum_native(const at::Tensor& x,at::OptionalIntArrayRef dims,bool keep,std::optional<at::ScalarType> dtype){if(!x.is_floating_point()&&!x.is_complex())return at::sum(x,dims,keep,dtype);return at::sum(at::where(at::isnan(x),at::zeros({},x.options()),x),dims,keep,dtype);}
at::Tensor std_native(const at::Tensor& x,at::OptionalIntArrayRef dims,const std::optional<at::Scalar>& correction,bool keep){return at::sqrt(at::var(x,dims,correction,keep));}
std::tuple<at::Tensor,at::Tensor> std_mean_native(const at::Tensor& x,at::OptionalIntArrayRef dims,const std::optional<at::Scalar>& correction,bool keep){auto result=at::var_mean(x,dims,correction,keep);return {at::sqrt(std::get<0>(result)),std::get<1>(result)};}

at::Tensor masked_select_native(const at::Tensor& x,const at::Tensor& mask){
 TORCH_CHECK(mask.scalar_type()==at::kBool,"masked_select: expected BoolTensor for mask");TORCH_CHECK(mask.device()==x.device(),"masked_select tensors must be on the same device");
 auto shape=at::infer_size_dimvector(x.sizes(),mask.sizes());auto data=x.expand(shape).reshape({-1}),selection=mask.expand(shape).reshape({-1});
 c10::List<std::optional<at::Tensor>> indices;indices.push_back(selection);return at::index(data,indices);
}
at::Tensor& masked_select_out(const at::Tensor& x,const at::Tensor& mask,at::Tensor& out){
 TORCH_CHECK(out.device()==x.device()&&out.scalar_type()==x.scalar_type(),"masked_select output device/dtype mismatch");at::assert_no_internal_overlap(out);at::assert_no_overlap(out,x);at::assert_no_overlap(out,mask);auto value=masked_select_native(x,mask);at::native::resize_output(out,value.sizes());out.copy_(value);return out;
}
at::Tensor masked_scatter_native(const at::Tensor& x,const at::Tensor& mask,const at::Tensor& source){
 TORCH_CHECK(mask.scalar_type()==at::kBool,"masked_scatter_ only supports boolean masks");TORCH_CHECK(mask.device()==x.device()&&source.device()==x.device()&&source.scalar_type()==x.scalar_type(),"masked_scatter source device/dtype mismatch");
 auto shape=at::infer_size_dimvector(x.sizes(),mask.sizes());auto result=x.expand(shape).clone().reshape({-1});auto indices=at::nonzero(mask.expand(shape).reshape({-1})).select(1,0);
 TORCH_CHECK(source.numel()>=indices.numel(),"Number of elements of source < number of ones in mask");c10::List<std::optional<at::Tensor>> list;list.push_back(indices);at::_index_put_impl_(result,list,source.reshape({-1}).narrow(0,0,indices.numel()),false,false);return result.reshape(shape);
}
at::Tensor& masked_scatter_inplace(at::Tensor& x,const at::Tensor& mask,const at::Tensor& source){at::assert_no_partial_overlap(x,source);at::assert_no_overlap(x,mask);return assign(x,masked_scatter_native(x,mask,source));}

template<bool Max>at::Tensor finite_extrema(const at::Tensor& x,const at::Tensor& other){auto value=Max?at::maximum(x,other):at::minimum(x,other);if(x.is_floating_point())value=at::where(at::isnan(x),other,value);if(other.is_floating_point())value=at::where(at::isnan(other),x,value);return value;}
std::tuple<at::Tensor,at::Tensor> aminmax_native(const at::Tensor& x,std::optional<int64_t> dim,bool keep){if(!dim)TORCH_CHECK(x.numel(),"cannot compute aminmax over an empty dimension as the operation has no identity");if(dim)return {std::get<0>(at::min(x,*dim,keep)),std::get<0>(at::max(x,*dim,keep))};auto a=at::min(x),b=at::max(x);if(keep){std::vector<int64_t> shape(x.dim(),1);a=a.reshape(shape);b=b.reshape(shape);}return {a,b};}
at::Tensor count_nonzero_dims(const at::Tensor& x,at::IntArrayRef dims){return at::sum(at::ne(x,0),dims,false,at::kLong);}
at::Tensor count_nonzero_dim(const at::Tensor& x,std::optional<int64_t> dim){if(dim)return count_nonzero_dims(x,{*dim});return at::sum(at::ne(x,0),at::kLong);}
at::Tensor baddbmm_native(const at::Tensor& bias,const at::Tensor& a,const at::Tensor& b,const at::Scalar& beta,const at::Scalar& alpha){auto product=at::bmm(a,b).mul(alpha);auto shape=at::infer_size_dimvector(product.sizes(),bias.sizes());TORCH_CHECK(shape==product.sizes(),"baddbmm bias cannot expand output");if(beta.toDouble()==0)return product;return product.add(bias.mul(beta));}
at::Tensor cross_native(const at::Tensor& a,const at::Tensor& b,int64_t raw){TORCH_CHECK(a.dim()==b.dim(),"linalg.cross: inputs must have the same number of dimensions.");auto d=at::maybe_wrap_dim(raw,a.dim());TORCH_CHECK(a.size(d)==3&&b.size(d)==3,"linalg.cross: inputs dimension ",raw," must have length 3");return at::stack({a.select(d,1).mul(b.select(d,2)).sub(a.select(d,2).mul(b.select(d,1))),a.select(d,2).mul(b.select(d,0)).sub(a.select(d,0).mul(b.select(d,2))),a.select(d,0).mul(b.select(d,1)).sub(a.select(d,1).mul(b.select(d,0)))},d);}
at::Tensor& relu_inplace(at::Tensor& x){TORCH_CHECK(x.scalar_type()!=at::kBool,"Boolean inputs not supported for relu");return assign(x,at::clamp_min(x,0));}
at::Tensor signbit_native(const at::Tensor& x){TORCH_CHECK(!x.is_complex(),"signbit is not implemented for complex tensors");if(x.scalar_type()==at::kFloat)return at::lt(at::resolve_neg(x).contiguous().view(at::kInt),0);return at::lt(x,0);}
at::Tensor take_general(const at::Tensor& x,const at::Tensor& index){TORCH_CHECK(index.scalar_type()==at::kLong,"take(): Expected a long tensor for index");c10::List<std::optional<at::Tensor>> indices;indices.push_back(index);return at::index(x.reshape({-1}),indices);}
at::Tensor& take_out_general(const at::Tensor& x,const at::Tensor& index,at::Tensor& out){at::assert_no_internal_overlap(out);at::assert_no_overlap(out,x);at::assert_no_overlap(out,index);TORCH_CHECK(out.scalar_type()==x.scalar_type()&&out.device()==x.device(),"take output dtype/device mismatch");auto result=take_general(x,index);at::native::resize_output(out,result.sizes());out.copy_(result);return out;}
at::Tensor& put_general(at::Tensor& x,const at::Tensor& index,const at::Tensor& source,bool accumulate){TORCH_CHECK(index.scalar_type()==at::kLong&&source.scalar_type()==x.scalar_type()&&index.numel()==source.numel(),"put_ index/source dtype or size mismatch");at::assert_no_internal_overlap(x);at::assert_no_overlap(x,index);at::assert_no_overlap(x,source);auto flat=x.reshape({-1});c10::List<std::optional<at::Tensor>> indices;indices.push_back(index.reshape({-1}));at::_index_put_impl_(flat,indices,source.reshape({-1}),accumulate,false);x.copy_(flat.reshape(x.sizes()));return x;}
at::Tensor nonzero_static_general(const at::Tensor& x,c10::SymInt raw,int64_t fill){auto size=raw.expect_int();TORCH_CHECK(size>=0,"nonzero_static: size must be non-negative");auto result=at::full({size,x.dim()},fill,x.options().dtype(at::kLong));if(size&&x.numel()){auto indices=at::nonzero(x);auto n=std::min(size,indices.size(0));if(n)result.narrow(0,0,n).copy_(indices.narrow(0,0,n));}return result;}
at::Tensor channel_shuffle_general(const at::Tensor& x,int64_t groups){TORCH_CHECK(x.dim()>2,"channel_shuffle expects input with > 2 dims");TORCH_CHECK(groups>0&&x.size(1)%groups==0,"Number of channels must be divisible by groups");auto shape=x.sizes().vec();auto expanded=shape;expanded[1]=groups;expanded.insert(expanded.begin()+2,x.size(1)/groups);return x.reshape(expanded).transpose(1,2).reshape(shape);}
at::Tensor trace_general(const at::Tensor& x){TORCH_CHECK(x.dim()==2,"trace: expected a matrix");return at::sum(x.diagonal());}
at::Tensor addbmm_general(const at::Tensor& bias,const at::Tensor& a,const at::Tensor& b,const at::Scalar& beta,const at::Scalar& alpha){TORCH_CHECK(a.dim()==3&&b.dim()==3&&a.size(0)==b.size(0),"addbmm expects matching batches");auto result=at::sum(at::bmm(a,b),{0}).mul(alpha);return beta.toDouble()==0?result:result.add(bias.mul(beta));}

}
TORCH_LIBRARY_IMPL(aten,PrivateUse1,m){
 m.impl("take",&take_general);m.impl("take.out",&take_out_general);m.impl("put_",&put_general);m.impl("nonzero_static",&nonzero_static_general);m.impl("channel_shuffle",&channel_shuffle_general);m.impl("trace",&trace_general);m.impl("addbmm",&addbmm_general);
 m.impl("fmax",&finite_extrema<true>);m.impl("fmin",&finite_extrema<false>);m.impl("aminmax",&aminmax_native);m.impl("count_nonzero.dim_IntList",&count_nonzero_dims);m.impl("count_nonzero",&count_nonzero_dim);m.impl("baddbmm",&baddbmm_native);m.impl("linalg_cross",&cross_native);
 m.impl("relu_",&relu_inplace);m.impl("signbit",&signbit_native);
 m.impl("masked_select",&masked_select_native);m.impl("masked_select.out",&masked_select_out);m.impl("masked_scatter",&masked_scatter_native);m.impl("masked_scatter_",&masked_scatter_inplace);
 m.impl("masked_fill.Scalar",&masked_fill_scalar);m.impl("masked_fill.Tensor",&masked_fill_tensor);m.impl("masked_fill_.Scalar",&masked_fill_scalar_);m.impl("masked_fill_.Tensor",&masked_fill_tensor_);
 m.impl("clamp_min",&bound_scalar<true>);m.impl("clamp_max",&bound_scalar<false>);m.impl("clamp_min.Tensor",&bound_tensor<true>);m.impl("clamp_max.Tensor",&bound_tensor<false>);
 m.impl("clamp_min_",&bound_scalar_<true>);m.impl("clamp_max_",&bound_scalar_<false>);m.impl("clamp_min_.Tensor",&bound_tensor_<true>);m.impl("clamp_max_.Tensor",&bound_tensor_<false>);
 m.impl("clamp.Tensor",&clamp_tensor);m.impl("clamp_.Tensor",&clamp_tensor_);m.impl("clamp_",&clamp_scalar_);m.impl("hardtanh",&hardtanh_native);m.impl("hardtanh_",&hardtanh_);
 m.impl("roll",&roll_native);m.impl("nansum",&nansum_native);m.impl("std.correction",&std_native);m.impl("std_mean.correction",&std_mean_native);
 m.impl("any",&any_all);m.impl("any.dim",&any_dim);m.impl("any.dims",&boolean_dims<true>);m.impl("all.dims",&boolean_dims<false>);
 m.impl("rsub.Tensor",&reverse_sub);m.impl("rsub.Scalar",&reverse_sub_scalar);m.impl("addmv",&addmv);
}
}
