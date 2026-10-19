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
at::Tensor predicate(const at::Tensor& x,const at::Scalar& value,int op){
 if(x.scalar_type()!=at::kFloat||!can_pack_view(x)){
  auto a=native::cpu_fallback_tensor(x,__func__);at::Tensor y;
  if(op==0)y=at::isneginf(a);else if(op==1)y=at::gt(a,value);else if(op==2)y=at::eq(a,value);else if(op==3)y=at::ge(a,value);else if(op==4)y=at::le(a,value);else y=at::lt(a,value);return y.to(x.device());
 }
 auto out=at::empty(x.sizes(),x.options().dtype(at::kBool));if(out.numel()){auto a=packed(x);run("predicate",{base(a)},{write_base(out,at::kBool)},{u32(x.numel()),uint32_t(op),bits(value.toFloat())},(x.numel()+511)/512);}return out;
}
at::Tensor isneginf(const at::Tensor& x){return predicate(x,0,0);}
template<int Op>at::Tensor compare(const at::Tensor& x,const at::Scalar& s){return predicate(x,s,Op);}
at::Tensor all_dim(const at::Tensor& x,int64_t raw,bool keepdim){
 if(x.scalar_type()!=at::kBool){auto r=all_dim(x.ne(0),raw,keepdim);return x.scalar_type()==at::kByte?r.to(at::kByte):r;}
 TORCH_CHECK(can_pack_view(x),"all: Vulkan indexing/storage limits exceeded");
 int64_t rank=std::max<int64_t>(1,x.dim()),axis=raw<0?raw+rank:raw;TORCH_CHECK(axis>=0&&axis<rank,"all dimension out of range");
 auto shape=x.sizes().vec();if(x.dim()){if(keepdim)shape[axis]=1;else shape.erase(shape.begin()+axis);}
 auto out=at::empty(shape,x.options());if(!out.numel())return out;
 uint64_t outer=1,inner=1;for(int64_t d=0;d<x.dim();d++){if(d<axis)outer*=x.size(d);else if(d>axis)inner*=x.size(d);}
 auto a=x.numel()?packed(x,at::kBool):x;auto input=x.numel()?base(a,at::kBool):VulkanAllocator::instance().constant_float_buffer(1,0.f);run("bool_all",{input},{write_base(out,at::kBool)},{u32(outer),u32(x.dim()?x.size(axis):1),u32(inner)},(out.numel()+511)/512);return out;
}
at::Tensor all_native(const at::Tensor& x){
 if(x.scalar_type()!=at::kBool){auto r=all_native(x.ne(0));return x.scalar_type()==at::kByte?r.to(at::kByte):r;}
 TORCH_CHECK(can_pack_view(x),"all: Vulkan indexing/storage limits exceeded");
 return all_dim((x.numel()?packed(x,at::kBool):x).view({-1}),0,false);
}
at::Tensor where_native(const at::Tensor& cond,const at::Tensor& left,const at::Tensor& right){
 TORCH_CHECK(cond.scalar_type()==at::kBool,"where expected condition to be a boolean tensor");
 auto dev=cond.is_cpu()?(left.is_cpu()?right.device():left.device()):cond.device();auto dtype=at::result_type(left,right);
 if(!device_copy_supported(dtype)&&dtype!=at::kComplexFloat)return at::where(cpu_fallback_tensor(cond,__func__),cpu_fallback_tensor(left,__func__),cpu_fallback_tensor(right,__func__)).to(dev);
 for(const auto& t:{cond,left,right})TORCH_CHECK(t.device()==dev||(t.is_cpu()&&t.dim()==0),"where device mismatch");
 auto shape=at::infer_size_dimvector(at::infer_size_dimvector(cond.sizes(),left.sizes()),right.sizes());auto out=at::empty(shape,left.options().device(dev).dtype(dtype));if(!out.numel())return out;
 auto c=cond.to(dev).expand(shape),a=left.to(dev,dtype).expand(shape),b=right.to(dev,dtype).expand(shape);
 TORCH_CHECK(can_pack_view(c)&&can_pack_view(a)&&can_pack_view(b),"where exceeds Vulkan indexing limits");c=packed(c,at::kBool);a=packed(a,dtype);b=packed(b,dtype);
 auto lanes=out.element_size()==1?512:out.element_size()==2?256:128;
 run("where_typed",{base(c,at::kBool),base(a,dtype),base(b,dtype)},{write_base(out,dtype)},{u32(out.numel()),uint32_t(out.element_size())},(out.numel()+lanes-1)/lanes);return out;
}
at::Tensor& where_out(const at::Tensor& cond,const at::Tensor& left,const at::Tensor& right,at::Tensor& out){
 TORCH_CHECK(out.device()==cond.device(),"where output device mismatch");
 at::assert_no_internal_overlap(out);at::assert_no_partial_overlap(out,cond);at::assert_no_partial_overlap(out,left);at::assert_no_partial_overlap(out,right);
 auto value=where_native(cond,left,right);
 TORCH_CHECK(value.scalar_type()==out.scalar_type(),"where output dtype mismatch");
 at::native::resize_output(out,value.sizes());out.copy_(value);return out;
}
at::Tensor where_scalar_left(const at::Tensor& cond,const at::Scalar& value,const at::Tensor& other){
 auto dtype=at::result_type(other,value);return where_native(cond,at::scalar_tensor(value,at::TensorOptions().dtype(dtype)),other.to(dtype));
}
at::Tensor where_scalar_right(const at::Tensor& cond,const at::Tensor& other,const at::Scalar& value){
 auto dtype=at::result_type(other,value);return where_native(cond,other.to(dtype),at::scalar_tensor(value,at::TensorOptions().dtype(dtype)));
}
at::Tensor boolean_and(const at::Tensor& left,const at::Tensor& right){
 if(left.scalar_type()!=at::kBool||right.scalar_type()!=at::kBool||!can_pack_view(left)||!can_pack_view(right))return at::bitwise_and(native::cpu_fallback_tensor(left,__func__),native::cpu_fallback_tensor(right,__func__)).to(left.device());
 TORCH_CHECK(left.device()==right.device(),"Boolean device mismatch");
 auto shape=at::infer_size_dimvector(left.sizes(),right.sizes());auto a=left.expand(shape),b=right.expand(shape);
 if(!can_pack_view(a)||!can_pack_view(b))return at::bitwise_and(native::cpu_fallback_tensor(left,__func__),native::cpu_fallback_tensor(right,__func__)).to(left.device());
 auto out=at::empty(shape,left.options());if(out.numel()){a=packed(a,at::kBool);b=packed(b,at::kBool);run("bool_logic",{base(a,at::kBool),base(b,at::kBool)},{write_base(out,at::kBool)},{u32(out.numel()),0},(out.numel()+511)/512);}return out;
}
at::Tensor logical_and_native(const at::Tensor& a,const at::Tensor& b){
 if(a.scalar_type()==at::kBool&&b.scalar_type()==at::kBool)return boolean_and(a,b);
 return at::logical_and(native::cpu_fallback_tensor(a,__func__),native::cpu_fallback_tensor(b,__func__)).to(a.device());
}
at::Tensor& logical_and_out(const at::Tensor& a,const at::Tensor& b,at::Tensor& out){
 at::assert_no_internal_overlap(out);at::assert_no_partial_overlap(out,a);at::assert_no_partial_overlap(out,b);
 TORCH_CHECK(a.device()==b.device()&&out.device()==a.device(),"logical_and device mismatch");
 if(out.scalar_type()!=at::kBool){auto host=native::cpu_fallback_tensor(out,__func__);at::logical_and_out(host,native::cpu_fallback_tensor(a,__func__),native::cpu_fallback_tensor(b,__func__));at::native::resize_output(out,host.sizes());out.copy_(host);return out;}
 auto value=logical_and_native(a,b);at::native::resize_output(out,value.sizes());
 if(!out.numel())return out;
 if(out.is_contiguous()&&out.storage_offset()==0&&out.storage().nbytes()==size_t(out.numel())&&can_pack_view(out))run("bool_logic",{base(value,at::kBool),base(value,at::kBool)},{write_base(out,at::kBool)},{u32(out.numel()),0},(out.numel()+511)/512);
 else out.copy_(value);
 return out;
}
at::Tensor boolean_not(const at::Tensor& x){
 if(x.scalar_type()!=at::kBool||!can_pack_view(x))return at::bitwise_not(native::cpu_fallback_tensor(x,__func__)).to(x.device());
 auto out=at::empty(x.sizes(),x.options());if(out.numel()){auto a=packed(x,at::kBool);run("bool_logic",{base(a,at::kBool),base(a,at::kBool)},{write_base(out,at::kBool)},{u32(out.numel()),1},(out.numel()+511)/512);}return out;
}
at::Tensor power(const at::Tensor& x,const at::Scalar& exponent){
 if(x.scalar_type()!=at::kFloat||!can_pack_view(x))return at::pow(native::cpu_fallback_tensor(x,__func__),exponent).to(x.device());
 auto out=at::empty(x.sizes(),x.options());if(out.numel()){auto a=packed(x);run("power_clamp",{base(a)},{write_base(out)},{u32(out.numel()),0,bits(exponent.toFloat()),0},(out.numel()+127)/128);}return out;
}
at::Tensor clamp(const at::Tensor& x,const std::optional<at::Scalar>& low,const std::optional<at::Scalar>& high){
 TORCH_CHECK(low||high,"clamp requires at least one bound");
 if(device_copy_supported(x.scalar_type())&&x.scalar_type()!=at::kFloat){
  auto dtype=x.scalar_type();if(low)dtype=at::promote_types(dtype,at::result_type(x,*low));if(high)dtype=at::promote_types(dtype,at::result_type(x,*high));auto y=x.to(dtype);if(low)y=at::clamp_min(y,*low);if(high)y=at::clamp_max(y,*high);return y;
 }
 if(x.scalar_type()!=at::kFloat||!can_pack_view(x))return at::clamp(native::cpu_fallback_tensor(x,__func__),low,high).to(x.device());
 auto out=at::empty(x.sizes(),x.options());if(out.numel()){auto a=packed(x);run("power_clamp",{base(a)},{write_base(out)},{u32(out.numel()),1,bits(low?low->toFloat():-INFINITY),bits(high?high->toFloat():INFINITY)},(out.numel()+127)/128);}return out;
}
}
TORCH_LIBRARY_IMPL(aten,PrivateUse1,m){
 if(std::getenv("TORCH_VULKAN_DISABLE_NATIVE_PREDICATES"))return;



 m.impl("all.dim",&all_dim);m.impl("all",&all_native);m.impl("where.self",&where_native);m.impl("where.self_out",&where_out);m.impl("where.ScalarSelf",&where_scalar_left);m.impl("where.ScalarOther",&where_scalar_right);m.impl("pow.Tensor_Scalar",&power);m.impl("clamp",&clamp);
}
}
