#include "native_utils.h"
#include "../vulkan_device.h"
#include <ATen/EmptyTensor.h>
#include <ATen/native/TensorFactories.h>
#include "../vulkan_generator.h"
#include <ATen/native/Resize.h>
#include <ATen/MemoryOverlap.h>
#include <torch/library.h>
#include <cmath>
namespace torch_vulkan::ops {
namespace {
using namespace native;
at::Tensor efficient_zero(c10::SymIntArrayRef size,std::optional<at::ScalarType> dtype,std::optional<at::Layout> layout,std::optional<at::Device> device,std::optional<bool> pin){
 auto dev=device.value_or(c10::Device(c10::DeviceType::PrivateUse1,0));validate_vulkan_device(dev);TORCH_CHECK(layout.value_or(at::kStrided)==at::kStrided&&!pin.value_or(false),"invalid ZeroTensor layout or pinning");
 static at::native::ZeroTensorAllocator allocator(c10::Device(c10::DeviceType::PrivateUse1,0));
 return at::Tensor(at::detail::empty_generic_symint(size,&allocator,c10::DispatchKeySet({c10::DispatchKey::PrivateUse1,c10::DispatchKey::ZeroTensor}),c10::dtype_or_default(dtype),std::nullopt));
}

at::Tensor& arange_out(const at::Scalar& start,const at::Scalar& end,const at::Scalar& step,at::Tensor& out){
 double a=start.toDouble(),b=end.toDouble(),s=step.toDouble();
 TORCH_CHECK(s!=0,"step must be nonzero");TORCH_CHECK(std::isfinite(a)&&std::isfinite(b)&&std::isfinite(s),"unsupported range: non-finite bounds or step");
 TORCH_CHECK((s>0&&b>=a)||(s<0&&b<=a),"upper bound and larger bound inconsistent with step sign");
 bool integer=out.scalar_type()==at::kLong||out.scalar_type()==at::kInt;
 TORCH_CHECK(integer||out.scalar_type()==at::kFloat,"Vulkan arange supports FP32/int32/int64");
 uint64_t n;
 int64_t ia=0,is=0;
 if(integer){ia=start.toLong();auto ib=end.toLong();is=step.toLong();TORCH_CHECK(is,"step must be nonzero");__int128 d=__int128(ib)-ia;if(is<0)d=-d;auto delta=is<0?-__int128(is):__int128(is);TORCH_CHECK(d>=0,"upper bound and larger bound inconsistent with step sign");auto count=(d+delta-1)/delta;TORCH_CHECK(count<=UINT32_MAX,"arange exceeds Vulkan indexing limits");n=count;}
 else {double count=std::ceil((b-a)/s);TORCH_CHECK(count>=0&&count<=UINT32_MAX,"arange exceeds Vulkan indexing limits");n=uint64_t(count);}
 at::assert_no_internal_overlap(out);at::native::resize_output(out,{int64_t(n)});if(!n)return out;
 auto value=at::empty({int64_t(n)},out.options());uint64_t aa=uint64_t(ia),ss=uint64_t(is);
 run("sequence_factory",{},{write_base(value,value.scalar_type())},{u32(n),integer?uint32_t(out.element_size()/4):0u,0,bits(float(a)),bits(float(s)),uint32_t(aa),uint32_t(aa>>32),uint32_t(ss),uint32_t(ss>>32)},(n+127)/128);out.copy_(value);return out;
}
at::Tensor& arange_end(const at::Scalar& end,at::Tensor& out){return arange_out(0,end,1,out);}
at::Tensor& linspace_out(const at::Scalar& start,const at::Scalar& end,int64_t steps,at::Tensor& out){
 TORCH_CHECK(steps>=0,"number of steps must be non-negative");u32(steps);
 TORCH_CHECK(out.scalar_type()==at::kFloat||out.scalar_type()==at::kInt||out.scalar_type()==at::kLong,"Vulkan linspace supports FP32/int32/int64");
 at::assert_no_internal_overlap(out);at::native::resize_output(out,{steps});if(!steps)return out;
 auto value=at::empty({steps},out.options().dtype(at::kFloat));
 run("sequence_factory",{},{write_base(value)},{u32(steps),0,1,bits(start.toFloat()),bits(end.toFloat()),0,0,0,0},(steps+127)/128);out.copy_(value);return out;
}
at::Tensor& randperm_out(c10::SymInt count,std::optional<at::Generator> gen,at::Tensor& out){
 int64_t n=count.expect_int();TORCH_CHECK(n>=0,"n must be non-negative");u32(n);
 TORCH_CHECK(out.scalar_type()==at::kFloat||out.scalar_type()==at::kInt||out.scalar_type()==at::kLong,"Vulkan randperm supports FP32/int32/int64");
 if(out.scalar_type()==at::kFloat)TORCH_CHECK(n<=16777217,"n is too large for float32");
 at::assert_no_internal_overlap(out);at::native::resize_output(out,{n});if(!n)return out;
 auto value=at::empty({n},out.options().dtype(at::kLong));auto state=random_state(gen,n);
 run("randperm",{},{write_base(value,at::kLong)},{u32(n),state[0],state[1],state[2],state[3]},1);out.copy_(value);return out;
}
at::Tensor& randperm_default(c10::SymInt count,at::Tensor& out){return randperm_out(count,std::nullopt,out);}
}
TORCH_LIBRARY_IMPL(aten,PrivateUse1,m){m.impl("_efficientzerotensor",&efficient_zero);m.impl("arange.start_out",&arange_out);m.impl("arange.out",&arange_end);m.impl("linspace.out",&linspace_out);m.impl("randperm.generator_out",&randperm_out);m.impl("randperm.out",&randperm_default);}
}
namespace torch_vulkan::ops {namespace {
at::Tensor& logspace_out_native(const at::Scalar& start,const at::Scalar& end,int64_t steps,double base_value,at::Tensor& out){TORCH_CHECK(steps>=0,"number of steps must be non-negative");at::assert_no_internal_overlap(out);auto exponents=at::linspace(start,end,steps,out.options().dtype(at::kFloat));auto value=at::pow(at::scalar_tensor(base_value,out.options().dtype(at::kFloat)),exponents);at::native::resize_output(out,{steps});out.copy_(value);return out;}
}
TORCH_LIBRARY_IMPL(aten,PrivateUse1,m){m.impl("logspace.out",&logspace_out_native);}
}
namespace torch_vulkan::ops {namespace {
at::Tensor triangular_indices_native(int64_t rows,int64_t cols,int64_t offset,std::optional<at::ScalarType> dtype,std::optional<at::Layout> layout,std::optional<at::Device> device,std::optional<bool> pin,bool upper){
 TORCH_CHECK(rows>=0,"row must be non-negative, got ",rows);TORCH_CHECK(cols>=0,"col must be non-negative, got ",cols);TORCH_CHECK(layout.value_or(at::kStrided)==at::kStrided,"only support layout=torch.strided");TORCH_CHECK(!pin.value_or(false),"Vulkan triangular indices cannot be pinned");auto type=dtype.value_or(at::kLong);TORCH_CHECK(type==at::kInt||type==at::kLong,"Vulkan triangular indices requires int32 or int64");auto dev=device.value_or(at::Device(at::kPrivateUse1,0));validate_vulkan_device(dev);u32(rows);u32(cols);
 auto clip=[&](__int128 x){return int64_t(std::max<__int128>(0,std::min<__int128>(rows,x)));};auto a=upper?clip(1-__int128(offset)):clip(-__int128(offset));auto b=upper?clip(__int128(cols)-offset):clip(__int128(cols)-offset-1);b=std::max(a,b);int64_t first=b>a?int64_t(upper?__int128(cols)-a-offset:__int128(a)+offset+1):0;__int128 m=b-a;__int128 count=upper?__int128(a)*cols+m*first-m*(m-1)/2:m*first+m*(m-1)/2+__int128(rows-b)*cols;TORCH_CHECK(count<=UINT32_MAX/2,"triangular indices exceeds Vulkan indexing limits");auto n=int64_t(count);auto out=at::empty({2,n},at::TensorOptions().device(dev).dtype(type));if(n)run("triangular_indices",{},{write_base(out,type)},{u32(n),u32(rows),u32(cols),u32(a),u32(b),u32(first),uint32_t(upper),uint32_t(type==at::kLong?2:1)},(n+127)/128);return out;
}
at::Tensor triu_indices_native(int64_t r,int64_t c,int64_t k,std::optional<at::ScalarType> t,std::optional<at::Layout> l,std::optional<at::Device> d,std::optional<bool> p){return triangular_indices_native(r,c,k,t,l,d,p,true);}
at::Tensor tril_indices_native(int64_t r,int64_t c,int64_t k,std::optional<at::ScalarType> t,std::optional<at::Layout> l,std::optional<at::Device> d,std::optional<bool> p){return triangular_indices_native(r,c,k,t,l,d,p,false);}
at::Tensor& triu_indices_out_native(int64_t r,int64_t c,int64_t k,at::Tensor& out){at::assert_no_internal_overlap(out);auto value=triu_indices_native(r,c,k,out.scalar_type(),out.layout(),out.device(),false);at::native::resize_output(out,value.sizes());out.copy_(value);return out;}
at::Tensor& tril_indices_out_native(int64_t r,int64_t c,int64_t k,at::Tensor& out){at::assert_no_internal_overlap(out);auto value=tril_indices_native(r,c,k,out.scalar_type(),out.layout(),out.device(),false);at::native::resize_output(out,value.sizes());out.copy_(value);return out;}
}
TORCH_LIBRARY_IMPL(aten,PrivateUse1,m){m.impl("triu_indices",&triu_indices_native);m.impl("tril_indices",&tril_indices_native);m.impl("triu_indices.out",&triu_indices_out_native);m.impl("tril_indices.out",&tril_indices_out_native);}
}
