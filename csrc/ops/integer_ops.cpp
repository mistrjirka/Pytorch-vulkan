#include "native_utils.h"
#include "integer_ops.h"
#include <ATen/ExpandUtils.h>
namespace torch_vulkan::ops {
using namespace native;
bool integer_type(at::ScalarType type){return type==at::kInt||type==at::kLong;}
at::Tensor integer_binary(const at::Tensor& x,const at::Tensor& y,const at::Scalar& alpha,bool multiply){
 TORCH_CHECK(integer_type(x.scalar_type())&&integer_type(y.scalar_type())&&alpha.isIntegral(true),"integer arithmetic requires integer tensors and alpha");
 auto device=x.is_cpu()?y.device():x.device();TORCH_CHECK(device.type()==c10::DeviceType::PrivateUse1&&(x.device()==device||(x.is_cpu()&&x.dim()==0))&&(y.device()==device||(y.is_cpu()&&y.dim()==0)),"integer arithmetic device mismatch");
 auto dtype=at::result_type(x,y);auto shape=at::infer_size_dimvector(x.sizes(),y.sizes());auto out=at::empty(shape,x.options().device(device).dtype(dtype));if(!out.numel())return out;
 auto a=x.is_cpu()?x.to(device):x,b=y.is_cpu()?y.to(device):y;a=packed(a.expand(shape),a.scalar_type());b=packed(b.expand(shape),b.scalar_type());uint64_t value=uint64_t(alpha.toLong());
 run("integer_binary",{base(a,a.scalar_type()),base(b,b.scalar_type())},{base(out,dtype)},{u32(out.numel()),uint32_t(a.element_size()/4),uint32_t(b.element_size()/4),uint32_t(out.element_size()/4),uint32_t(value),uint32_t(value>>32),uint32_t(multiply)},(out.numel()+127)/128);return out;
}
at::Tensor integer_sum(const at::Tensor& x,at::OptionalIntArrayRef dims,bool keepdim,std::optional<at::ScalarType> dtype){
 auto output_type=dtype.value_or(at::kLong);TORCH_CHECK((integer_type(x.scalar_type())||x.scalar_type()==at::kBool)&&integer_type(output_type)&&can_pack_view(x),"integer sum supports int32/int64 output");
 int64_t rank=std::max<int64_t>(1,x.dim());uint32_t mask=0;if(!dims||dims->empty())mask=(1u<<rank)-1;else for(auto raw:*dims){auto d=raw<0?raw+rank:raw;TORCH_CHECK(d>=0&&d<rank&&!(mask&(1u<<d)),"invalid sum dimensions");mask|=1u<<d;}
 std::vector<int64_t> shape;uint64_t reduced=1;for(int d=0;d<x.dim();d++){if(mask&(1u<<d)){reduced*=x.size(d);if(keepdim)shape.push_back(1);}else shape.push_back(x.size(d));}
 auto out=at::empty(shape,x.options().dtype(output_type));if(!out.numel())return out;auto input=x.numel()?packed(x,x.scalar_type()):at::Tensor();auto dummy=VulkanAllocator::instance().constant_float_buffer(1,0);
 std::vector<uint32_t> p={u32(out.numel()),u32(reduced),mask,x.scalar_type()==at::kBool?0u:uint32_t(x.element_size()/4),uint32_t(out.element_size()/4)};p.resize(13,1);for(int d=0;d<x.dim();d++)p[5+d]=u32(x.size(d));
 run("integer_reduce",{input.defined()?base(input,input.scalar_type()):dummy},{base(out,output_type)},p,(out.numel()+127)/128);return out;
}
}
