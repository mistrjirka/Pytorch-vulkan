#include "native_utils.h"
#include "device_copy.h"
#include "integer_ops.h"
#include <ATen/MemoryOverlap.h>
#include <ATen/ExpandUtils.h>
#include <ATen/native/Resize.h>
#include <torch/library.h>

namespace torch_vulkan::ops {
namespace {
using namespace native;
enum class Pointwise : uint32_t { LerpScalar, LerpTensor, Addcmul, Addcdiv, Maximum, Minimum };

void fp32(const at::Tensor& x) {
  TORCH_CHECK(x.scalar_type()==at::kFloat && can_pack_view(x),
              "Vulkan optimizer primitives require packable FP32 tensors");
}
bool fusion_enabled() {
  // Startup-only diagnostic switch for matched comparisons with the reference.
  static const bool enabled=[] {
    const auto* flag=std::getenv("TORCH_VULKAN_DISABLE_FUSED_OPTIMIZER_PRIMITIVES");
    return !flag || std::strcmp(flag,"1")!=0;
  }();
  return enabled;
}
at::DimVector broadcast(const at::Tensor& a,const at::Tensor& b,const at::Tensor& c) {
  auto shape=at::infer_size_dimvector(a.sizes(),b.sizes());
  if(c.defined())shape=at::infer_size_dimvector(shape,c.sizes());
  return shape;
}
bool same_mapping(const at::Tensor& a,const at::Tensor& b) {
  return a.sizes()==b.sizes() && a.strides()==b.strides() &&
         a.storage_offset()==b.storage_offset();
}
bool independent(const at::Tensor& x) {
  if(x.is_contiguous() || x.numel()<2)return true;
  std::vector<std::pair<int64_t,int64_t>> dims;
  for(int64_t d=0;d<x.dim();++d)if(x.size(d)>1)dims.emplace_back(x.stride(d),x.size(d));
  std::sort(dims.begin(),dims.end());
  uint64_t span=1;
  for(auto [stride,size]:dims) {
    if(stride<0 || uint64_t(stride)<span)return false;
    span+=uint64_t(size-1)*uint64_t(stride);
  }
  return true;
}

// Keep the composed device implementation for unsupported geometry/capabilities
// and for A/B measurements. No CPU numerical fallback is introduced.
at::Tensor reference(const at::Tensor& a,const at::Tensor& b,const at::Tensor& c,
                     Pointwise op,const at::Scalar& value) {
  switch(op) {
    case Pointwise::LerpScalar:return a.add(b.sub(a).mul(value));
    case Pointwise::LerpTensor:return a.add(b.sub(a).mul(c));
    case Pointwise::Addcmul:return a.add(b.mul(c),value);
    case Pointwise::Addcdiv:return a.add(b.div(c),value);
    default: {
      auto shape=broadcast(a,b,{});
      auto out=at::empty(shape,a.options());if(!out.numel())return out;
      auto x=packed(a.expand(shape)),y=packed(b.expand(shape));
      run("extrema_fp32",{base(x),base(y)},{write_base(out)},
          {u32(out.numel()),uint32_t(op==Pointwise::Minimum)},(out.numel()+127)/128);
      return out;
    }
  }
}

bool fused_write(const at::Tensor& a,const at::Tensor& b,const at::Tensor& c,
                 at::Tensor& out,at::IntArrayRef shape,Pointwise op,const at::Scalar& value) {
  if(!fusion_enabled())return false;
  const auto& limits=device_limits();
  if(limits.maxPerStageDescriptorStorageBuffers<5 ||
     limits.maxDescriptorSetStorageBuffers<5 || limits.maxPushConstantsSize<36 ||
     !can_pack_view(out) || out.is_neg() || out.is_conj() ||
     a.is_neg() || b.is_neg() || (c.defined() && c.is_neg()) ||
     !independent(out))return false;
  for(const auto& input:{a,b,c}) {
    if(!input.defined())continue;
    // A permutation or broadcast of an aliased destination needs a snapshot;
    // the reference computes that snapshot before copying back.
    if(out.storage().is_alias_of(input.storage()) && !same_mapping(out,input))return false;
  }
  if(!out.numel())return true;
  auto flat_input=[&](const at::Tensor& input) {
    return input.numel()==1 || (input.sizes()==shape && input.is_contiguous());
  };
  const bool flat=out.is_contiguous() && flat_input(a) && flat_input(b) &&
                  (!c.defined() || flat_input(c));
  at::Tensor x,y,z,metadata;
  if(!flat) {
    // Inputs and output have already passed can_pack_view. Expansion keeps the
    // input storage bounds/strides and adopts the checked output rank/numel.
    x=a.expand(shape);y=b.expand(shape);
    if(c.defined())z=c.expand(shape);
    const auto rank=shape.size();
    std::vector<uint32_t> data(5*rank);
    for(size_t d=0;d<rank;++d) {
      data[d]=u32(shape[d]);data[rank+d]=u32(x.stride(d));
      data[2*rank+d]=u32(y.stride(d));data[3*rank+d]=z.defined()?u32(z.stride(d)):0;
      data[4*rank+d]=u32(out.stride(d));
    }
    auto host=at::empty({int64_t(data.size())},at::TensorOptions().dtype(at::kInt).device(at::kCPU));
    std::memcpy(host.data_ptr<int32_t>(),data.data(),data.size()*sizeof(uint32_t));
    metadata=host.to(out.device());
  }
  uint32_t scalar_mask=uint32_t(a.numel()==1) | (uint32_t(b.numel()==1)<<1) |
                       (uint32_t(!c.defined() || c.numel()==1)<<2);
  auto dummy=VulkanAllocator::instance().constant_float_buffer(1,0);
  // Mutable access must happen before retrieving bindings (PyTorch COW).
  auto output_buffer=write_base(out);
  run("optimizer_pointwise",
      {base(flat?a:x),base(flat?b:y),c.defined()?base(flat?c:z):dummy,
       metadata.defined()?base(metadata,at::kInt):dummy},
      {output_buffer},
      {u32(out.numel()),flat?0:u32(shape.size()),u32(a.storage_offset()),
       u32(b.storage_offset()),c.defined()?u32(c.storage_offset()):0,
       u32(out.storage_offset()),scalar_mask,uint32_t(op),bits(value.toFloat())},
      (out.numel()+127)/128);
  return true;
}
void check_inputs(const at::Tensor& a,const at::Tensor& b,const at::Tensor& c) {
  fp32(a);fp32(b);
  TORCH_CHECK(a.device()==b.device(),"optimizer primitives device mismatch");
  if(c.defined()){fp32(c);TORCH_CHECK(a.device()==c.device(),"optimizer primitives device mismatch");}
}
at::Tensor compute(const at::Tensor& a,const at::Tensor& b,const at::Tensor& c,
                   Pointwise op,const at::Scalar& value) {
  check_inputs(a,b,c);
  if(!fusion_enabled())return reference(a,b,c,op,value);
  auto shape=broadcast(a,b,c);auto out=at::empty(shape,a.options());
  if(fused_write(a,b,c,out,shape,op,value))return out;
  return reference(a,b,c,op,value);
}
at::Tensor& output(at::Tensor& out,const at::Tensor& result) {
  TORCH_CHECK(out.device()==result.device() && out.scalar_type()==result.scalar_type(),
              "optimizer output device/dtype mismatch");
  at::assert_no_internal_overlap(out);at::native::resize_output(out,result.sizes());
  out.copy_(result);return out;
}
at::Tensor& into(at::Tensor& out,const at::Tensor& a,const at::Tensor& b,const at::Tensor& c,
                 Pointwise op,const at::Scalar& value,bool inplace=false) {
  check_inputs(a,b,c);
  TORCH_CHECK(out.device()==a.device() && out.scalar_type()==a.scalar_type(),
              "optimizer output device/dtype mismatch");
  auto shape=broadcast(a,b,c);
  if(inplace)TORCH_CHECK(out.sizes()==at::IntArrayRef(shape),
      "output with shape ",out.sizes()," does not match the broadcast shape ",shape);
  at::assert_no_internal_overlap(out);
  bool alias=false;
  for(const auto& input:{a,b,c})if(input.defined()) {
    at::assert_no_partial_overlap(out,input);
    alias|=out.storage().is_alias_of(input.storage());
  }
  // Compute before resizing when a destination also supplies input storage.
  if(out.sizes()!=at::IntArrayRef(shape) && alias)
    return output(out,reference(a,b,c,op,value));
  at::native::resize_output(out,shape);
  if(fused_write(a,b,c,out,shape,op,value))return out;
  return output(out,reference(a,b,c,op,value));
}

at::Tensor lerp_scalar(const at::Tensor& x,const at::Tensor& e,const at::Scalar& w){return compute(x,e,{},Pointwise::LerpScalar,w);}
at::Tensor lerp_tensor(const at::Tensor& x,const at::Tensor& e,const at::Tensor& w){return compute(x,e,w,Pointwise::LerpTensor,0);}
at::Tensor addcmul_native(const at::Tensor& x,const at::Tensor& a,const at::Tensor& b,const at::Scalar& v){
  if(integer_type(x.scalar_type())&&a.scalar_type()==x.scalar_type()&&b.scalar_type()==x.scalar_type())return x.add(a.mul(b).mul(v.toLong()));
  return compute(x,a,b,Pointwise::Addcmul,v);
}
at::Tensor addcdiv_native(const at::Tensor& x,const at::Tensor& a,const at::Tensor& b,const at::Scalar& v){return compute(x,a,b,Pointwise::Addcdiv,v);}
at::Tensor extrema(const at::Tensor& x,const at::Tensor& y,bool minimum){
  if(device_copy_supported(x.scalar_type())&&device_copy_supported(y.scalar_type())&&
     (x.scalar_type()!=at::kFloat||y.scalar_type()!=at::kFloat||x.device()!=y.device()))
    return minimum?at::clamp_max(x,y):at::clamp_min(x,y);
  return compute(x,y,{},minimum?Pointwise::Minimum:Pointwise::Maximum,0);
}
at::Tensor maximum_native(const at::Tensor& x,const at::Tensor& y){return extrema(x,y,false);}
at::Tensor minimum_native(const at::Tensor& x,const at::Tensor& y){return extrema(x,y,true);}
at::Tensor& extrema_out(const at::Tensor& x,const at::Tensor& y,at::Tensor& out,bool minimum){
  if(x.scalar_type()!=at::kFloat||y.scalar_type()!=at::kFloat||x.device()!=y.device())return output(out,extrema(x,y,minimum));
  return into(out,x,y,{},minimum?Pointwise::Minimum:Pointwise::Maximum,0);
}
at::Tensor& maximum_out(const at::Tensor& x,const at::Tensor& y,at::Tensor& out){return extrema_out(x,y,out,false);}
at::Tensor& minimum_out(const at::Tensor& x,const at::Tensor& y,at::Tensor& out){return extrema_out(x,y,out,true);}
at::Tensor& lerp_scalar_out(const at::Tensor& x,const at::Tensor& e,const at::Scalar& w,at::Tensor& out){return into(out,x,e,{},Pointwise::LerpScalar,w);}
at::Tensor& lerp_tensor_out(const at::Tensor& x,const at::Tensor& e,const at::Tensor& w,at::Tensor& out){return into(out,x,e,w,Pointwise::LerpTensor,0);}
at::Tensor& lerp_scalar_inplace(at::Tensor& x,const at::Tensor& e,const at::Scalar& w){return into(x,x,e,{},Pointwise::LerpScalar,w,true);}
at::Tensor& lerp_tensor_inplace(at::Tensor& x,const at::Tensor& e,const at::Tensor& w){return into(x,x,e,w,Pointwise::LerpTensor,0,true);}
at::Tensor& addcmul_out(const at::Tensor& x,const at::Tensor& a,const at::Tensor& b,const at::Scalar& v,at::Tensor& out){
  if(integer_type(x.scalar_type())&&a.scalar_type()==x.scalar_type()&&b.scalar_type()==x.scalar_type())return output(out,addcmul_native(x,a,b,v));
  return into(out,x,a,b,Pointwise::Addcmul,v);
}
at::Tensor& addcdiv_out(const at::Tensor& x,const at::Tensor& a,const at::Tensor& b,const at::Scalar& v,at::Tensor& out){return into(out,x,a,b,Pointwise::Addcdiv,v);}
at::Tensor& addcmul_inplace(at::Tensor& x,const at::Tensor& a,const at::Tensor& b,const at::Scalar& v){
  if(integer_type(x.scalar_type())&&a.scalar_type()==x.scalar_type()&&b.scalar_type()==x.scalar_type()){at::assert_no_partial_overlap(x,a);at::assert_no_partial_overlap(x,b);auto result=addcmul_native(x,a,b,v);TORCH_CHECK(x.sizes()==result.sizes(),"inplace addcmul broadcast mismatch");return output(x,result);}
  return into(x,x,a,b,Pointwise::Addcmul,v,true);
}
at::Tensor& addcdiv_inplace(at::Tensor& x,const at::Tensor& a,const at::Tensor& b,const at::Scalar& v){return into(x,x,a,b,Pointwise::Addcdiv,v,true);}
}
TORCH_LIBRARY_IMPL(aten,PrivateUse1,m){
 m.impl("maximum",&maximum_native);m.impl("minimum",&minimum_native);m.impl("maximum.out",&maximum_out);m.impl("minimum.out",&minimum_out);
 m.impl("lerp.Scalar",&lerp_scalar);m.impl("lerp.Tensor",&lerp_tensor);m.impl("lerp.Scalar_out",&lerp_scalar_out);m.impl("lerp.Tensor_out",&lerp_tensor_out);m.impl("lerp_.Scalar",&lerp_scalar_inplace);m.impl("lerp_.Tensor",&lerp_tensor_inplace);
 m.impl("addcmul",&addcmul_native);m.impl("addcdiv",&addcdiv_native);m.impl("addcmul.out",&addcmul_out);m.impl("addcdiv.out",&addcdiv_out);m.impl("addcmul_",&addcmul_inplace);m.impl("addcdiv_",&addcdiv_inplace);
}
}
