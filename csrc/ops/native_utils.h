#pragma once
#include "../vulkan_context.h"
#include "../execution_audit.h"
#include "../vulkan_allocator.h"
#include <ATen/Functions.h>
#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <limits>

namespace torch_vulkan::ops::native {
// Startup-only A/B control: retain the exact existing ATen CPU fallback.
inline bool batch_norm_resize_enabled() {
  const char* flag=std::getenv("TORCH_VULKAN_DISABLE_BN_RESIZE");
  return !flag || std::strcmp(flag,"1")!=0;
}
// Inline CPU implementations must obey the same strict gate as dispatcher fallback.
// Explicit user transfers and scalar validation/status downloads do not use this.
inline void cpu_fallback_check(const char* operation) {
  audit::fallback_calls.fetch_add(1);
  TORCH_CHECK(!std::getenv("TORCH_VULKAN_FORBID_CPU_FALLBACK"),
              "CPU fallback forbidden: Vulkan inline::",operation);
}
inline at::Tensor cpu_fallback_tensor(const at::Tensor& t,const char* operation) {
  cpu_fallback_check(operation);return t.cpu();
}
using Buffer = std::shared_ptr<kp::Tensor>;
inline uint32_t u32(int64_t v) {
  TORCH_CHECK(v>=0 && v<=UINT32_MAX,"Vulkan native operator indexing exceeds uint32");
  return static_cast<uint32_t>(v);
}
inline uint32_t bits(float v) { uint32_t r; std::memcpy(&r,&v,4); return r; }
inline Buffer base(const at::Tensor& t,at::ScalarType dtype=at::kFloat) {
  TORCH_CHECK(t.device().type()==c10::DeviceType::PrivateUse1 && t.scalar_type()==dtype,
              "Vulkan native operator tensor device/dtype mismatch");
  if(t._is_zerotensor())return VulkanAllocator::instance().constant_float_buffer(std::max<size_t>(1,(t.storage().nbytes()+3)/4),0);
  return VulkanAllocator::instance().get_kompute_tensor(t.storage().data_ptr().get());
}
// Mutable access triggers PyTorch copy-on-write before a shader can write.
inline Buffer write_base(const at::Tensor& t,at::ScalarType dtype=at::kFloat) {
  t.mutable_data_ptr();
  return base(t,dtype);
}
// A process selects one physical device before constructing VulkanContext.
// Its immutable limits can be shared by all native operators and dispatches.
inline const vk::PhysicalDeviceLimits& device_limits() {
  static const auto limits=VulkanContext::instance().manager().getDeviceProperties().limits;
  return limits;
}
inline kp::Workgroup grid(uint64_t groups) {
  const auto& l=device_limits();
  auto x=std::min<uint64_t>(groups,l.maxComputeWorkGroupCount[0]);
  auto y=x ? (groups+x-1)/x : 0;
  TORCH_CHECK(x && y<=l.maxComputeWorkGroupCount[1],"Vulkan dispatch exceeds device limits");
  return kp::Workgroup({static_cast<uint32_t>(x),static_cast<uint32_t>(y),1});
}
inline void run(const std::string& shader,const std::vector<Buffer>& inputs,
                const std::vector<Buffer>& outputs,const std::vector<uint32_t>& push,
                uint64_t groups,uint32_t shared_bytes=0) {
  auto& ctx=VulkanContext::instance();
  const auto& l=device_limits();
  auto bindings=inputs; bindings.insert(bindings.end(),outputs.begin(),outputs.end());
  TORCH_CHECK(l.maxComputeWorkGroupInvocations>=128 && l.maxComputeWorkGroupSize[0]>=128 &&
              l.maxComputeSharedMemorySize>=shared_bytes &&
              push.size()*4<=l.maxPushConstantsSize &&
              bindings.size()<=l.maxPerStageDescriptorStorageBuffers &&
              bindings.size()<=l.maxDescriptorSetStorageBuffers,
              "Vulkan native operator exceeds physical device capabilities");
  for(const auto& b:bindings)
    TORCH_CHECK(b->constructDescriptorBufferInfo().range<=l.maxStorageBufferRange,
                "Vulkan native operator exceeds storage buffer range");
  auto algo=ctx.get_or_create_algorithm(shader,bindings,grid(groups),push);
  if(ctx.can_enqueue(bindings)) ctx.enqueue_compute(algo,bindings,outputs);
  else ctx.run_compute(algo,shader,inputs,outputs);
}
// General rank/stride iteration uses a device metadata buffer instead of the
// eight-axis push-constant ABI. Existing fixed-axis eligibility stays unchanged.
inline bool can_iterate_fp32(const at::Tensor& x) {
 if(x.layout()!=at::kStrided||x.device().type()!=c10::DeviceType::PrivateUse1||x.scalar_type()!=at::kFloat||x.dim()>64||x.numel()>UINT32_MAX||x.storage_offset()<0||uint64_t(x.storage_offset())>UINT32_MAX)return false;
 if(x.storage().nbytes()>device_limits().maxStorageBufferRange)return false;
 uint64_t last=x.storage_offset();for(int64_t d=0;d<x.dim();d++){if(x.size(d)>UINT32_MAX||x.stride(d)<0||uint64_t(x.stride(d))>UINT32_MAX)return false;if(x.size(d)>1){auto extent=uint64_t(x.size(d)-1)*x.stride(d);if(extent>UINT32_MAX-last)return false;last+=extent;}}return true;
}
inline void iterate_fp32(const at::Tensor& input,const at::Tensor& other,const at::Tensor& out,uint32_t op,float scalar=0,float alpha=1,const at::Tensor& auxiliary={}) {
 TORCH_CHECK(can_iterate_fp32(input)&&can_iterate_fp32(other)&&can_iterate_fp32(out)&&input.device()==out.device()&&other.device()==out.device(),"Vulkan general iteration exceeds dtype/device/indexing limits");
 if(!out.numel())return;
 auto a=input.expand(out.sizes()),b=other.expand(out.sizes());TORCH_CHECK(can_iterate_fp32(a)&&can_iterate_fp32(b),"Vulkan broadcast iteration exceeds indexing limits");
 auto rank=out.dim();std::vector<uint32_t> data(std::max<int64_t>(1,4*rank),0);for(int64_t d=0;d<rank;d++){data[d]=u32(out.size(d));data[rank+d]=u32(a.stride(d));data[2*rank+d]=u32(b.stride(d));data[3*rank+d]=u32(out.stride(d));}
 auto host=at::empty({int64_t(data.size())},at::TensorOptions().dtype(at::kInt).device(at::kCPU));std::memcpy(host.data_ptr<int32_t>(),data.data(),data.size()*4);auto metadata=host.to(out.device());auto aux=auxiliary.defined()?auxiliary:at::empty({1},out.options());
 auto flags=uint32_t(a.is_neg())|(uint32_t(b.is_neg())<<1)|(uint32_t(out.is_neg())<<2);
 run("iteration_fp32",{base(a),base(b),base(metadata,at::kInt)},{write_base(out),write_base(aux)},{u32(out.numel()),u32(rank),u32(a.storage_offset()),u32(b.storage_offset()),u32(out.storage_offset()),op,bits(scalar),bits(alpha),flags},(out.numel()+127)/128);
}
inline at::Tensor iterated_unary_fp32(const at::Tensor& x,uint32_t op,float scalar=0,float alpha=1){auto out=at::empty(x.sizes(),x.options());iterate_fp32(x,x,out,op,scalar,alpha);return out;}
inline bool can_pack_data(const at::Tensor& src);

// Eligibility for materializing a view with native_pack. Callers that formerly
// supported broader CPU layouts can retain that path without catching dispatch errors.
inline bool can_pack_view(const at::Tensor& src) {
  if(src.device().type()!=c10::DeviceType::PrivateUse1 ||
     (src.scalar_type()!=at::kFloat && src.scalar_type()!=at::kLong && src.scalar_type()!=at::kInt && src.scalar_type()!=at::kComplexFloat && src.scalar_type()!=at::kBool && src.scalar_type()!=at::kByte && src.scalar_type()!=at::kChar && src.scalar_type()!=at::kShort) ||
     src.dim()>8 || src.numel()>UINT32_MAX || src.storage_offset()<0 ||
     uint64_t(src.storage_offset())>UINT32_MAX) return false;
  if(!src.numel())return true;
  const auto& limits=device_limits();
  if(src.storage().nbytes()>limits.maxStorageBufferRange ||
     limits.maxComputeWorkGroupInvocations<128 || limits.maxComputeWorkGroupSize[0]<128 ||
     limits.maxPushConstantsSize<72 || limits.maxPerStageDescriptorStorageBuffers<2 ||
     limits.maxDescriptorSetStorageBuffers<2) return false;
  uint64_t last=src.storage_offset();
  for(int64_t d=0;d<src.dim();++d) {
    if(src.size(d)>UINT32_MAX || src.stride(d)<0 || uint64_t(src.stride(d))>UINT32_MAX)return false;
    if(src.size(d)>1) {
      const uint64_t extent=uint64_t(src.size(d)-1)*src.stride(d);
      if(extent>UINT32_MAX-last)return false;
      last+=extent;
    }
  }
  return true;
}
inline bool can_pack_data(const at::Tensor& src){return can_pack_view(src)||can_iterate_fp32(src);}
inline at::Tensor packed(const at::Tensor& src,at::ScalarType dtype=at::kFloat) {
  TORCH_CHECK(dtype==at::kFloat || dtype==at::kLong || dtype==at::kInt || dtype==at::kComplexFloat || dtype==at::kBool || dtype==at::kByte || dtype==at::kChar || dtype==at::kShort,"Vulkan native packing supports FP32/complex64/int32/int64/bool");
  TORCH_CHECK(src.device().type()==c10::DeviceType::PrivateUse1 && src.scalar_type()==dtype,
              "Vulkan native operator tensor device/dtype mismatch");
  if(!src.numel())return at::empty(src.sizes(),src.options());
  if((src.dim()>8||src.is_neg())&&can_iterate_fp32(src))return iterated_unary_fp32(src,0);
  if(src.is_neg()||src.is_conj()){auto out=at::empty(src.sizes(),src.options());out.copy_(src);return out;}
  base(src,dtype);
  if(src.is_contiguous() && src.storage_offset()==0) return src;
  TORCH_CHECK(src.dim()<=8,"Vulkan native packing supports at most 8 dimensions");
  std::vector<uint32_t> p(18,0); p[0]=u32(src.numel());p[1]=u32(src.storage_offset());
  std::fill(p.begin()+2,p.begin()+10,1);
  uint64_t last=src.storage_offset();
  for(int64_t d=0;d<src.dim();++d) {
    p[2+d]=u32(src.size(d));p[10+d]=u32(src.stride(d));
    if(src.size(d))last+=uint64_t(src.size(d)-1)*src.stride(d);
  }
  TORCH_CHECK(last<=UINT32_MAX,"Vulkan native view indexing overflow");
  auto out=at::empty(src.sizes(),src.options());
  if(src.numel())run(src.element_size()==1?"native_pack_bool":src.element_size()==2?"native_pack_short":(dtype==at::kLong||dtype==at::kComplexFloat)?"native_pack_int64":dtype==at::kInt?"native_pack_int32":"native_pack",
      {base(src,dtype)},{base(out,dtype)},p,(src.numel()+(src.element_size()==1?511:src.element_size()==2?255:127))/(src.element_size()==1?512:src.element_size()==2?256:128));
  return out;
}
} // namespace torch_vulkan::ops::native
