#pragma once
#include "../vulkan_context.h"
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
using Buffer = std::shared_ptr<kp::Tensor>;
inline uint32_t u32(int64_t v) {
  TORCH_CHECK(v>=0 && v<=UINT32_MAX,"Vulkan native operator indexing exceeds uint32");
  return static_cast<uint32_t>(v);
}
inline uint32_t bits(float v) { uint32_t r; std::memcpy(&r,&v,4); return r; }
inline Buffer base(const at::Tensor& t,at::ScalarType dtype=at::kFloat) {
  TORCH_CHECK(t.device().type()==c10::DeviceType::PrivateUse1 && t.scalar_type()==dtype,
              "Vulkan native operator tensor device/dtype mismatch");
  return VulkanAllocator::instance().get_kompute_tensor(t.storage().data_ptr().get());
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
// Eligibility for materializing a view with native_pack. Callers that formerly
// supported broader CPU layouts can retain that path without catching dispatch errors.
inline bool can_pack_view(const at::Tensor& src) {
  if(src.device().type()!=c10::DeviceType::PrivateUse1 ||
     (src.scalar_type()!=at::kFloat && src.scalar_type()!=at::kLong && src.scalar_type()!=at::kBool) ||
     src.dim()>8 || src.numel()>UINT32_MAX || src.storage_offset()<0 ||
     uint64_t(src.storage_offset())>UINT32_MAX) return false;
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
inline at::Tensor packed(const at::Tensor& src,at::ScalarType dtype=at::kFloat) {
  TORCH_CHECK(dtype==at::kFloat || dtype==at::kLong || dtype==at::kBool,"Vulkan native packing supports FP32/int64/bool");
  base(src,dtype); // Validate device and dtype even on the no-copy path.
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
  if(src.numel())run(dtype==at::kBool?"native_pack_bool":dtype==at::kLong?"native_pack_int64":"native_pack",
      {base(src,dtype)},{base(out,dtype)},p,(src.numel()+(dtype==at::kBool?511:127))/(dtype==at::kBool?512:128));
  return out;
}
} // namespace torch_vulkan::ops::native
