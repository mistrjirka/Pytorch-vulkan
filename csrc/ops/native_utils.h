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
inline Buffer base(const at::Tensor& t) {
  TORCH_CHECK(t.device().type()==c10::DeviceType::PrivateUse1 && t.scalar_type()==at::kFloat,
              "Vulkan native operator requires FP32 Vulkan tensors");
  return VulkanAllocator::instance().get_kompute_tensor(t.storage().data_ptr().get());
}
inline kp::Workgroup grid(uint64_t groups) {
  auto l=VulkanContext::instance().manager().getDeviceProperties().limits;
  auto x=std::min<uint64_t>(groups,l.maxComputeWorkGroupCount[0]);
  auto y=x ? (groups+x-1)/x : 0;
  TORCH_CHECK(x && y<=l.maxComputeWorkGroupCount[1],"Vulkan dispatch exceeds device limits");
  return kp::Workgroup({static_cast<uint32_t>(x),static_cast<uint32_t>(y),1});
}
inline void run(const std::string& shader,const std::vector<Buffer>& inputs,
                const std::vector<Buffer>& outputs,const std::vector<uint32_t>& push,
                uint64_t groups,uint32_t shared_bytes=0) {
  auto& ctx=VulkanContext::instance();
  auto l=ctx.manager().getDeviceProperties().limits;
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
inline at::Tensor packed(const at::Tensor& src) {
  base(src); // Validate device and dtype even on the no-copy path.
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
  if(src.numel())run("native_pack",{base(src)},{base(out)},p,(src.numel()+127)/128);
  return out;
}
} // namespace torch_vulkan::ops::native
