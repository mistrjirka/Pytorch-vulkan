// Vulkan-native FP32 reductions, initially optimized for single-axis
// contiguous reductions such as nn.Linear's sum-to-bias backward.
#include "../vulkan_context.h"
#include "../vulkan_allocator.h"
#include <ATen/Functions.h>
#include <torch/library.h>
#include <c10/util/Exception.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <numeric>
#include <optional>
#include <vector>

namespace torch_vulkan::ops {
static at::Tensor sum_dim_vulkan(const at::Tensor& self,
                                at::OptionalIntArrayRef dim,
                                bool keepdim,
                                std::optional<at::ScalarType> dtype) {
  const auto requested_dtype=dtype.value_or(self.scalar_type());
  // Keep the original semantics through CPU fallback until a particular
  // layout/dtype is explicitly implemented and tested on GPU.
  auto cpu_fallback=[&](){
    auto on_cpu=self.to(at::kCPU);
    return at::sum(on_cpu,dim,keepdim,dtype).to(self.device());
  };
  const char* opt=std::getenv("TORCH_VULKAN_NATIVE_SUM");
  if ((opt && std::strcmp(opt,"0")==0) ||
      !dim.has_value() || dim->size()!=1 || self.dim()<1 ||
      self.scalar_type()!=at::kFloat || requested_dtype!=at::kFloat ||
      !self.is_contiguous() || self.storage_offset()!=0 ||
      self.numel()==0)
    return cpu_fallback();

  const auto axis0=(*dim)[0];
  const auto rank=self.dim();
  const int64_t axis=axis0<0 ? axis0+rank : axis0;
  TORCH_CHECK(axis>=0 && axis<rank,
              "sum.dim_IntList: dimension out of range (got ",axis0,")");
  uint64_t outer=1, inner=1;
  for(int64_t i=0;i<axis;++i)
    outer*=static_cast<uint64_t>(self.size(i));
  for(int64_t i=axis+1;i<rank;++i)
    inner*=static_cast<uint64_t>(self.size(i));
  const auto reduced=static_cast<uint64_t>(self.size(axis));
  const uint64_t count=outer*inner;
  if (count==0 || outer>UINT32_MAX || inner>UINT32_MAX ||
      reduced>UINT32_MAX || count>65535ull*256ull)
    return cpu_fallback();
  std::vector<int64_t> shape;
  shape.reserve(rank);
  for(int64_t i=0;i<rank;++i) {
    if (i==axis) {
      if (keepdim)shape.push_back(1);
    } else shape.push_back(self.size(i));
  }
  auto result=at::empty(shape,self.options());
  auto& alloc=VulkanAllocator::instance();
  auto src=alloc.get_kompute_tensor(self.data_ptr());
  auto dst=alloc.get_kompute_tensor(result.data_ptr());
  auto& ctx=VulkanContext::instance();
  const auto algo=ctx.get_or_create_algorithm("sum_single_axis",
      {src,dst},kp::Workgroup({static_cast<uint32_t>((count+255)/256),1,1}),
      {static_cast<uint32_t>(outer),static_cast<uint32_t>(reduced),
       static_cast<uint32_t>(inner),static_cast<uint32_t>(count)});
  if (ctx.can_enqueue({src,dst}))
    ctx.enqueue_compute(algo,{src,dst},{dst});
  else
    ctx.run_compute(algo,"sum_single_axis",{src},{dst});
  return result;
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("sum.dim_IntList",&sum_dim_vulkan);
}
} // namespace torch_vulkan::ops
