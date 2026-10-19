// Original general FP32 batched GEMM for Vulkan PrivateUse1.
// Supports independently transposed/sliced/offset/broadcast-strided rank-3
// tensors and normal PyTorch autograd, including attention Q/K/V gradients.
#include "native_utils.h"
#include "integer_ops.h"
#include <ATen/Functions.h>
#include <torch/library.h>
#include <cstdlib>
#include <cstring>

namespace torch_vulkan::ops {
namespace {
using namespace native;
at::Tensor bmm(const at::Tensor& a,const at::Tensor& b) {
  TORCH_CHECK(a.device()==b.device() && a.dim()==3 && b.dim()==3 &&
              a.size(0)==b.size(0) && a.size(2)==b.size(1),
              "Vulkan bmm: expected matching batch dimensions and contraction size");
  TORCH_CHECK(a.scalar_type()==b.scalar_type(),"Vulkan bmm: dtype mismatch");
  if(a.scalar_type()==at::kComplexFloat){
    // Reuse the general FP32 GEMM implementation for complex64 arithmetic.
    // ATen real/imag views preserve lazy conjugation and arbitrary strides.
    auto ar=at::real(a),ai=at::imag(a),br=at::real(b),bi=at::imag(b);
    return at::complex(at::bmm(ar,br).sub(at::bmm(ai,bi)),at::bmm(ar,bi).add(at::bmm(ai,br)));
  }
  if(integer_type(a.scalar_type()))return integer_matmul(a,b);
  if(a.scalar_type()==at::kFloat&&(a.is_neg()||b.is_neg()))return at::bmm(packed(a),packed(b));
  const int64_t batch=a.size(0),m=a.size(1),k=a.size(2),n=b.size(2);
  if(batch==0 || m==0 || n==0)return at::empty({batch,m,n},a.options());
  // Skip accelerator dispatch when a view or dtype is outside the native
  // capability envelope; keep the established CPU reference as fallback.
  const auto& l=device_limits();
  const bool native_enabled=!(std::getenv("TORCH_VULKAN_DISABLE_NATIVE_BMM") &&
     std::strcmp(std::getenv("TORCH_VULKAN_DISABLE_NATIVE_BMM"),"1")==0);
  const bool can_dispatch=native_enabled && a.scalar_type()==at::kFloat &&
     can_pack_view(a) && can_pack_view(b) &&
     l.maxComputeWorkGroupInvocations>=256 &&
     l.maxComputeWorkGroupSize[0]>=16 && l.maxComputeWorkGroupSize[1]>=16 &&
     l.maxComputeSharedMemorySize>=2048 &&
     uint64_t(batch)<=l.maxComputeWorkGroupCount[2] &&
     (uint64_t(m)+15)/16<=l.maxComputeWorkGroupCount[1] &&
     (uint64_t(n)+15)/16<=l.maxComputeWorkGroupCount[0] &&
     uint64_t(batch)*uint64_t(m)*uint64_t(n)<=UINT32_MAX;
  if(!can_dispatch)return at::bmm(native::cpu_fallback_tensor(a,__func__),native::cpu_fallback_tensor(b,__func__)).to(a.device());

  auto out=at::empty({batch,m,n},a.options());
  if(k==0) {
    // BLAS semantics: zero accumulator for a zero-length contraction.
    out.zero_();
    return out;
  }
  std::vector<uint32_t> push={
      u32(batch),u32(m),u32(k),u32(n),
      u32(a.storage_offset()),u32(b.storage_offset()),
      u32(a.stride(0)),u32(a.stride(1)),u32(a.stride(2)),
      u32(b.stride(0)),u32(b.stride(1)),u32(b.stride(2))};
  auto aa=base(a),bb=base(b),cc=base(out);
  auto& ctx=VulkanContext::instance();
  auto algo=ctx.get_or_create_algorithm("bmm_strided",{aa,bb,cc},
      kp::Workgroup({static_cast<uint32_t>((n+15)/16),
                     static_cast<uint32_t>((m+15)/16),
                     static_cast<uint32_t>(batch)}),push);
  if(ctx.can_enqueue({aa,bb,cc}))
    ctx.enqueue_compute(algo,{aa,bb,cc},{cc});
  else ctx.run_compute(algo,"bmm_strided",{aa,bb},{cc});
  return out;
}
} // namespace
TORCH_LIBRARY_IMPL(aten,PrivateUse1,m) {
  m.impl("bmm",&bmm);
}
} // namespace torch_vulkan::ops
