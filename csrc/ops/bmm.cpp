// Original general FP32 batched GEMM for Vulkan PrivateUse1.
// Supports independently transposed/sliced/offset/broadcast-strided rank-3
// tensors and normal PyTorch autograd, including attention Q/K/V gradients.
#include "native_utils.h"
#include "integer_ops.h"
#include <ATen/Functions.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/native/Resize.h>
#include <torch/library.h>
#include <cstdlib>
#include <cstring>

namespace torch_vulkan::ops {
namespace {
using namespace native;
at::Tensor bmm_impl(const at::Tensor& a,const at::Tensor& b,at::Tensor* destination=nullptr) {
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

  auto out=destination?*destination:at::empty({batch,m,n},a.options());
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
  auto aa=base(a),bb=base(b),cc=write_base(out);
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
at::Tensor bmm(const at::Tensor& a,const at::Tensor& b) {
  return bmm_impl(a,b);
}
// A general direct-output fast path for eligible FP32 contiguous destinations.
// Shape, alias, cast and noncontiguous restrictions match the existing boxed
// NativeOut behavior; when not eligible the full prior functional+copy path
// remains available, with no model-specific branches or new shader.
at::Tensor& bmm_out(const at::Tensor& a,const at::Tensor& b,at::Tensor& out) {
  TORCH_CHECK(out.device()==a.device() && a.device()==b.device(),
              "out tensor must be on the same device as the inputs");
  at::assert_no_internal_overlap(out);
  at::assert_no_partial_overlap(out,a);
  at::assert_no_partial_overlap(out,b);
  const bool aliases=out.storage().is_alias_of(a.storage()) ||
                     out.storage().is_alias_of(b.storage());
  const bool valid_shapes=a.dim()==3 && b.dim()==3 && a.size(0)==b.size(0) &&
                          a.size(2)==b.size(1);
  const bool right_size=valid_shapes &&
        out.sizes()==at::IntArrayRef({a.size(0),a.size(1),b.size(2)});
  const auto* disable=std::getenv("TORCH_VULKAN_DISABLE_DIRECT_BMM_OUT");
  const bool direct_enabled=!disable || std::strcmp(disable,"1")!=0;
  if(direct_enabled && right_size && !aliases &&
     a.scalar_type()==at::kFloat && b.scalar_type()==at::kFloat &&
     out.scalar_type()==at::kFloat && out.is_contiguous() &&
     out.storage_offset()==0 && !out.is_neg() && !out.is_conj() &&
     can_pack_view(out)) {
    // bmm_impl applies the usual device/workgroup/stride/capability checks.
    // If it cannot dispatch natively, it returns a separate functional result.
    auto value=bmm_impl(a,b,&out);
    if(value.is_same(out))return out;
    TORCH_CHECK(c10::canCast(value.scalar_type(),out.scalar_type()),
                "result type cannot be cast to the desired output type");
    out.copy_(value);
    return out;
  }
  auto value=bmm_impl(a,b);
  TORCH_CHECK(c10::canCast(value.scalar_type(),out.scalar_type()),
              "result type cannot be cast to the desired output type");
  at::native::resize_output(out,value.sizes());
  out.copy_(value);
  return out;
}
} // namespace
TORCH_LIBRARY_IMPL(aten,PrivateUse1,m) {
  m.impl("bmm",&bmm);
  m.impl("bmm.out",&bmm_out);
}
} // namespace torch_vulkan::ops
