// GPU-native row-wise FP32 log-softmax forward/backward. Cross-entropy
// otherwise invokes CPU fallback and repeatedly copies Vulkan tensors.
#include "../vulkan_context.h"
#include "../vulkan_allocator.h"
#include <ATen/Functions.h>
#include <torch/library.h>
#include <c10/util/Exception.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace torch_vulkan::ops {
static bool supports_rowwise(const at::Tensor& t, int64_t dim) {
  if (t.dim()<1 || t.scalar_type()!=at::kFloat ||
      !t.is_contiguous() || t.storage_offset()!=0 ||
      t.numel()==0)
    return false;
  const int64_t axis=dim<0 ? dim+t.dim():dim;
  return axis==t.dim()-1 && t.size(-1)>0 &&
         t.size(-1)<=UINT32_MAX &&
         t.numel()/t.size(-1)<=65535;
}

static bool native_enabled(int64_t elements) {
  // CPU fallback is faster for tiny logits such as batch 128 x classes 10.
  // The native Vulkan shader wins on transformer-sized last-axis reductions.
  // Allow 0/1 to force either for controlled benchmarking.
  const auto* env=std::getenv("TORCH_VULKAN_NATIVE_LOGSOFTMAX");
  if (env && std::strcmp(env,"0")==0) return false;
  if (env && std::strcmp(env,"1")==0) return true;
  return elements >= 32768;
}
static at::Tensor log_softmax_native(const at::Tensor& input,
                                    int64_t dim, bool half_to_float) {
  if (!native_enabled(input.numel()) || !supports_rowwise(input,dim) || half_to_float)
    return at::_log_softmax(input.to(at::kCPU),dim,half_to_float).to(input.device());
  const auto rows=static_cast<uint32_t>(input.numel()/input.size(-1));
  const auto cols=static_cast<uint32_t>(input.size(-1));
  auto result=at::empty(input.sizes(),input.options());
  auto& alloc=VulkanAllocator::instance();
  auto src=alloc.get_kompute_tensor(input.data_ptr());
  auto dst=alloc.get_kompute_tensor(result.data_ptr());
  auto& ctx=VulkanContext::instance();
  auto algo=ctx.get_or_create_algorithm("log_softmax",
      {src,dst},kp::Workgroup({rows,1,1}),{rows,cols});
  if (ctx.can_enqueue({src,dst}))
    ctx.enqueue_compute(algo,{src,dst},{dst});
  else
    ctx.run_compute(algo,"log_softmax",{src},{dst});
  return result;
}

static at::Tensor log_softmax_backward_native(
    const at::Tensor& grad_output,
    const at::Tensor& output, int64_t dim, at::ScalarType input_dtype) {
  TORCH_CHECK(grad_output.sizes()==output.sizes(),
              "log_softmax_backward_data: gradient/output shape mismatch");
  if (!native_enabled(grad_output.numel()) || !supports_rowwise(grad_output,dim) ||
      !supports_rowwise(output,dim) || input_dtype!=at::kFloat) {
    auto grad_cpu=grad_output.to(at::kCPU);
    auto output_cpu=output.to(at::kCPU);
    return at::_log_softmax_backward_data(
        grad_cpu,output_cpu,dim,input_dtype).to(output.device());
  }
  const auto rows=static_cast<uint32_t>(output.numel()/output.size(-1));
  const auto cols=static_cast<uint32_t>(output.size(-1));
  auto result=at::empty(output.sizes(),output.options());
  auto& alloc=VulkanAllocator::instance();
  auto grad=alloc.get_kompute_tensor(grad_output.data_ptr());
  auto logp=alloc.get_kompute_tensor(output.data_ptr());
  auto dst=alloc.get_kompute_tensor(result.data_ptr());
  auto& ctx=VulkanContext::instance();
  auto algo=ctx.get_or_create_algorithm("log_softmax_backward",
      {grad,logp,dst},kp::Workgroup({rows,1,1}),{rows,cols});
  if (ctx.can_enqueue({grad,logp,dst}))
    ctx.enqueue_compute(algo,{grad,logp,dst},{dst});
  else
    ctx.run_compute(algo,"log_softmax_backward",{grad,logp},{dst});
  return result;
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("_log_softmax",&log_softmax_native);
  m.impl("_log_softmax_backward_data",&log_softmax_backward_native);
}
} // namespace torch_vulkan::ops
