// GPU-native FP32 LayerNorm backward. Removes the CPU fallback that otherwise
// stages dY/X/mean/rstd/gamma on every autograd step.
#include "native_utils.h"
#include "../vulkan_allocator.h"
#include <ATen/Functions.h>
#include <torch/library.h>
#include <c10/util/Exception.h>
#include <cstdint>
#include <vector>
#include <array>
namespace torch_vulkan::ops {
using TensorBuffer = std::shared_ptr<kp::Tensor>;
static at::Tensor contiguous_for_native_layer_norm(const at::Tensor& src) {
  if (src.is_contiguous() && src.storage_offset() == 0) return src;
  // Pack readable views on GPU without introducing a host boundary.
  if(native::can_pack_view(src))return native::packed(src);
  return native::cpu_fallback_tensor(src,__func__).contiguous().to(src.device());
}

static TensorBuffer buffer(const at::Tensor& t) {
  TORCH_CHECK(t.device().type() == c10::DeviceType::PrivateUse1 &&
              t.scalar_type() == at::kFloat && t.is_contiguous() &&
              t.storage_offset() == 0,
              "Vulkan layer_norm_backward requires contiguous FP32 Vulkan tensors");
  return VulkanAllocator::instance().get_kompute_tensor(const_cast<void*>(t.const_data_ptr()));
}
static void dispatch(const std::string& shader,
                     const std::vector<TensorBuffer>& inputs,
                     const std::vector<TensorBuffer>& outputs,
                     const std::vector<uint32_t>& constants,
                     uint32_t workgroups) {
  auto& ctx=VulkanContext::instance();
  auto& mgr=ctx.manager();
  std::vector<TensorBuffer> bindings(inputs);
  bindings.insert(bindings.end(),outputs.begin(),outputs.end());
  auto algorithm=ctx.get_or_create_algorithm(
      shader,bindings,kp::Workgroup({workgroups,1,1}),constants);
  // run_compute tracks dependencies and owns queued bindings until completion.
  ctx.run_compute(algorithm,shader,inputs,outputs);
}
static std::tuple<at::Tensor,at::Tensor,at::Tensor> native_layer_norm_backward(
    const at::Tensor& grad_out,
    const at::Tensor& input,
    c10::SymIntArrayRef normalized_shape,
    const at::Tensor& mean,
    const at::Tensor& rstd,
    const std::optional<at::Tensor>& weight,
    const std::optional<at::Tensor>& bias,
    std::array<bool,3> output_mask) {
  TORCH_CHECK(grad_out.sizes() == input.sizes(),
              "layer_norm_backward: grad_output must have input shape");
  TORCH_CHECK(!normalized_shape.empty() &&
              normalized_shape.size() <= static_cast<size_t>(input.dim()),
              "layer_norm_backward: invalid normalized shape");
  int64_t k=1;
  for (size_t i=0;i<normalized_shape.size();++i) {
    const auto extent=normalized_shape[i].expect_int();
    TORCH_CHECK(input.size(input.dim()-normalized_shape.size()+i) == extent,
                "layer_norm_backward: shape mismatch");
    k*=extent;
  }
  TORCH_CHECK(k>0 && input.numel()>0 && input.numel()%k==0,
              "layer_norm_backward: only nonempty FP32 tensors supported");
  const auto rows=input.numel()/k;
  TORCH_CHECK(rows <= 65535 && k <= 65535,
              "layer_norm_backward: dispatch dimension too large");
  TORCH_CHECK(mean.numel()==rows && rstd.numel()==rows,
              "layer_norm_backward: saved mean/rstd size mismatch");
  auto go_c=contiguous_for_native_layer_norm(grad_out);
  auto in_c=contiguous_for_native_layer_norm(input);
  auto mean_c=contiguous_for_native_layer_norm(mean);
  auto rstd_c=contiguous_for_native_layer_norm(rstd);
  auto go_buf=buffer(go_c), in_buf=buffer(in_c);
  auto mean_buf=buffer(mean_c),rstd_buf=buffer(rstd_c);
  const std::vector<TensorBuffer> inputs={in_buf,go_buf,mean_buf,rstd_buf};
  const std::vector<uint32_t> constants={static_cast<uint32_t>(rows),static_cast<uint32_t>(k)};
  at::Tensor dx, dw, db;
  if (output_mask[0]) {
    TensorBuffer gamma;
    at::Tensor gamma_c; // Keep any packed allocation alive until enqueue.
    if (weight && weight->defined()) {
      TORCH_CHECK(weight->numel()==k,"layer_norm_backward: weight size mismatch");
      gamma_c=contiguous_for_native_layer_norm(*weight);
      gamma=buffer(gamma_c);
    } else gamma=VulkanAllocator::instance().constant_float_buffer(k,1.0f);
    dx=at::empty(input.sizes(),input.options());
    auto args=inputs;
    args.push_back(gamma);
    dispatch("layer_norm_backward_input",args,{buffer(dx)},constants,
             static_cast<uint32_t>(rows));
  }
  if (output_mask[1] || output_mask[2]) {
    // Both outputs share the same expensive reduction, so compute together.
    // Drop the unused result if the autograd engine didn't request it.
    auto out_w=at::empty({k},input.options());
    auto out_b=at::empty({k},input.options());
    dispatch("layer_norm_backward_param",inputs,{buffer(out_w),buffer(out_b)},
             constants,static_cast<uint32_t>((k+255)/256));
    at::DimVector original_shape;
    for (auto extent : normalized_shape) original_shape.push_back(extent.expect_int());
    if (output_mask[1]) dw=out_w.view(original_shape);
    if (output_mask[2]) db=out_b.view(original_shape);
  }
  return {dx,dw,db};
}
TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("native_layer_norm_backward", &native_layer_norm_backward);
}
} // namespace torch_vulkan::ops
