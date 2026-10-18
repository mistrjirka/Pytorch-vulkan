// Small, correctness-first FP32 NCHW convolutions on real Vulkan compute.
// Direct shaders handle forward, dInput, dWeight, and dBias. Not yet optimized.
#include "../vulkan_context.h"
#include "../vulkan_allocator.h"

#include <ATen/Functions.h>
#include <c10/util/Exception.h>
#include <torch/library.h>
#include <cstdint>
#include <limits>
#include <vector>

namespace torch_vulkan::ops {

using TensorBuffer = std::shared_ptr<kp::Tensor>;

static TensorBuffer buffer(const at::Tensor& t) {
  TORCH_CHECK(t.device().type() == c10::DeviceType::PrivateUse1,
              "Vulkan convolution requires Vulkan tensors");
  TORCH_CHECK(t.is_contiguous() && t.storage_offset() == 0,
              "Vulkan convolution currently requires contiguous, unoffset tensors");
  TORCH_CHECK(t.scalar_type() == at::kFloat, "Vulkan convolution supports FP32 only");
  return VulkanAllocator::instance().get_kompute_tensor(t.data_ptr());
}

struct ConvParams {
  uint32_t n, cin, cout, h, w, kh, kw, oh, ow;
  uint32_t sh, sw, ph, pw, dh, dw, groups;

  std::vector<uint32_t> as_push_constants() const {
    return {n,cin,cout,h,w,kh,kw,oh,ow,sh,sw,ph,pw,dh,dw,groups};
  }
};

static uint32_t checked_uint(int64_t x, const char* name) {
  TORCH_CHECK(x >= 0 && x <= std::numeric_limits<uint32_t>::max(),
              "Vulkan convolution: invalid ", name, "=", x);
  return static_cast<uint32_t>(x);
}

static ConvParams convolution_params(const at::Tensor& input,
                                     const at::Tensor& weight,
                                     c10::SymIntArrayRef stride,
                                     c10::SymIntArrayRef padding,
                                     c10::SymIntArrayRef dilation,
                                     bool transposed,
                                     c10::SymIntArrayRef output_padding,
                                     c10::SymInt groups) {
  TORCH_CHECK(!transposed, "Vulkan ConvTranspose is not implemented yet");
  TORCH_CHECK(input.dim() == 4 && weight.dim() == 4,
              "Vulkan convolution supports NCHW tensors only");
  TORCH_CHECK(stride.size() == 2 && padding.size() == 2 &&
              dilation.size() == 2 && output_padding.size() == 2,
              "Vulkan convolution requires two spatial dimensions");
  TORCH_CHECK(output_padding[0].expect_int() == 0 &&
              output_padding[1].expect_int() == 0,
              "Vulkan convolution: output_padding requires transposed convolution");
  const int64_t n = input.size(0), cin = input.size(1),
      cout = weight.size(0), h = input.size(2), w = input.size(3),
      kh = weight.size(2), kw = weight.size(3),
      sh = stride[0].expect_int(), sw = stride[1].expect_int(),
      ph = padding[0].expect_int(), pw = padding[1].expect_int(),
      dh = dilation[0].expect_int(), dw = dilation[1].expect_int(),
      group_count = groups.expect_int();
  TORCH_CHECK(n > 0 && cin > 0 && cout > 0 && h > 0 && w > 0 &&
              kh > 0 && kw > 0 && sh > 0 && sw > 0 &&
              dh > 0 && dw > 0 && ph >= 0 && pw >= 0 &&
              group_count > 0 && cin % group_count == 0 &&
              cout % group_count == 0 &&
              weight.size(1) == cin / group_count,
              "Vulkan convolution: invalid shapes or parameters");
  const int64_t oh = (h + 2*ph - dh*(kh-1) - 1) / sh + 1;
  const int64_t ow = (w + 2*pw - dw*(kw-1) - 1) / sw + 1;
  TORCH_CHECK(oh > 0 && ow > 0, "Vulkan convolution: empty output");
  return {checked_uint(n,"N"),checked_uint(cin,"Cin"),checked_uint(cout,"Cout"),
          checked_uint(h,"H"),checked_uint(w,"W"),
          checked_uint(kh,"Kh"),checked_uint(kw,"Kw"),
          checked_uint(oh,"Oh"),checked_uint(ow,"Ow"),
          checked_uint(sh,"stride_h"),checked_uint(sw,"stride_w"),
          checked_uint(ph,"padding_h"),checked_uint(pw,"padding_w"),
          checked_uint(dh,"dilation_h"),checked_uint(dw,"dilation_w"),
          checked_uint(group_count,"groups")};
}

static void run(const std::string& shader,
                const std::vector<TensorBuffer>& inputs,
                const TensorBuffer& output,
                int64_t output_count,
                const ConvParams& shape) {
  TORCH_CHECK(output_count >= 0 && output_count <= 65535LL * 256,
              "Vulkan convolution: dispatch exceeds 1D workgroup limit");
  if (output_count == 0) return;
  std::vector<TensorBuffer> bindings(inputs);
  bindings.push_back(output);
  auto& context = VulkanContext::instance();
  auto& mgr = context.manager();
  auto algo = context.get_or_create_algorithm(shader, bindings,
      kp::Workgroup({static_cast<uint32_t>((output_count+255)/256),1,1}),
      shape.as_push_constants());
  if (context.can_enqueue(bindings)) {
    context.enqueue_compute(algo, bindings, {output});
    return;
  }
  auto seq = mgr.sequence();
  seq->record<kp::OpTensorSyncDevice>(inputs);
  seq->record<kp::OpAlgoDispatch>(algo);
  seq->record<kp::OpTensorSyncLocal>(std::vector<TensorBuffer>{output});
  seq->eval();
}

static at::Tensor convolution_overrideable(
    const at::Tensor& input, const at::Tensor& weight,
    const std::optional<at::Tensor>& bias,
    c10::SymIntArrayRef stride, c10::SymIntArrayRef padding,
    c10::SymIntArrayRef dilation, bool transposed,
    c10::SymIntArrayRef output_padding, c10::SymInt groups) {
  auto p = convolution_params(input, weight, stride, padding, dilation,
                               transposed, output_padding, groups);
  auto x = buffer(input), w = buffer(weight);
  auto& mgr = VulkanContext::instance().manager();
  TensorBuffer b;
  if (bias && bias->defined()) {
    TORCH_CHECK(bias->numel() == p.cout && bias->scalar_type() == at::kFloat,
                "Vulkan convolution: bias shape/dtype mismatch");
    b = buffer(*bias);
  } else {
    b = mgr.tensor(std::vector<float>(p.cout, 0.0f));
  }
  auto result = at::empty({p.n, p.cout, p.oh, p.ow}, input.options());
  run("conv2d_forward", {x,w,b}, buffer(result), result.numel(), p);
  return result;
}

static std::tuple<at::Tensor,at::Tensor,at::Tensor> convolution_backward_overrideable(
    const at::Tensor& grad_output, const at::Tensor& input,
    const at::Tensor& weight,
    c10::SymIntArrayRef stride, c10::SymIntArrayRef padding,
    c10::SymIntArrayRef dilation, bool transposed,
    c10::SymIntArrayRef output_padding, c10::SymInt groups,
    std::array<bool,3> output_mask) {
  auto p = convolution_params(input, weight, stride, padding, dilation,
                               transposed, output_padding, groups);
  TORCH_CHECK(grad_output.dim() == 4 && grad_output.size(0) == p.n &&
              grad_output.size(1) == p.cout &&
              grad_output.size(2) == p.oh && grad_output.size(3) == p.ow,
              "Vulkan convolution backward: output gradient shape mismatch");
  const auto go = buffer(grad_output);
  at::Tensor dx, dw, db;
  if (output_mask[0]) {
    dx = at::empty(input.sizes(), input.options());
    run("conv2d_backward_input", {go,buffer(weight)}, buffer(dx),dx.numel(),p);
  }
  if (output_mask[1]) {
    dw = at::empty(weight.sizes(),weight.options());
    run("conv2d_backward_weight", {buffer(input),go},buffer(dw),dw.numel(),p);
  }
  if (output_mask[2]) {
    db = at::empty({p.cout},weight.options());
    run("conv2d_backward_bias",{go},buffer(db),db.numel(),p);
  }
  return {dx,dw,db};
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("convolution_overrideable", &convolution_overrideable);
  m.impl("convolution_backward_overrideable", &convolution_backward_overrideable);
}

} // namespace torch_vulkan::ops
