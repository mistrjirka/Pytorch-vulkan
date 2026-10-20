// Experimental native GPU AdamW implementation. This op deliberately exposes
// a device-native fused primitive instead of patching torch.optim.AdamW.
// This makes the host/device transfer and submit savings measurable while
// preserving existing torch.optim.AdamW behavior for other tensor devices.
#include "../vulkan_allocator.h"
#include "../vulkan_context.h"
#include <ATen/Functions.h>
#include <torch/library.h>
#include <c10/util/Exception.h>
#include <cmath>
#include <cstring>
#include <limits>
#include <optional>

namespace torch_vulkan::ops {

static std::shared_ptr<kp::TensorT<float>> buffer(const at::Tensor& t) {
  TORCH_CHECK(t.device().type() == c10::DeviceType::PrivateUse1,
              "fused AdamW requires Vulkan tensors");
  TORCH_CHECK(t.scalar_type() == at::kFloat && t.is_contiguous() &&
              t.storage_offset() == 0,
              "fused AdamW requires contiguous unoffset float32 tensors");
  return VulkanAllocator::instance().get_kompute_tensor(t.data_ptr());
}

static void fused_adamw_step_(
    const at::Tensor& param, const at::Tensor& grad,
    const at::Tensor& moment1, const at::Tensor& moment2,
    double lr, double beta1, double beta2, double eps,
    double weight_decay, int64_t step) {
  TORCH_CHECK(param.defined() && grad.defined() && moment1.defined() && moment2.defined(),
              "fused AdamW: all four tensors must be defined");
  TORCH_CHECK(param.sizes() == grad.sizes() &&
              param.sizes() == moment1.sizes() && param.sizes() == moment2.sizes(),
              "fused AdamW: parameter, gradient and moments must have identical shapes");
  TORCH_CHECK(step >= 1, "fused AdamW: step must be positive");
  TORCH_CHECK(lr >= 0.0 && beta1 >= 0.0 && beta1 < 1.0 &&
              beta2 >= 0.0 && beta2 < 1.0 && eps >= 0.0 && weight_decay >= 0.0,
              "fused AdamW: invalid hyperparameters");
  TORCH_CHECK(param.numel() <= 65535ll * 256,
              "fused AdamW: unsupported 1D dispatch size");
  if (param.numel() == 0) return;

  auto p = buffer(param), g = buffer(grad), m = buffer(moment1), v = buffer(moment2);
  std::vector<uint32_t> constants;
  constants.reserve(8);
  constants.push_back(static_cast<uint32_t>(param.numel()));
  for (float value : {static_cast<float>(lr),static_cast<float>(beta1),
       static_cast<float>(beta2),static_cast<float>(eps),
       static_cast<float>(weight_decay),
       static_cast<float>(1.0 - std::pow(beta1,static_cast<double>(step))),
       static_cast<float>(1.0 - std::pow(beta2,static_cast<double>(step)))}) {
    uint32_t bits;
    std::memcpy(&bits,&value,sizeof(value));
    constants.push_back(bits);
  }

  auto& ctx = VulkanContext::instance();
  // A deferred resident graph already orders gradient writes before updates.
  // Flushing here charges backward GPU execution to the optimizer and inserts
  // an unnecessary CPU wait. Eager paths retain their existing boundary.
  const char* force=std::getenv("TORCH_VULKAN_ADAMW_FORCE_SYNC");
  if(!ctx.gpu_resident_enabled() || !ctx.deferred_enabled() ||
      (force && std::strcmp(force,"1")==0))ctx.synchronize();
  auto& mgr = ctx.manager();
  auto algo = ctx.get_or_create_algorithm(
      "adamw_fused", {p,g,m,v},
      kp::Workgroup({static_cast<uint32_t>((param.numel()+255)/256),1,1}),
      constants);
  ctx.run_compute(algo,"adamw_fused",{p,g,m,v},{p,m,v});
}

// Batch independent parameter updates into a single Vulkan command buffer.
// Upload all parameters/moments, dispatch all shaders, then download results.
// This removes one host-side VkFence wait per parameter tensor.
static void fused_adamw_step_batch_impl(
    at::TensorList params, at::TensorList grads,
    at::TensorList moments1, at::TensorList moments2,
    std::optional<at::TensorList> max_moments,
    double lr, double beta1, double beta2, double eps,
    double weight_decay, at::IntArrayRef steps) {
  const auto count = params.size();
  TORCH_CHECK(grads.size() == count && moments1.size() == count &&
              moments2.size() == count && steps.size() == count &&
              (!max_moments || max_moments->size() == count),
              "fused AdamW batch: all parameter/gradient/moment/step lists must match");
  TORCH_CHECK(lr >= 0 && beta1 >= 0 && beta1 < 1 && beta2 >= 0 && beta2 < 1 &&
              eps >= 0 && weight_decay >= 0,
              "fused AdamW batch: invalid hyperparameters");
  if (!count) return;

  auto& ctx = VulkanContext::instance();
  // A deferred resident graph already orders gradient writes before updates.
  // Flushing here charges backward GPU execution to the optimizer and inserts
  // an unnecessary CPU wait. Eager paths retain their existing boundary.
  const char* force=std::getenv("TORCH_VULKAN_ADAMW_FORCE_SYNC");
  if(!ctx.gpu_resident_enabled() || !ctx.deferred_enabled() ||
      (force && std::strcmp(force,"1")==0))ctx.synchronize();
  auto& mgr = ctx.manager();
  // Shared_ptr holders remain alive for the whole command buffer evaluation.
  std::vector<std::vector<std::shared_ptr<kp::Tensor>>> inputs;
  std::vector<std::vector<std::shared_ptr<kp::Tensor>>> outputs;
  std::vector<std::shared_ptr<kp::Algorithm>> algorithms;
  inputs.reserve(count);
  outputs.reserve(count);
  algorithms.reserve(count);
  for (size_t i=0; i<count; ++i) {
    const auto& p = params[i];
    TORCH_CHECK(p.sizes() == grads[i].sizes() &&
                p.sizes() == moments1[i].sizes() &&
                p.sizes() == moments2[i].sizes() &&
                (!max_moments || p.sizes() == (*max_moments)[i].sizes()),
                "fused AdamW batch: incompatible tensor shapes at ", i);
    TORCH_CHECK(steps[i] >= 1, "fused AdamW batch: step must be positive");
    TORCH_CHECK(p.numel() <= 65535ll * 256,
                "fused AdamW batch: tensor too large");
    if (!p.numel()) continue;
    auto pbuf=buffer(p), gbuf=buffer(grads[i]);
    auto mbuf=buffer(moments1[i]), vbuf=buffer(moments2[i]);
    // Validate every AMSGrad maximum buffer even when the parameter is empty.
    auto maximum=max_moments ? buffer((*max_moments)[i]) :
                             std::shared_ptr<kp::TensorT<float>>{};
    std::vector<uint32_t> constants;
    constants.reserve(8);
    constants.push_back(static_cast<uint32_t>(p.numel()));
    for (float value: {static_cast<float>(lr),static_cast<float>(beta1),
         static_cast<float>(beta2),static_cast<float>(eps),
         static_cast<float>(weight_decay),
         static_cast<float>(1.0-std::pow(beta1,static_cast<double>(steps[i]))),
         static_cast<float>(1.0-std::pow(beta2,static_cast<double>(steps[i])))}) {
      uint32_t bits;
      std::memcpy(&bits,&value,sizeof(value));
      constants.push_back(bits);
    }
    std::vector<std::shared_ptr<kp::Tensor>> bindings={pbuf,gbuf,mbuf,vbuf};
    std::vector<std::shared_ptr<kp::Tensor>> writes={pbuf,mbuf,vbuf};
    if(maximum){bindings.push_back(maximum);writes.push_back(maximum);}
    inputs.push_back(bindings);
    outputs.push_back(writes);
    algorithms.push_back(ctx.get_or_create_algorithm(
      max_moments ? "adamw_fused_amsgrad" : "adamw_fused", bindings,
      kp::Workgroup({static_cast<uint32_t>((p.numel()+255)/256),1,1}),constants));
  }
  if (algorithms.empty()) return;
  ctx.run_compute_batch(algorithms,inputs,outputs);
}

// Keep the existing ABI for ordinary AdamW and add a distinct AMSGrad op.
// This prevents silently changing the semantics of existing model training.
static void fused_adamw_step_batch_(
    at::TensorList params, at::TensorList grads,
    at::TensorList moments1, at::TensorList moments2,
    double lr, double beta1, double beta2, double eps,
    double weight_decay, at::IntArrayRef steps) {
  fused_adamw_step_batch_impl(params,grads,moments1,moments2,std::nullopt,
                             lr,beta1,beta2,eps,weight_decay,steps);
}
static void fused_adamw_amsgrad_step_batch_(
    at::TensorList params, at::TensorList grads,
    at::TensorList moments1, at::TensorList moments2,
    at::TensorList max_moments,
    double lr, double beta1, double beta2, double eps,
    double weight_decay, at::IntArrayRef steps) {
  fused_adamw_step_batch_impl(params,grads,moments1,moments2,max_moments,
                             lr,beta1,beta2,eps,weight_decay,steps);
}

TORCH_LIBRARY(torch_vulkan, m) {
  m.def("adamw_step_(Tensor(a!) param, Tensor grad, Tensor(b!) exp_avg, "
        "Tensor(c!) exp_avg_sq, float lr, float beta1, float beta2, "
        "float eps, float weight_decay, int step) -> ()");
  m.def("adamw_step_batch_(Tensor(a!)[] params, Tensor[] grads, "
        "Tensor(b!)[] exp_avgs, Tensor(c!)[] exp_avg_sqs, "
        "float lr, float beta1, float beta2, float eps, "
        "float weight_decay, int[] steps) -> ()");
  m.def("adamw_amsgrad_step_batch_(Tensor(a!)[] params, Tensor[] grads, "
        "Tensor(b!)[] exp_avgs, Tensor(c!)[] exp_avg_sqs, "
        "Tensor(d!)[] max_exp_avg_sqs, float lr, float beta1, "
        "float beta2, float eps, float weight_decay, int[] steps) -> ()");
}

TORCH_LIBRARY_IMPL(torch_vulkan, PrivateUse1, m) {
  m.impl("adamw_step_", &fused_adamw_step_);
  m.impl("adamw_step_batch_", &fused_adamw_step_batch_);
  m.impl("adamw_amsgrad_step_batch_", &fused_adamw_amsgrad_step_batch_);
}

} // namespace torch_vulkan::ops
