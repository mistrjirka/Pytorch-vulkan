#include <ATen/Functions.h>
#include <ATen/ops/from_blob.h>
#include <ATen/NativeFunctions.h>
#include <ATen/InferSize.h>
#include <ATen/ExpandUtils.h>
#include <ATen/TensorUtils.h>
#include <ATen/native/Resize.h>
#include "../vulkan_allocator.h"
#include "../vulkan_engine.h"
#include "../vulkan_context.h"
#include <ATen/core/op_registration/op_registration.h>
#include <ATen/native/CPUFallback.h>
#include <torch/library.h>
#include <cstring>
#include <chrono>
#include <cstdio>
#include <cstdlib>

namespace torch_vulkan {
void resize_vulkan_storage(c10::StorageImpl* storage, size_t bytes);
namespace ops {


// Host mirror of a Vulkan tensor's storage. All current Kompute dispatches
// synchronize their outputs back to the host, so views can be copied with
// CPU's stride-aware kernels without reading stale data. The alias borrows the
// original storage; it must never escape the call that holds `src` alive.
static at::Tensor host_mirror(const at::Tensor& src) {
  if (src.device().type() != c10::DeviceType::PrivateUse1) return src;
  VulkanContext::instance().synchronize();
  VulkanContext::instance().ensure_host_read(src.storage().data_ptr().get());
  if (src.numel() == 0)
    return at::empty(src.sizes(), src.options().device(at::kCPU));
  TORCH_CHECK(src.const_data_ptr() != nullptr, "vulkan: unmapped host mirror");
  return at::from_blob(const_cast<void*>(src.const_data_ptr()),
      src.sizes(), src.strides(), [](void*) {},
      at::TensorOptions().dtype(src.scalar_type()).device(at::kCPU));
}

static at::Tensor contiguous_for_kernel(const at::Tensor& src) {
  // A view with nonzero offset is NOT the base Kompute allocation, and a
  // strided view must not be consumed as a flat SPIR-V buffer. Materialize
  // through the host mirror until native strided kernels exist.
  if (src.device().type() != c10::DeviceType::PrivateUse1)
    return src.contiguous();
  if (src.is_contiguous() && src.storage_offset() == 0)
    return src;
  return host_mirror(src).contiguous().to(src.device());
}

// Helper: get existing Kompute tensor from a Vulkan PyTorch tensor (no copy!)
static std::shared_ptr<kp::TensorT<float>> get_kp_tensor(const at::Tensor& t) {
  auto& alloc = VulkanAllocator::instance();
  return alloc.get_kompute_tensor(t.data_ptr());
}

// Helper: sync a Vulkan tensor's host data to device
static void sync_to_device(std::shared_ptr<kp::TensorT<float>> t) {
  auto& ctx = VulkanContext::instance();
  auto& mgr = ctx.manager();
  ctx.synchronize();
  auto seq = mgr.sequence();
  seq->record<kp::OpTensorSyncDevice>({t});
  seq->eval();
}

// Helper: sync a Vulkan tensor's device data back to host
static void sync_to_local(std::shared_ptr<kp::TensorT<float>> t) {
  auto& ctx = VulkanContext::instance();
  auto& mgr = ctx.manager();
  ctx.synchronize();
  auto seq = mgr.sequence();
  seq->record<kp::OpTensorSyncLocal>({t});
  seq->eval();
}

// Helper: run a 1D elementwise shader (3 buffers: a, b, out)
// Uses persistent Kompute tensors from allocator — no per-call allocation.
// Phase 4: Algorithm cache eliminates per-call VkPipeline + descriptor set
// creation when the same tensor buffers are dispatched repeatedly.
static at::Tensor elementwise_binary(
    const at::Tensor& self, const at::Tensor& other,
    const std::string& shader_name,
    std::vector<uint32_t> push_constants) {

  TORCH_CHECK(self.scalar_type() == at::kFloat && other.scalar_type() == at::kFloat,
              shader_name, ": float32 inputs required");
  auto self_c = contiguous_for_kernel(self);
  auto other_c = contiguous_for_kernel(other);
  int64_t n = self_c.numel();
  TORCH_CHECK(self_c.sizes() == other_c.sizes(), shader_name,
              ": broadcasting requires matching shapes or an explicit expanded operand");

  auto& ctx = VulkanContext::instance();
  auto& mgr = ctx.manager();

  // Get existing Kompute tensors if on Vulkan, else create temp
  bool a_on_vk = self_c.device().type() == c10::DeviceType::PrivateUse1;
  bool b_on_vk = other_c.device().type() == c10::DeviceType::PrivateUse1;

  auto tensor_a = a_on_vk ? get_kp_tensor(self_c)
      : mgr.tensor(std::vector<float>(self_c.data_ptr<float>(), self_c.data_ptr<float>() + n));
  auto tensor_b = b_on_vk ? get_kp_tensor(other_c)
      : mgr.tensor(std::vector<float>(other_c.data_ptr<float>(), other_c.data_ptr<float>() + n));

  auto output = at::empty(self.sizes(), self.options().device(c10::Device(c10::DeviceType::PrivateUse1, 0)));
  auto tensor_c = get_kp_tensor(output);

  uint32_t wg = (n + 255) / 256;

  // Phase 4: use cached algorithm -- saves ~3-5ms per call when the same
  // (shader, tensor_a, tensor_b, tensor_c) combo is dispatched again.
  auto algorithm = ctx.get_or_create_algorithm(
      shader_name,
      {tensor_a, tensor_b, tensor_c},
      kp::Workgroup({wg, 1, 1}),
      push_constants);

  if (ctx.can_enqueue({tensor_a,tensor_b,tensor_c})) {
    ctx.enqueue_compute(algorithm, {tensor_a,tensor_b,tensor_c}, {tensor_c});
    return output;
  }
  ctx.run_compute(algorithm, shader_name,
      {tensor_a,tensor_b},{tensor_c});

  return output;
}

// Helper: run a 1D elementwise shader (2 buffers: in, out)
// Uses persistent Kompute tensors — no per-call allocation.
// Phase 4: Algorithm cache.
static at::Tensor elementwise_unary(
    const at::Tensor& self, const std::string& shader_name,
    std::vector<uint32_t> push_constants) {

  TORCH_CHECK(self.scalar_type() == at::kFloat,
              shader_name, ": float32 input required");
  auto self_c = contiguous_for_kernel(self);
  int64_t n = self_c.numel();

  auto& ctx = VulkanContext::instance();
  auto& mgr = ctx.manager();

  bool on_vk = self_c.device().type() == c10::DeviceType::PrivateUse1;
  auto tensor_in = on_vk ? get_kp_tensor(self_c)
      : mgr.tensor(std::vector<float>(self_c.data_ptr<float>(), self_c.data_ptr<float>() + n));

  auto output = at::empty(self.sizes(), self.options().device(c10::Device(c10::DeviceType::PrivateUse1, 0)));
  auto tensor_out = get_kp_tensor(output);

  uint32_t wg = (n + 255) / 256;

  // Phase 4: cached algorithm
  auto algorithm = ctx.get_or_create_algorithm(
      shader_name,
      {tensor_in, tensor_out},
      kp::Workgroup({wg, 1, 1}),
      push_constants);

  if (ctx.can_enqueue({tensor_in,tensor_out})) {
    ctx.enqueue_compute(algorithm, {tensor_in,tensor_out}, {tensor_out});
    return output;
  }
  ctx.run_compute(algorithm, shader_name,{tensor_in},{tensor_out});

  return output;
}

// ---- Operators ----
// Forward declarations
static at::Tensor mul_scalar(const at::Tensor& self, const at::Scalar& other);


// aten::mm via raw VulkanEngine — pooled buffers, zero per-call allocation
static at::Tensor mm_raw(const at::Tensor& self, const at::Tensor& mat2) {
  int64_t M = self.size(0), K = self.size(1), N = mat2.size(1);

  auto& engine = VulkanEngine::instance();
  auto& pipeline = engine.getOrCreatePipeline("matmul_tiled", 3, 3 * sizeof(uint32_t));

  auto self_c = contiguous_for_kernel(self);
  auto mat2_c = contiguous_for_kernel(mat2);

  VkDeviceSize a_size = M * K * sizeof(float);
  VkDeviceSize b_size = K * N * sizeof(float);
  VkDeviceSize c_size = M * N * sizeof(float);

  // Acquire from pool (no vkAllocateMemory after warmup)
  auto a_buf = engine.acquireBuffer(a_size);
  auto b_buf = engine.acquireBuffer(b_size);
  auto c_buf = engine.acquireBuffer(c_size);

  // memcpy only (host-coherent unified memory)
  std::memcpy(a_buf.mapped, self_c.data_ptr<float>(), a_size);
  std::memcpy(b_buf.mapped, mat2_c.data_ptr<float>(), b_size);

  uint32_t push[3] = {(uint32_t)M, (uint32_t)K, (uint32_t)N};
  uint32_t wg_x = (N + 15) / 16;
  uint32_t wg_y = (M + 15) / 16;

  engine.dispatch(pipeline,
    {a_buf.buffer, b_buf.buffer, c_buf.buffer},
    {a_size, b_size, c_size},
    push, sizeof(push),
    wg_x, wg_y, 1);

  auto output = at::empty({M, N}, self.options().device(c10::Device(c10::DeviceType::PrivateUse1, 0)));
  std::memcpy(output.data_ptr<float>(), c_buf.mapped, c_size);

  // Return to pool (not destroyed, reused next call)
  engine.releaseBuffer(a_buf);
  engine.releaseBuffer(b_buf);
  engine.releaseBuffer(c_buf);

  return output;
}

// aten::mm — general matmul via matmul_tiled.spv
// Phase 4: Algorithm cache — the same weight matrix gets the same pipeline.
at::Tensor mm(const at::Tensor& self, const at::Tensor& mat2) {
  TORCH_CHECK(self.dim() == 2 && mat2.dim() == 2, "mm: inputs must be 2D");
  TORCH_CHECK(self.size(1) == mat2.size(0), "mm: dimension mismatch");
  TORCH_CHECK(self.dtype() == at::kFloat, "mm: only float32 for now");

  int64_t M = self.size(0), K = self.size(1), N = mat2.size(1);
  TORCH_CHECK(M >= 0 && K >= 0 && N >= 0 && M <= UINT32_MAX &&
              K <= UINT32_MAX && N <= UINT32_MAX,
              "mm: unsupported matrix dimensions");

  auto& ctx = VulkanContext::instance();
  auto& mgr = ctx.manager();
  // A stride-only transpose shares its original Vulkan allocation. Matrix
  // multiplication consumes that layout directly rather than first copying
  // an entire weight matrix on the host.
  auto self_c = self.device().type() == c10::DeviceType::PrivateUse1 &&
                  self.storage_offset() == 0
                    ? self : contiguous_for_kernel(self);
  auto mat2_c = mat2.device().type() == c10::DeviceType::PrivateUse1 &&
                  mat2.storage_offset() == 0
                    ? mat2 : contiguous_for_kernel(mat2);

  // Get or create Kompute tensors
  bool a_on_vk = self_c.device().type() == c10::DeviceType::PrivateUse1;
  bool b_on_vk = mat2_c.device().type() == c10::DeviceType::PrivateUse1;

  auto tensor_a = a_on_vk ? get_kp_tensor(self_c)
      : mgr.tensor(std::vector<float>(self_c.data_ptr<float>(), self_c.data_ptr<float>() + M * K));
  auto tensor_b = b_on_vk ? get_kp_tensor(mat2_c)
      : mgr.tensor(std::vector<float>(mat2_c.data_ptr<float>(), mat2_c.data_ptr<float>() + K * N));

  auto output = at::empty({M, N}, self.options().device(c10::Device(c10::DeviceType::PrivateUse1, 0)));
  auto tensor_c = get_kp_tensor(output);

  uint32_t m_val = static_cast<uint32_t>(M);
  uint32_t k_val = static_cast<uint32_t>(K);
  uint32_t n_val = static_cast<uint32_t>(N);
  // Prefer 64x64 register-tiling for large output tiles; 16x16 is faster
  // for narrow classifiers and small batches. Tuned on RTX 2080 Ti + V100.
  // Force a kernel for reproducible A/B profiling by setting the env var.
  const auto* kernel = std::getenv("TORCH_VULKAN_MM_KERNEL");
  const bool forced_register = kernel && std::strcmp(kernel,"register")==0;
  const bool forced_tiled = kernel && std::strcmp(kernel,"tiled")==0;
  const bool use_register_kernel = forced_register ||
      (!forced_tiled && M>=256 && N>=256 && K>=64);
  const uint32_t tile_size = use_register_kernel ? 64 : 16;
  uint32_t wg_x = (N + tile_size - 1) / tile_size;
  uint32_t wg_y = (M + tile_size - 1) / tile_size;

  // Phase 4: cached algorithm.
  // For transformer inference, the weight tensor_b is the same every call
  // (same data pointer). The activation tensor_a may vary, but if the model
  // reuses the same buffer size, the allocator returns the same pointer.
  auto algorithm = ctx.get_or_create_algorithm(
      use_register_kernel ? "matmul_strided_reg" : "matmul_strided",
      {tensor_a, tensor_b, tensor_c},
      kp::Workgroup({wg_x, wg_y, 1}),
      std::vector<uint32_t>{m_val, k_val, n_val,
                            static_cast<uint32_t>(self_c.stride(0)),
                            static_cast<uint32_t>(self_c.stride(1)),
                            static_cast<uint32_t>(mat2_c.stride(0)),
                            static_cast<uint32_t>(mat2_c.stride(1))});

  if (ctx.can_enqueue({tensor_a,tensor_b,tensor_c})) {
    ctx.enqueue_compute(algorithm, {tensor_a,tensor_b,tensor_c}, {tensor_c});
    return output;
  }
  ctx.run_compute(algorithm,
      use_register_kernel ? "mm_register" : "mm",
      {tensor_a,tensor_b},{tensor_c});

  return output;
}

// Expand broadcast dimensions using PyTorch's standard NumPy-style rules.
// Currently strided expanded views are materialized via the host mirror before
// compute; this preserves correctness but is slower than a stride-aware shader.
static std::pair<at::Tensor, at::Tensor> broadcast_binary_operands(
    const at::Tensor& a, const at::Tensor& b) {
  const auto shape = at::infer_size(a.sizes(), b.sizes());
  const auto ea = a.sizes().equals(shape) ? a : a.expand(shape);
  const auto eb = b.sizes().equals(shape) ? b : b.expand(shape);
  return {contiguous_for_kernel(ea), contiguous_for_kernel(eb)};
}

// aten::add.Tensor — alpha scaling with arbitrary compatible broadcasting.
at::Tensor add_tensor(const at::Tensor& self, const at::Tensor& other,
                      const at::Scalar& alpha) {
  auto [a, b] = broadcast_binary_operands(self, other);
  TORCH_CHECK(a.numel() <= UINT32_MAX, "Vulkan add: tensor too large");
  const uint32_t n = static_cast<uint32_t>(a.numel());
  float alpha_f = alpha.toFloat();
  uint32_t alpha_bits;
  std::memcpy(&alpha_bits, &alpha_f, sizeof(alpha_bits));
  return elementwise_binary(a, b, "add", {n, alpha_bits});
}

// aten::mul.Tensor — elementwise multiplication with arbitrary broadcasting.
at::Tensor mul_tensor(const at::Tensor& self, const at::Tensor& other) {
  auto [a, b] = broadcast_binary_operands(self, other);
  TORCH_CHECK(a.numel() <= UINT32_MAX, "Vulkan mul: tensor too large");
  const uint32_t n = static_cast<uint32_t>(a.numel());
  return elementwise_binary(a, b, "mul", {n});
}

// aten::relu — rectified linear
at::Tensor relu(const at::Tensor& self) {
  uint32_t n = static_cast<uint32_t>(self.numel());
  return elementwise_unary(self, "relu", {n});
}

// Autograd ReLU gradient: threshold_backward(grad_output, forward_output, 0).
// Unlike the generic CPU fallback, dispatches its derivative shader on Vulkan.
static at::Tensor threshold_backward(const at::Tensor& grad,
                                      const at::Tensor& self,
                                      const at::Scalar& threshold) {
  TORCH_CHECK(grad.scalar_type() == at::kFloat && self.scalar_type() == at::kFloat,
              "threshold_backward supports float32 only");
  TORCH_CHECK(grad.sizes() == self.sizes(),
              "threshold_backward requires identical gradient/output shapes");
  const uint32_t n = static_cast<uint32_t>(grad.numel());
  const float value = threshold.toFloat();
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  return elementwise_binary(self, grad, "threshold_backward", {n, bits});
}

// _to_copy — .to(device) implementation
at::Tensor vulkan_to_copy(
    const at::Tensor& self,
    std::optional<at::ScalarType> dtype,
    std::optional<at::Layout> layout,
    std::optional<at::Device> device,
    std::optional<bool> pin_memory,
    bool non_blocking,
    std::optional<at::MemoryFormat> memory_format) {

  auto target_device = device.value_or(self.device());
  auto target_dtype = dtype.value_or(self.scalar_type());

  if (target_device.type() == c10::DeviceType::PrivateUse1) {
    auto self_cpu = host_mirror(self).contiguous();
    auto self_cast = self_cpu.to(target_dtype).contiguous();
    auto result = at::empty(self_cast.sizes(),
        self_cast.options().device(c10::Device(c10::DeviceType::PrivateUse1, 0)));
    std::memcpy(result.data_ptr(), self_cast.data_ptr(),
        self_cast.numel() * self_cast.element_size());
    return result;
  } else if (target_device.type() == c10::DeviceType::CPU) {
    // First copy the *source* dtype. Copying FP32 bytes into an FP64 allocation
    // without conversion used to return uninitialized/wrong data.
    auto src = host_mirror(self).contiguous();
    auto result = at::empty(src.sizes(),
        at::TensorOptions().dtype(src.scalar_type()).device(at::kCPU));
    std::memcpy(result.data_ptr(), src.data_ptr(),
        src.numel() * src.element_size());
    return result.scalar_type() == target_dtype ? result : result.to(target_dtype);
  }
  TORCH_CHECK(false, "_to_copy: unsupported target device");
}

// copy_ — in-place copy
at::Tensor& vulkan_copy_(at::Tensor& self, const at::Tensor& src, bool non_blocking) {
  // A Vulkan allocation is represented by a mapped host mirror. CPU's
  // stride-aware copy correctly handles offsets, permutations and casting.
  // Inputs are synchronized to the GPU by the next shader submission.
  host_mirror(self).copy_(host_mirror(src));
  VulkanAllocator::instance().mark_host_dirty(self.storage().data_ptr().get());
  return self;
}

// _copy_from_and_resize — used by torch.randn(..., device="vulkan") path
at::Tensor vulkan_copy_from_and_resize(const at::Tensor& src, const at::Tensor& dst) {
  TORCH_CHECK(src.scalar_type() == dst.scalar_type(),
      "vulkan _copy_from_and_resize: dtype conversion is not implemented");
  if (dst.device().type() == c10::DeviceType::PrivateUse1) {
    // The CPU fallback uses an initially empty destination; copying without
    // resizing is incorrect and could overflow. Grow the underlying storage
    // via the registered allocator, then assign contiguous tensor metadata.
    auto* impl = dst.unsafeGetTensorImpl();
    impl->set_sizes_contiguous(src.sizes());
    auto storage = dst.storage();
    const auto bytes = static_cast<size_t>(src.numel()) * src.element_size();
    if (bytes > storage.nbytes()) {
      torch_vulkan::resize_vulkan_storage(storage.unsafeGetStorageImpl(), bytes);
    }
  } else {
    dst.resize_(src.sizes());
  }
  return dst.copy_(src);
}

// _copy_from — generic cross-device copy entry point
at::Tensor vulkan_copy_from(const at::Tensor& self, const at::Tensor& dst, bool non_blocking) {
  TORCH_CHECK(self.scalar_type() == dst.scalar_type(),
      "vulkan _copy_from: dtype mismatch (", self.scalar_type(), " vs ", dst.scalar_type(), ")");
  TORCH_CHECK(self.numel() == dst.numel(),
      "vulkan _copy_from: numel mismatch (", self.numel(), " vs ", dst.numel(), ")");
  return dst.copy_(self, non_blocking);
}

// empty.memory_format
at::Tensor empty_memory_format(
    at::IntArrayRef size, std::optional<at::ScalarType> dtype,
    std::optional<at::Layout> layout, std::optional<at::Device> device,
    std::optional<bool> pin_memory, std::optional<at::MemoryFormat> memory_format) {
  auto actual_dtype = dtype.value_or(at::kFloat);
  auto allocator = &VulkanAllocator::instance();
  int64_t nel = 1;
  for (auto s : size) nel *= s;
  size_t nbytes = nel * at::elementSize(actual_dtype);

  auto storage = c10::Storage(c10::Storage::use_byte_size_t(), nbytes,
      allocator->allocate(nbytes), allocator, true);
  auto tensor = at::detail::make_tensor<c10::TensorImpl>(
      std::move(storage),
      c10::DispatchKeySet(c10::DispatchKey::PrivateUse1),
      at::scalarTypeToTypeMeta(actual_dtype));
  tensor.unsafeGetTensorImpl()->set_sizes_contiguous(size);
  return tensor;
}

// aten::_softmax — row-wise softmax via softmax.spv
// Phase 4: cached algorithm for repeated softmax on same-sized buffers
at::Tensor vulkan_softmax(const at::Tensor& self, int64_t dim, bool half_to_float) {
  VulkanContext::instance().synchronize();
  TORCH_CHECK(self.dtype() == at::kFloat, "softmax: only float32");
  auto self_c = contiguous_for_kernel(self);

  // Flatten to 2D: [batch, row_size]
  int64_t row_size = self_c.size(dim);
  int64_t num_rows = self_c.numel() / row_size;

  auto& ctx = VulkanContext::instance();
  auto& mgr = ctx.manager();

  auto output = at::empty(self.sizes(), self.options());
  auto tensor_in = get_kp_tensor(self_c);
  auto tensor_out = get_kp_tensor(output);

  uint32_t nr = static_cast<uint32_t>(num_rows);
  uint32_t rs = static_cast<uint32_t>(row_size);

  // Softmax uses temp tensors (not allocator-backed), so algorithm cache
  // won't hit for different calls. Still use the API for consistency --
  // the cache miss overhead is negligible.
  auto algorithm = ctx.get_or_create_algorithm(
      "softmax",
      {tensor_in, tensor_out},
      kp::Workgroup({nr, 1, 1}),
      std::vector<uint32_t>{nr, rs});

  ctx.synchronize();
  ctx.run_compute(algorithm, "softmax", {tensor_in}, {tensor_out});

  return output;
}

// aten::native_layer_norm — layer normalization via layer_norm.spv
// Phase 4: cached algorithm
std::tuple<at::Tensor, at::Tensor, at::Tensor> vulkan_layer_norm(
    const at::Tensor& input,
    at::IntArrayRef normalized_shape,
    const std::optional<at::Tensor>& weight_opt,
    const std::optional<at::Tensor>& bias_opt,
    double eps) {

  VulkanContext::instance().synchronize();
  TORCH_CHECK(input.dtype() == at::kFloat, "layer_norm: only float32");
  TORCH_CHECK(eps >= 0.0, "layer_norm: epsilon must be nonnegative");
  TORCH_CHECK(!normalized_shape.empty() &&
              normalized_shape.size() <= static_cast<size_t>(input.dim()),
              "layer_norm: invalid normalized_shape");
  auto input_c = contiguous_for_kernel(input);
  int64_t row_size = 1;
  const auto leading = input.dim() - static_cast<int64_t>(normalized_shape.size());
  std::vector<int64_t> stat_shape(input.sizes().begin(), input.sizes().end());
  for (int64_t k = 0; k < static_cast<int64_t>(normalized_shape.size()); ++k) {
    TORCH_CHECK(input.size(leading + k) == normalized_shape[k],
                "layer_norm: normalized_shape does not match input");
    row_size *= normalized_shape[k];
    stat_shape[leading + k] = 1;
  }
  TORCH_CHECK(row_size > 0 && input_c.numel() % row_size == 0,
              "layer_norm: invalid input row dimensions");
  auto num_rows = input_c.numel() / row_size;
  TORCH_CHECK(num_rows <= UINT32_MAX && row_size <= UINT32_MAX,
              "layer_norm: tensor too large for Vulkan dispatch");

  auto& ctx = VulkanContext::instance();
  auto& mgr = ctx.manager();
  const auto n = input_c.numel();
  auto output = at::empty(input.sizes(), input.options());
  auto mean = at::empty(stat_shape, input.options());
  auto rstd = at::empty(stat_shape, input.options());

  // The input can reuse its existing Vulkan allocation; temporary scalar
  // parameters are uploaded through the same Kompute submission.
  auto in_tensor = get_kp_tensor(input_c);
  auto out_tensor = get_kp_tensor(output);
  auto mean_tensor = get_kp_tensor(mean);
  auto rstd_tensor = get_kp_tensor(rstd);
  // Affine parameters already live in the backend allocator. Reuse their
  // VkBuffers instead of allocating/copying temporary buffers on every call.
  std::shared_ptr<kp::TensorT<float>> w_tensor, b_tensor;
  if (weight_opt.has_value() && weight_opt->defined()) {
    const auto w = contiguous_for_kernel(*weight_opt);
    TORCH_CHECK(w.dtype() == at::kFloat && w.numel() == row_size,
                "layer_norm: invalid weight");
    w_tensor = get_kp_tensor(w);
  } else {
    w_tensor = mgr.tensor(std::vector<float>(row_size, 1.0f));
  }
  if (bias_opt.has_value() && bias_opt->defined()) {
    const auto b = contiguous_for_kernel(*bias_opt);
    TORCH_CHECK(b.dtype() == at::kFloat && b.numel() == row_size,
                "layer_norm: invalid bias");
    b_tensor = get_kp_tensor(b);
  } else {
    b_tensor = mgr.tensor(std::vector<float>(row_size, 0.0f));
  }
  const auto nr = static_cast<uint32_t>(num_rows);
  const auto rs = static_cast<uint32_t>(row_size);
  float eps_f = static_cast<float>(eps);
  uint32_t eps_bits;
  std::memcpy(&eps_bits, &eps_f, sizeof(eps_bits));
  auto algorithm = ctx.get_or_create_algorithm(
      "layer_norm", {in_tensor, w_tensor, b_tensor, out_tensor, mean_tensor, rstd_tensor},
      kp::Workgroup({nr, 1, 1}), std::vector<uint32_t>{nr, rs, eps_bits});
  ctx.synchronize();
  ctx.run_compute(algorithm, "layer_norm", {in_tensor,w_tensor,b_tensor}, {out_tensor,mean_tensor,rstd_tensor});
  return std::make_tuple(output, mean, rstd);
}

// aten::gelu — Gaussian Error Linear Unit via gelu.spv
at::Tensor vulkan_gelu(const at::Tensor& self, c10::string_view approximate) {
  TORCH_CHECK(approximate == "none", "Vulkan GELU only supports approximate='none'");
  uint32_t n = static_cast<uint32_t>(self.numel());
  return elementwise_unary(self, "gelu", {n});
}

at::Tensor vulkan_gelu_backward(const at::Tensor& grad,
                                 const at::Tensor& input,
                                 c10::string_view approximate) {
  TORCH_CHECK(approximate == "none", "Vulkan GELU backward only supports approximate='none'");
  uint32_t n = static_cast<uint32_t>(input.numel());
  TORCH_CHECK(grad.sizes() == input.sizes(), "Vulkan GELU backward shape mismatch");
  return elementwise_binary(input, grad, "gelu_backward", {n});
}

// aten::embedding — lookup table via embedding.spv
// Phase 4: cached algorithm
at::Tensor vulkan_embedding(
    const at::Tensor& weight,
    const at::Tensor& indices,
    int64_t padding_idx,
    bool scale_grad_by_freq,
    bool sparse) {

  VulkanContext::instance().synchronize();
  TORCH_CHECK(weight.dtype() == at::kFloat, "embedding: only float32 weight");
  auto weight_c = contiguous_for_kernel(weight);
  auto indices_c = contiguous_for_kernel(indices).to(at::kFloat);  // pass as float buffer

  int64_t num_indices = indices_c.numel();
  int64_t embedding_dim = weight.size(1);
  int64_t total = num_indices * embedding_dim;

  auto& ctx = VulkanContext::instance();
  auto& mgr = ctx.manager();

  auto out_sizes = indices.sizes().vec();
  out_sizes.push_back(embedding_dim);
  auto output = at::empty(out_sizes, weight.options());
  auto tensor_w = get_kp_tensor(weight_c);
  auto tensor_idx = get_kp_tensor(indices_c);
  auto tensor_out = get_kp_tensor(output);

  uint32_t wg = (total + 255) / 256;
  uint32_t ni = static_cast<uint32_t>(num_indices);
  uint32_t ed = static_cast<uint32_t>(embedding_dim);

  auto algorithm = ctx.get_or_create_algorithm(
      "embedding",
      {tensor_w, tensor_idx, tensor_out},
      kp::Workgroup({wg, 1, 1}),
      std::vector<uint32_t>{ni, ed});

  ctx.synchronize();
  ctx.run_compute(algorithm, "embedding", {tensor_w,tensor_idx}, {tensor_out});

  return output;
}


// Single-launch matmul+bias for torch.nn.Linear. A vector bias has no
// costly host materialization or separate Vulkan addition dispatch.
static at::Tensor addmm_bias_vulkan(
    const at::Tensor& bias, const at::Tensor& lhs, const at::Tensor& rhs,
    float beta, float alpha) {
  TORCH_CHECK(lhs.dim()==2 && rhs.dim()==2 && lhs.size(1)==rhs.size(0) &&
              bias.dim()==1 && bias.numel()==rhs.size(1),
              "fused addmm: invalid dimensions");
  TORCH_CHECK(lhs.scalar_type()==at::kFloat &&
              rhs.scalar_type()==at::kFloat &&
              bias.scalar_type()==at::kFloat,
              "fused addmm: FP32 only");
  const int64_t M=lhs.size(0),K=lhs.size(1),N=rhs.size(1);
  TORCH_CHECK(M<=UINT32_MAX && K<=UINT32_MAX && N<=UINT32_MAX,
              "fused addmm: matrix too large");
  const auto a=lhs.storage_offset()==0 ? lhs : contiguous_for_kernel(lhs);
  const auto b=rhs.storage_offset()==0 ? rhs : contiguous_for_kernel(rhs);
  const auto bv=contiguous_for_kernel(bias);
  const auto* choice=std::getenv("TORCH_VULKAN_MM_KERNEL");
  const bool use_reg=(choice && std::strcmp(choice,"register")==0) ||
      ((!choice || std::strcmp(choice,"tiled")!=0) &&
       M>=256 && N>=256 && K>=64);
  const uint32_t tile=use_reg ? 64 : 16;
  auto out=at::empty({M,N},lhs.options());
  auto ta=get_kp_tensor(a),tb=get_kp_tensor(b);
  auto td=get_kp_tensor(bv),tc=get_kp_tensor(out);
  uint32_t alpha_bits,beta_bits;
  std::memcpy(&alpha_bits,&alpha,sizeof(alpha));
  std::memcpy(&beta_bits,&beta,sizeof(beta));
  const std::string shader=use_reg ?
      "matmul_strided_reg_bias" : "matmul_strided_bias";
  auto& ctx=VulkanContext::instance();
  auto algo=ctx.get_or_create_algorithm(shader,{ta,tb,td,tc},
      kp::Workgroup({static_cast<uint32_t>((N+tile-1)/tile),
                     static_cast<uint32_t>((M+tile-1)/tile),1}),
      {static_cast<uint32_t>(M),static_cast<uint32_t>(K),
       static_cast<uint32_t>(N),
       static_cast<uint32_t>(a.stride(0)),static_cast<uint32_t>(a.stride(1)),
       static_cast<uint32_t>(b.stride(0)),static_cast<uint32_t>(b.stride(1)),
       alpha_bits,beta_bits});
  if (ctx.can_enqueue({ta,tb,td,tc}))
    ctx.enqueue_compute(algo,{ta,tb,td,tc},{tc});
  else
    ctx.run_compute(algo,use_reg ? "addmm_register" : "addmm",
                    {ta,tb,td},{tc});
  return out;
}

// aten::addmm — fused bias + matmul: out = beta*self + alpha*(mat1 @ mat2)
at::Tensor vulkan_addmm(
    const at::Tensor& self,
    const at::Tensor& mat1,
    const at::Tensor& mat2,
    const at::Scalar& beta,
    const at::Scalar& alpha) {

  // Set TORCH_VULKAN_FUSE_ADDMM=0 for controlled A/B comparisons.
  const auto* fuse_flag=std::getenv("TORCH_VULKAN_FUSE_ADDMM");
  const bool fuse_bias=!(fuse_flag && std::strcmp(fuse_flag,"0")==0);
  if (fuse_bias && self.dim()==1 && mat1.dim()==2 && mat2.dim()==2 &&
      self.numel()==mat2.size(1) &&
      self.scalar_type()==at::kFloat &&
      mat1.scalar_type()==at::kFloat && mat2.scalar_type()==at::kFloat &&
      self.device().type()==c10::DeviceType::PrivateUse1 &&
      mat1.device().type()==c10::DeviceType::PrivateUse1 &&
      mat2.device().type()==c10::DeviceType::PrivateUse1)
    return addmm_bias_vulkan(self,mat1,mat2,beta.toFloat(),alpha.toFloat());

  auto product = torch_vulkan::ops::mm(mat1, mat2);

  if (alpha.toFloat() != 1.0f) {
    auto alpha_t = at::full({1}, alpha.toFloat(), product.options());
    product = mul_tensor(product, contiguous_for_kernel(alpha_t.expand_as(product)));
  }

  auto bias = self;
  if (self.dim() == 1) {
    bias = contiguous_for_kernel(self.unsqueeze(0).expand_as(product));
  }

  return add_tensor(bias, product, beta);
}


// aten::mul.Scalar — elementwise scalar multiply via scalar_mul.spv
// Phase 4: cached algorithm
at::Tensor mul_scalar(const at::Tensor& self, const at::Scalar& other) {
  TORCH_CHECK(self.scalar_type() == at::kFloat, "scalar multiply: FP32 only");
  const uint32_t n = static_cast<uint32_t>(self.numel());
  const float scalar = other.toFloat();
  uint32_t bits;
  std::memcpy(&bits, &scalar, sizeof(bits));
  return elementwise_unary(self, "scalar_mul", {n, bits});
}

// aten::div.Tensor — elementwise divide (a / b)
at::Tensor div_tensor(const at::Tensor& self, const at::Tensor& other) {
  // For scalar-like divisors, use scalar path
  if (other.numel() == 1) {
    float divisor = other.item<float>();
    return mul_scalar(self, at::Scalar(1.0f / divisor));
  }
  // General case: compute reciprocal and multiply
  TORCH_CHECK(false, "div.Tensor: only scalar-like divisor supported for now");
}

// The old fused scaled-dot-product-attention code was removed because it
// disagreed with PyTorch for common head layouts and had no backward rule.
// Use PyTorch's CPU fallback until a fully tested Vulkan forward+backward
// implementation is ready.

// Proper storage-sharing views, adapted from MIT-licensed pytorch-XDNA2.
// CPU fallback cannot implement view ops without breaking aliasing and autograd.
// This maps PyTorch view metadata onto the same Vulkan-backed Storage.
static at::Tensor alias_with_strides(const at::Tensor& tensor,
                                     c10::SymIntArrayRef sizes,
                                     c10::SymIntArrayRef strides,
                                     c10::SymInt storage_offset) {
  auto impl = c10::make_intrusive<c10::TensorImpl>(
      c10::TensorImpl::VIEW,
      c10::Storage(tensor.storage()),
      tensor.key_set(), tensor.dtype());
  at::Tensor alias(std::move(impl));
  at::native::setStrided(alias, sizes, strides, std::move(storage_offset));
  return alias;
}

static at::Tensor vulkan_as_strided(const at::Tensor& tensor,
                                    c10::SymIntArrayRef sizes,
                                    c10::SymIntArrayRef strides,
                                    std::optional<c10::SymInt> storage_offset) {
  return alias_with_strides(tensor, sizes, strides,
                            storage_offset.value_or(tensor.sym_storage_offset()));
}

static at::Tensor vulkan_reshape_alias(const at::Tensor& tensor,
                                     c10::SymIntArrayRef sizes,
                                     c10::SymIntArrayRef strides) {
  return alias_with_strides(tensor, sizes, strides,
                            tensor.sym_storage_offset());
}

static at::Tensor vulkan_view(const at::Tensor& tensor,
                              c10::SymIntArrayRef requested) {
  auto shape = at::infer_size_dv(requested, tensor.sym_numel());
  auto strides = at::detail::computeStride(
      tensor.sym_sizes(), tensor.sym_strides(), c10::SymIntArrayRef(shape));
  TORCH_CHECK(strides.has_value(),
              "view size is incompatible with the input tensor's size/stride. "
              "Use .reshape(...) to copy non-contiguous tensors.");
  return alias_with_strides(tensor, c10::SymIntArrayRef(shape),
                            c10::SymIntArrayRef(*strides),
                            tensor.sym_storage_offset());
}

// aten::t — matrix transpose (2D only)
at::Tensor vulkan_t(const at::Tensor& self) {
  TORCH_CHECK(self.dim() == 2, "t: only 2D tensors");
  // This is a true PyTorch view; no CPU memory traversal or GPU transfer.
  const std::vector<c10::SymInt> shape = {self.sym_size(1), self.sym_size(0)};
  const std::vector<c10::SymInt> stride = {self.sym_stride(1), self.sym_stride(0)};
  return alias_with_strides(self, shape, stride, self.sym_storage_offset());
}

// Q4 quantized matmul: A (float32) @ B_packed (Q4) with scales
// B_packed: uint8 array, 2 values per byte, stored as uint32 words
// B_scales: float32 [K/32, N], one scale per block of 32
static at::Tensor mm_q4(const at::Tensor& input, const at::Tensor& weight_packed,
                         const at::Tensor& weight_scales) {
  VulkanContext::instance().synchronize();
  auto input_c = contiguous_for_kernel(input);
  int64_t M = input_c.size(0);
  int64_t K = input_c.size(1);
  int64_t N = weight_scales.size(1);

  auto& ctx = VulkanContext::instance();
  auto& mgr = ctx.manager();

  // Input: float32 [M, K]
  bool a_on_vk = input_c.device().type() == c10::DeviceType::PrivateUse1;
  auto tensor_a = a_on_vk ? get_kp_tensor(input_c)
      : mgr.tensor(std::vector<float>(input_c.data_ptr<float>(), input_c.data_ptr<float>() + M * K));

  // Weight packed: uint32 words (K*N/8 uint32s)
  auto wp_c = contiguous_for_kernel(weight_packed);
  bool wp_on_vk = wp_c.device().type() == c10::DeviceType::PrivateUse1;
  std::shared_ptr<kp::TensorT<float>> tensor_b;
  if (wp_on_vk) {
    tensor_b = get_kp_tensor(wp_c);
  } else {
    // Reinterpret packed uint8/uint32 data as float for Kompute
    size_t packed_floats = (K * N / 8 + 3) / 4 * 4;  // round up
    std::vector<float> packed_data(packed_floats, 0.0f);
    std::memcpy(packed_data.data(), wp_c.data_ptr(), wp_c.numel() * wp_c.element_size());
    tensor_b = mgr.tensor(packed_data);
  }

  // Scales: float32 [K/32, N]
  auto sc_c = contiguous_for_kernel(weight_scales);
  bool sc_on_vk = sc_c.device().type() == c10::DeviceType::PrivateUse1;
  auto tensor_scales = sc_on_vk ? get_kp_tensor(sc_c)
      : mgr.tensor(std::vector<float>(sc_c.data_ptr<float>(), sc_c.data_ptr<float>() + sc_c.numel()));

  // Output: float32 [M, N]
  auto output = at::empty({M, N}, input.options().device(c10::Device(c10::DeviceType::PrivateUse1, 0)));
  auto tensor_c = get_kp_tensor(output);

  auto spirv = ctx.load_shader("matmul_q4");
  uint32_t m_val = static_cast<uint32_t>(M);
  uint32_t k_val = static_cast<uint32_t>(K);
  uint32_t n_val = static_cast<uint32_t>(N);
  uint32_t wg_x = (N + 15) / 16;
  uint32_t wg_y = (M + 15) / 16;

  auto algorithm = ctx.get_or_create_algorithm(
      "matmul_q4", {tensor_a, tensor_b, tensor_scales, tensor_c},
      kp::Workgroup({wg_x, wg_y, 1}),
      std::vector<uint32_t>{m_val, k_val, n_val});

  ctx.synchronize();
  ctx.run_compute(algorithm, "matmul_q4", {tensor_a,tensor_b,tensor_scales}, {tensor_c});

  return output;
}

// Optional diagnostics for unsupported operations and implicit host copies.
// Usage: TORCH_VULKAN_TRACE_FALLBACK=1 python examples/linear_train.py
void cpu_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  VulkanContext::instance().synchronize();
  const bool trace = std::getenv("TORCH_VULKAN_TRACE_FALLBACK") != nullptr;
  if (!trace) {
    at::native::cpu_fallback(op, stack);
    return;
  }
  auto start = std::chrono::steady_clock::now();
  at::native::cpu_fallback(op, stack);
  double elapsed_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - start).count();
  std::fprintf(stderr, "PV_FALLBACK %s %.3f ms\n",
               op.schema().name().c_str(), elapsed_ms);
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("mm", &mm);
  m.impl("add.Tensor", &add_tensor);
  m.impl("mul.Tensor", &mul_tensor);
  m.impl("relu", &relu);
  m.impl("threshold_backward", &threshold_backward);
  m.impl("empty.memory_format", &empty_memory_format);
  m.impl("_softmax", &vulkan_softmax);
  m.impl("native_layer_norm", &vulkan_layer_norm);
  m.impl("_to_copy", &vulkan_to_copy);
  m.impl("copy_", &vulkan_copy_);
  m.impl("_copy_from_and_resize", &vulkan_copy_from_and_resize);
  m.impl("_copy_from", &vulkan_copy_from);
  m.impl("gelu", &vulkan_gelu);
  m.impl("gelu_backward", &vulkan_gelu_backward);
  m.impl("embedding", &vulkan_embedding);
  m.impl("addmm", &vulkan_addmm);
  m.impl("mul.Scalar", &mul_scalar);
  m.impl("div.Tensor", &div_tensor);
  // The old custom SDPA shader is not accurate for all head layouts and has
  // no Autograd implementation. Do not register it: dispatch through the
  // correctness-first CPU fallback until a tested forward+backward exists.
  // Let PyTorch's composite reshape preserve autograd and call native view.
  // The prior custom memcpy reshape bypassed the required autograd rule.
  m.impl("t", &vulkan_t);
  m.impl("view", &vulkan_view);
  m.impl("as_strided", &vulkan_as_strided);
  m.impl("_reshape_alias", &vulkan_reshape_alias);
}

TORCH_LIBRARY_IMPL(_, PrivateUse1, m) {
  m.fallback(torch::CppFunction::makeFromBoxedFunction<&cpu_fallback>());
}

} // namespace ops
} // namespace torch_vulkan

// Custom ops (outside namespace for TORCH_LIBRARY)
static at::Tensor mm_q4_dispatch(const at::Tensor& input, const at::Tensor& weight_packed,
                                  const at::Tensor& weight_scales) {
  return torch_vulkan::ops::mm_q4(input, weight_packed, weight_scales);
}

TORCH_LIBRARY(torch_vulkan_ops, m) {
  m.def("mm_q4(Tensor input, Tensor weight_packed, Tensor weight_scales) -> Tensor");
  m.impl("mm_q4", &mm_q4_dispatch);
}
