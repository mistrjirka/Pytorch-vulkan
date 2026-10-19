#include "../execution_audit.h"
#include "fft.h"
#include "integer_ops.h"
#include "device_copy.h"
#include <ATen/Functions.h>
#include <ATen/EmptyTensor.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/native/BinaryOps.h>
#include <ATen/ops/from_blob.h>
#include <ATen/NativeFunctions.h>
#include <ATen/InferSize.h>
#include <ATen/ExpandUtils.h>
#include <ATen/TensorUtils.h>
#include <ATen/native/Resize.h>
#include "../vulkan_allocator.h"
#include "../vulkan_engine.h"
#include "../vulkan_context.h"
#include "native_utils.h"
#include <ATen/native/layer_norm.h>
#include "../vulkan_device.h"
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
  auto mirror=at::from_blob(const_cast<void*>(src.const_data_ptr()),
      src.sizes(), src.strides(), [](void*) {},
      at::TensorOptions().dtype(src.scalar_type()).device(at::kCPU));
  mirror.unsafeGetTensorImpl()->_set_conj(src.is_conj());
  mirror.unsafeGetTensorImpl()->_set_neg(src.is_neg());
  return mirror;
}

static at::Tensor contiguous_for_kernel(const at::Tensor& src) {
  // A view with nonzero offset is NOT the base Kompute allocation, and a
  // strided view must not be consumed as a flat SPIR-V buffer. Pack supported
  // layouts on GPU; retain the existing host path for other dtypes/ranks.
  if (src.device().type() != c10::DeviceType::PrivateUse1)
    return src.contiguous();
  if (src.is_contiguous() && src.storage_offset() == 0)
    return src;
  static const bool gpu_pack=[](){
    const char* flag=std::getenv("TORCH_VULKAN_GPU_PACK_VIEWS");
    return !flag || std::strcmp(flag,"0")!=0;
  }();
  if(gpu_pack && native::can_pack_data(src))
    return native::packed(src,src.scalar_type());
  native::cpu_fallback_check(__func__);
  return host_mirror(src).contiguous().to(src.device());
}

// Helper: get existing Kompute tensor from a Vulkan PyTorch tensor (no copy!)
static std::shared_ptr<kp::TensorT<float>> get_kp_tensor(const at::Tensor& t) {
  auto& alloc = VulkanAllocator::instance();
  if(t._is_zerotensor())return alloc.constant_float_buffer(std::max<size_t>(1,(t.storage().nbytes()+3)/4),0);
  return alloc.get_kompute_tensor(const_cast<void*>(t.const_data_ptr()));
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
  if(!n)return at::empty(self.sizes(),self.options());

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
  if(!n)return at::empty(self.sizes(),self.options());

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
  TORCH_CHECK(self.scalar_type() == mat2.scalar_type(), "mm: expected both matrices to have the same dtype");
  if(self.scalar_type()==at::kComplexFloat){
    // Reuse the general FP32 GEMM implementation for complex64 arithmetic.
    // ATen real/imag views preserve lazy conjugation and arbitrary strides.
    auto ar=at::real(self),ai=at::imag(self),br=at::real(mat2),bi=at::imag(mat2);
    return at::complex(at::mm(ar,br).sub(at::mm(ai,bi)),at::mm(ar,bi).add(at::mm(ai,br)));
  }
  if(integer_type(self.scalar_type()))return integer_matmul(self,mat2);
  TORCH_CHECK(self.scalar_type() == at::kFloat, "mm: only float32 for now");
  if(self.is_neg()||mat2.is_neg())return at::mm(native::packed(self),native::packed(mat2));
  TORCH_CHECK(self.device() == mat2.device() && self.device().type() == c10::DeviceType::PrivateUse1,
              "mm: expected both matrices on the same Vulkan device");

  int64_t M = self.size(0), K = self.size(1), N = mat2.size(1);
  TORCH_CHECK(M >= 0 && K >= 0 && N >= 0 && M <= UINT32_MAX &&
              K <= UINT32_MAX && N <= UINT32_MAX,
              "mm: unsupported matrix dimensions");

  auto output = at::empty({M, N}, self.options());
  if (!M || !N) return output;
  if (!K) return output.fill_(0);
  TORCH_CHECK(self.numel() <= UINT32_MAX && mat2.numel() <= UINT32_MAX &&
              output.numel() <= UINT32_MAX, "mm: indexing exceeds uint32");
  TORCH_CHECK(native::can_pack_view(self) && native::can_pack_view(mat2),
              "mm: input layout or storage exceeds Vulkan indexing/device limits");
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

  const auto& limits=native::device_limits();
  TORCH_CHECK(wg_x<=limits.maxComputeWorkGroupCount[0] && wg_y<=limits.maxComputeWorkGroupCount[1] &&
              limits.maxComputeWorkGroupInvocations>=256 && limits.maxComputeWorkGroupSize[0]>=16 &&
              limits.maxComputeWorkGroupSize[1]>=16 &&
              limits.maxComputeSharedMemorySize>=(use_register_kernel?8192u:2048u) &&
              output.nbytes()<=limits.maxStorageBufferRange &&
              limits.maxPerStageDescriptorStorageBuffers>=3 && limits.maxDescriptorSetStorageBuffers>=3,
              "mm: dispatch exceeds Vulkan physical device limits");

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

// Python numbers enter the Tensor overload as zero-dimensional CPU tensors.
// Keep them as push constants; expanding them on CPU both loses wrapped-scalar
// dtype promotion and allocates/copies an entire tensor for one number.
static std::optional<at::Tensor> cpu_scalar_binary(
    const at::Tensor& a,const at::Tensor& b,float alpha,bool multiply) {
  static const bool enabled=[](){
    const char* flag=std::getenv("TORCH_VULKAN_DISABLE_SCALAR_BINARY");
    return !flag || std::strcmp(flag,"1")!=0;
  }();
  if(!enabled)return std::nullopt;
  const bool right=b.device().is_cpu() && b.dim()==0 &&
      a.device().type()==c10::DeviceType::PrivateUse1;
  const bool left=a.device().is_cpu() && a.dim()==0 &&
      b.device().type()==c10::DeviceType::PrivateUse1;
  if(!right && !left)return std::nullopt;
  const auto& tensor=right?a:b;
  const auto& scalar=right?b:a;
  if(tensor.scalar_type()!=at::kFloat || at::result_type(a,b)!=at::kFloat)return std::nullopt;
  auto out=at::empty(tensor.sizes(),tensor.options());
  if(out.numel()) {
    auto input=native::packed(tensor);
    native::run("scalar_binary",{native::base(input)},{native::write_base(out)},
        {native::u32(out.numel()),multiply?2u:(right?0u:1u),
         native::bits(scalar.item<float>()),native::bits(alpha)},(out.numel()+127)/128);
  }
  return out;
}

static void binary_device_check(const at::Tensor& a,const at::Tensor& b){
 auto dev=a.is_cpu()?b.device():a.device();
 TORCH_CHECK(dev.type()==c10::DeviceType::PrivateUse1 && (a.device()==dev||(a.is_cpu()&&a.dim()==0)) && (b.device()==dev||(b.is_cpu()&&b.dim()==0)),"Expected all tensors to be on the same device");
}

// aten::add.Tensor — alpha scaling with arbitrary compatible broadcasting.
at::Tensor add_tensor(const at::Tensor& self, const at::Tensor& other,
                      const at::Scalar& alpha) {
  binary_device_check(self,other);auto dtype=at::result_type(self,other);at::native::alpha_check(dtype,alpha);
  if(self.scalar_type()==at::kBool&&other.scalar_type()==at::kBool){
    if(alpha.toLong()!=0)return at::logical_or(self,other);
    auto shape=at::infer_size(self.sizes(),other.sizes());auto dev=self.is_cpu()?other.device():self.device();return self.to(dev).expand(shape).clone();
  }
  if(auto out=cpu_scalar_binary(self,other,alpha.toFloat(),false))return *out;
  if(device_copy_supported(dtype)&&(self.scalar_type()!=dtype||other.scalar_type()!=dtype))return add_tensor(self.to(dtype),other.to(dtype),alpha);
  if(self.is_complex()||other.is_complex())return complex_binary(self,other,0,alpha);
  if(integer_type(self.scalar_type())&&integer_type(other.scalar_type()))return integer_binary(self,other,alpha,false);
  if(auto out=cpu_scalar_binary(self,other,alpha.toFloat(),false))return *out;
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
  binary_device_check(self,other);
  if(self.scalar_type()==at::kBool&&other.scalar_type()==at::kBool)return at::logical_and(self,other);
  auto dtype=at::result_type(self,other);
  if(auto out=cpu_scalar_binary(self,other,1.0f,true))return *out;
  if(device_copy_supported(dtype)&&(self.scalar_type()!=dtype||other.scalar_type()!=dtype)&&!(self.device()==other.device()&&((self.scalar_type()==at::kBool&&other.scalar_type()==at::kFloat)||(other.scalar_type()==at::kBool&&self.scalar_type()==at::kFloat))))return mul_tensor(self.to(dtype),other.to(dtype));
  if((self.scalar_type()==at::kBool&&other.scalar_type()==at::kFloat)||(other.scalar_type()==at::kBool&&self.scalar_type()==at::kFloat)){
    TORCH_CHECK(self.device()==other.device(),"masked multiplication requires the same device");
    auto shape=at::infer_size(self.sizes(),other.sizes());auto mask=self.scalar_type()==at::kBool?self:other;auto value=self.scalar_type()==at::kFloat?self:other;
    return at::native_dropout_backward(value.expand(shape),mask.expand(shape),1.0);
  }
  if(integer_type(self.scalar_type())&&integer_type(other.scalar_type()))return integer_binary(self,other,1,true);
  if(self.scalar_type()==at::kComplexFloat&&other.is_cpu()&&other.dim()==0)return complex_scale(self,other.item());
  if(other.scalar_type()==at::kComplexFloat&&self.is_cpu()&&self.dim()==0)return complex_scale(other,self.item());
  if(self.is_complex()||other.is_complex())return complex_binary(self,other,1);
  if(auto out=cpu_scalar_binary(self,other,1.0f,true))return *out;
  auto [a, b] = broadcast_binary_operands(self, other);
  TORCH_CHECK(a.numel() <= UINT32_MAX, "Vulkan mul: tensor too large");
  const uint32_t n = static_cast<uint32_t>(a.numel());
  return elementwise_binary(a, b, "mul", {n});
}

// aten::relu — rectified linear
at::Tensor relu(const at::Tensor& self) {
  if(self.scalar_type()!=at::kFloat){TORCH_CHECK(self.scalar_type()!=at::kBool,"Boolean inputs not supported for relu");return at::clamp_min(self,0);}
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
  if(self.layout()==at::kStrided&&self.scalar_type()==at::kUInt64&&device_copy_supported(target_dtype)&&target_dtype!=at::kFloat)return vulkan_to_copy(self.view(at::kLong),target_dtype,layout,device,pin_memory,non_blocking,memory_format);
  if(self.layout()==at::kStrided&&target_dtype==at::kUInt64&&device_copy_supported(self.scalar_type())&&self.scalar_type()!=at::kFloat)return vulkan_to_copy(self,at::kLong,layout,device,pin_memory,non_blocking,memory_format).view(at::kUInt64);


  if (target_device.type() == c10::DeviceType::PrivateUse1) {
    validate_vulkan_device(target_device);
    if(self.device()==target_device || (self.device().type()==c10::DeviceType::PrivateUse1&&!target_device.has_index())){
      TORCH_CHECK(layout.value_or(at::kStrided)==at::kStrided&&!pin_memory.value_or(false),"invalid Vulkan copy layout/pinning");
      if(target_dtype==self.scalar_type()){
        auto result=at::empty_like(self,self.options(),memory_format);result.copy_(self);return result;
      }
      if((device_copy_supported(self.scalar_type())||self.scalar_type()==at::kComplexFloat)&&(device_copy_supported(target_dtype)||target_dtype==at::kComplexFloat)){
        auto result=at::empty_like(self,self.options().dtype(target_dtype),memory_format);
        copy_on_device(result,self);return result;
      }
      if(self.scalar_type()==at::kBool&&target_dtype==at::kFloat){
        auto result=at::empty_like(self,self.options().dtype(at::kFloat),memory_format);
        if(self.numel()){auto input=native::packed(self,at::kBool);auto output=result.is_contiguous()?result:at::empty(self.sizes(),result.options());
          native::run("bool_to_float",{native::base(input,at::kBool)},{native::write_base(output)},{native::u32(self.numel())},(self.numel()+127)/128);
          if(!result.is_contiguous())result.copy_(output);
        }return result;
      }
    }
    if(self.device().type()==c10::DeviceType::PrivateUse1)native::cpu_fallback_check(__func__,std::string("source dtype: ")+c10::toString(self.scalar_type())+"; target dtype: "+c10::toString(target_dtype));
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
  if(self.device()==src.device() && self.scalar_type()==src.scalar_type() &&
     self.const_data_ptr()==src.const_data_ptr() && self.sizes()==src.sizes() && self.strides()==src.strides() && self.is_conj()==src.is_conj() && self.is_neg()==src.is_neg())return self;
  // Check aliases before host mirrors wrap the same bytes in separate storages.
  at::assert_no_internal_overlap(self);
  at::assert_no_partial_overlap(self,src);
  if(self.numel())self.mutable_data_ptr();
  // Equal-width unsigned storage copies preserve all bits through the signed view.
  if(self.scalar_type()==at::kUInt64&&(src.scalar_type()==at::kUInt64||(device_copy_supported(src.scalar_type())&&src.scalar_type()!=at::kFloat))){auto destination=self.view(at::kLong),source=src.scalar_type()==at::kUInt64?src.view(at::kLong):src;vulkan_copy_(destination,source,non_blocking);return self;}
  if(src.scalar_type()==at::kUInt64&&device_copy_supported(self.scalar_type())&&self.scalar_type()!=at::kFloat){auto source=src.view(at::kLong);vulkan_copy_(self,source,non_blocking);return self;}
  if(copy_on_device(self,src))return self;
  if(self.device().type()==c10::DeviceType::PrivateUse1&&src.device()==self.device()&&self.scalar_type()==at::kFloat&&src.scalar_type()==at::kBool){
    auto converted=vulkan_to_copy(src,at::kFloat,std::nullopt,std::nullopt,std::nullopt,false,std::nullopt);
    return vulkan_copy_(self,converted,non_blocking);
  }
  // Copy non-overlapping strided destinations directly on GPU, retaining the
  // enclosing allocation and its untouched elements.
  if(self.device().type()==c10::DeviceType::PrivateUse1 && src.device()==self.device() &&
     self.scalar_type()==src.scalar_type() && self.element_size()>=4 &&
     self.numel()>0 && !self.storage().is_alias_of(src.storage()) && native::can_pack_view(self) &&
     (self.is_neg()==src.is_neg()||self.is_floating_point()||self.is_complex())) {
    const auto shape=at::infer_size(self.sizes(),src.sizes());
    TORCH_CHECK(at::IntArrayRef(shape)==self.sizes(),"copy broadcast shape mismatch");
    auto input=src.expand(self.sizes());
    std::vector<std::pair<int64_t,int64_t>> axes;
    for(int d=0;d<self.dim();d++)if(self.size(d)>1)axes.push_back({self.stride(d),self.size(d)});
    std::sort(axes.begin(),axes.end());uint64_t span=1;bool independent=true;
    for(auto [stride,size]:axes){if(uint64_t(stride)<span)independent=false;span+=uint64_t(size-1)*stride;}
    if(independent && native::can_pack_view(input)) {
      std::vector<uint32_t> p(30,0);p[28]=uint32_t(self.is_conj()!=src.is_conj());p[29]=self.is_neg()!=src.is_neg()?(self.scalar_type()==at::kDouble||self.scalar_type()==at::kComplexDouble?2u:1u):0u;p[0]=native::u32(self.numel());p[1]=self.element_size()/4;
      p[2]=native::u32(input.storage_offset());p[3]=native::u32(self.storage_offset());
      std::fill(p.begin()+4,p.begin()+12,1);
      for(int d=0;d<self.dim();d++){p[4+d]=native::u32(self.size(d));p[12+d]=native::u32(input.stride(d));p[20+d]=native::u32(self.stride(d));}
      native::run("native_copy_strided",{native::base(input,input.scalar_type()),native::base(self,self.scalar_type())},{native::write_base(self,self.scalar_type())},p,(self.numel()+127)/128);
      return self;
    }
  }
  // General device copy into an entire contiguous allocation. Strided sources
  // use the existing pack shader; dtype casts, aliases and partial destinations
  // retain CPU's established copy semantics below.
  if(self.device().type()==c10::DeviceType::PrivateUse1 && src.device()==self.device() &&
     self.scalar_type()==src.scalar_type() && self.scalar_type()==at::kBool && self.is_neg()==src.is_neg() && self.is_conj()==src.is_conj() && self.is_contiguous() &&
     self.storage_offset()==0 && self.numel()>0 &&
     self.storage().nbytes()==uint64_t(self.numel())*self.element_size() &&
     !self.storage().is_alias_of(src.storage()) && native::can_pack_view(self)) {
    const auto shape=at::infer_size(self.sizes(),src.sizes());
    if(at::IntArrayRef(shape)==self.sizes()) {
      const auto input=src.sizes()==self.sizes()?src:src.expand(self.sizes());
      if(native::can_pack_view(input)) {
        std::vector<uint32_t> p(18,0);
        p[0]=native::u32(input.numel());p[1]=native::u32(input.storage_offset());
        std::fill(p.begin()+2,p.begin()+10,1);
        for(int64_t d=0;d<input.dim();++d) {
          p[2+d]=native::u32(input.size(d));p[10+d]=native::u32(input.stride(d));
        }
        const auto dtype=input.scalar_type();
        native::run(dtype==at::kBool?"native_pack_bool":dtype==at::kLong?"native_pack_int64":
                        dtype==at::kInt?"native_pack_int32":"native_pack",
            {native::base(input,dtype)},{native::write_base(self,dtype)},p,
            (input.numel()+(dtype==at::kBool?511:127))/(dtype==at::kBool?512:128));
        return self;
      }
    }
  }
  // A Vulkan allocation is represented by a mapped host mirror. CPU's
  // stride-aware copy correctly handles offsets, permutations and casting.
  // Inputs are synchronized to the GPU by the next shader submission.
  if(self.device().type()==c10::DeviceType::PrivateUse1&&src.device().type()==c10::DeviceType::PrivateUse1)native::cpu_fallback_check(__func__,std::string("source dtype: ")+c10::toString(src.scalar_type())+"; target dtype: "+c10::toString(self.scalar_type()));
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
  if (device) validate_vulkan_device(*device);
  TORCH_CHECK(layout.value_or(at::kStrided)==at::kStrided,"Vulkan empty requires strided layout");
  TORCH_CHECK(!pin_memory.value_or(false),"Vulkan empty does not support pinned memory");
  return at::Tensor(at::detail::empty_generic(size, &VulkanAllocator::instance(),
      c10::DispatchKeySet(c10::DispatchKey::PrivateUse1), c10::dtype_or_default(dtype), memory_format));
}

// Allocate arbitrary ATen strided storage directly; the generic helper validates
// sizes/strides and overflow and uses the backend allocator without a CPU tensor.
at::Tensor empty_strided_native(
    c10::SymIntArrayRef size,c10::SymIntArrayRef stride,
    std::optional<at::ScalarType> dtype,std::optional<at::Layout> layout,
    std::optional<at::Device> device,std::optional<bool> pin_memory) {
  if (device) validate_vulkan_device(*device);
  TORCH_CHECK(layout.value_or(at::kStrided)==at::kStrided,"Vulkan empty_strided requires strided layout");
  TORCH_CHECK(!pin_memory.value_or(false),"Vulkan empty_strided does not support pinned memory");
  return at::Tensor(at::detail::empty_strided_symint_generic(size,stride,
      &VulkanAllocator::instance(),c10::DispatchKeySet(c10::DispatchKey::PrivateUse1),
      c10::dtype_or_default(dtype)));
}

// General FP32 softmax along ANY axis, including the non-last dimension.
// Replaces the old shader that silently treated non-last axes as the last.
at::Tensor vulkan_softmax(const at::Tensor& self, int64_t dim, bool half_to_float) {
  TORCH_CHECK(self.scalar_type()==at::kFloat && !half_to_float,
              "Vulkan native softmax requires FP32 without half_to_float");
  const int64_t rank=std::max<int64_t>(1,self.dim());
  const int64_t axis=dim<0 ? dim+rank : dim;
  TORCH_CHECK(axis>=0 && axis<rank,"Vulkan softmax: dimension out of range");
  if(!native::can_pack_view(self))
    return at::_softmax(native::cpu_fallback_tensor(self,__func__),dim,half_to_float).to(self.device());
  auto output=at::empty(self.sizes(),self.options());
  if(!self.numel())return output;
  const int64_t axis_size=self.dim()?self.size(axis):1;
  uint64_t outer=1,inner=1;
  for(int64_t d=0;d<self.dim();++d){
    if(d<axis)outer*=uint64_t(self.size(d));
    else if(d>axis)inner*=uint64_t(self.size(d));
  }
  TORCH_CHECK(outer*inner*uint64_t(axis_size)<=UINT32_MAX,
              "Vulkan softmax: indexing exceeds uint32");
  const auto input=native::packed(self);
  native::run("softmax_axis",{native::base(input)},{native::write_base(output)},
      {native::u32(outer),native::u32(axis_size),native::u32(inner)},outer*inner,512);
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

  TORCH_CHECK(input.dtype() == at::kFloat, "layer_norm: only float32");
  const auto checked=at::native::_check_layer_norm_inputs(input,normalized_shape,weight_opt.value_or(at::Tensor()),bias_opt.value_or(at::Tensor()));
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
  auto num_rows = checked.first;
  TORCH_CHECK(num_rows <= UINT32_MAX && row_size <= UINT32_MAX,
              "layer_norm: tensor too large for Vulkan dispatch");

  auto& ctx = VulkanContext::instance();
  auto output = at::empty(input.sizes(), input.options());
  auto mean = at::empty(stat_shape, input.options());
  auto rstd = at::empty(stat_shape, input.options());
  if(!input.numel()){if(row_size==0){mean.zero_();rstd.fill_(NAN);}return {output,mean,rstd};}

  // The input can reuse its existing Vulkan allocation; temporary scalar
  // parameters are uploaded through the same Kompute submission.
  auto in_tensor = get_kp_tensor(input_c);
  auto out_tensor = get_kp_tensor(output);
  auto mean_tensor = get_kp_tensor(mean);
  auto rstd_tensor = get_kp_tensor(rstd);
  // Affine parameters already live in the backend allocator. Reuse their
  // VkBuffers instead of allocating/copying temporary buffers on every call.
  std::shared_ptr<kp::TensorT<float>> w_tensor, b_tensor;
  at::Tensor w_keep, b_keep; // Keep packed allocator storage alive through enqueue.
  if (weight_opt.has_value() && weight_opt->defined()) {
    w_keep = contiguous_for_kernel(*weight_opt);
    const auto& w = w_keep;
    TORCH_CHECK(w.dtype() == at::kFloat && w.numel() == row_size,
                "layer_norm: invalid weight");
    w_tensor = get_kp_tensor(w);
  } else {
    w_tensor = VulkanAllocator::instance().constant_float_buffer(row_size, 1.0f);
  }
  if (bias_opt.has_value() && bias_opt->defined()) {
    b_keep = contiguous_for_kernel(*bias_opt);
    const auto& b = b_keep;
    TORCH_CHECK(b.dtype() == at::kFloat && b.numel() == row_size,
                "layer_norm: invalid bias");
    b_tensor = get_kp_tensor(b);
  } else {
    b_tensor = VulkanAllocator::instance().constant_float_buffer(row_size, 0.0f);
  }
  const auto nr = static_cast<uint32_t>(num_rows);
  const auto rs = static_cast<uint32_t>(row_size);
  float eps_f = static_cast<float>(eps);
  uint32_t eps_bits;
  std::memcpy(&eps_bits, &eps_f, sizeof(eps_bits));
  auto algorithm = ctx.get_or_create_algorithm(
      "layer_norm", {in_tensor, w_tensor, b_tensor, out_tensor, mean_tensor, rstd_tensor},
      native::grid(nr), std::vector<uint32_t>{nr, rs, eps_bits});
  ctx.run_compute(algorithm, "layer_norm", {in_tensor,w_tensor,b_tensor}, {out_tensor,mean_tensor,rstd_tensor});
  return std::make_tuple(output, mean, rstd);
}

// aten::gelu — Gaussian Error Linear Unit via gelu.spv
at::Tensor vulkan_gelu(const at::Tensor& self, c10::string_view approximate) {
  TORCH_CHECK(approximate == "none" || approximate == "tanh", "approximate argument must be either none or tanh");
  if(approximate=="tanh"){auto t=at::tanh(self.add(self.mul(self).mul(self).mul(.044715)).mul(.7978845608028654));return self.mul(.5).mul(t.add(1));}
  uint32_t n = static_cast<uint32_t>(self.numel());
  return elementwise_unary(self, "gelu", {n});
}

at::Tensor vulkan_gelu_backward(const at::Tensor& grad,
                                 const at::Tensor& input,
                                 c10::string_view approximate) {
  TORCH_CHECK(approximate == "none" || approximate == "tanh", "approximate argument must be either none or tanh");
  if(approximate=="tanh"){auto t=at::tanh(input.add(input.mul(input).mul(input).mul(.044715)).mul(.7978845608028654));return grad.mul(t.add(1).mul(.5).add(input.mul(.3989422804014327).mul(at::rsub(t.mul(t),1)).mul(input.mul(input).mul(.134145).add(1))));}
  uint32_t n = static_cast<uint32_t>(input.numel());
  TORCH_CHECK(grad.sizes() == input.sizes(), "Vulkan GELU backward shape mismatch");
  return elementwise_binary(input, grad, "gelu_backward", {n});
}

// Exact integer indices, including int64 values beyond FP32's exact range.
// Validate on GPU before gathering; only a four-byte result crosses to the host.
at::Tensor vulkan_embedding(const at::Tensor& weight, const at::Tensor& indices,
    int64_t padding_idx, bool scale_grad_by_freq, bool sparse) {
  TORCH_CHECK(weight.dim()==2, "'weight' must be 2-D");
  TORCH_CHECK(weight.scalar_type()==at::kFloat, "embedding: only float32 weight");
  TORCH_CHECK(indices.scalar_type()==at::kLong || indices.scalar_type()==at::kInt,
              "embedding: indices must have int32 or int64 dtype");
  TORCH_CHECK(weight.device()==indices.device() && weight.device().type()==c10::DeviceType::PrivateUse1,
              "embedding: weight and indices must be on the same Vulkan device");
  auto sizes=indices.sizes().vec(); sizes.push_back(weight.size(1));
  auto output=at::empty(sizes, weight.options());
  if (!indices.numel()) return output;
  if (!native::can_pack_view(weight) || !native::can_pack_view(indices) ||
      output.numel()>UINT32_MAX || output.nbytes()>native::device_limits().maxStorageBufferRange)
    return at::embedding(native::cpu_fallback_tensor(weight,__func__),native::cpu_fallback_tensor(indices,__func__),padding_idx,scale_grad_by_freq,sparse).to(weight.device());
  auto idx=native::packed(indices,indices.scalar_type());
  const uint32_t words=indices.scalar_type()==at::kLong?2:1;
  const uint32_t groups=std::min<int64_t>(256,(idx.numel()+4095)/4096);
  auto invalid=at::empty({groups},weight.options());
  native::run("embedding_validate",{native::base(idx,idx.scalar_type())},{native::write_base(invalid)},
      {native::u32(idx.numel()),native::u32(weight.size(0)),words},groups,512);
  auto status=groups==1?invalid:invalid.sum();
  TORCH_CHECK_INDEX(status.cpu().item<float>()==0, "embedding: index out of range in self");
  if (!output.numel()) return output;
  auto w=native::packed(weight);
  native::run("embedding",{native::base(w),native::base(idx,idx.scalar_type())},{native::write_base(output)},
      {native::u32(output.numel()),native::u32(weight.size(1)),words},(output.numel()+127)/128);
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

  TORCH_CHECK(mat1.dim()==2 && mat2.dim()==2, "addmm: mat1 and mat2 must be matrices");
  TORCH_CHECK(mat1.size(1)==mat2.size(0), "addmm: matrix shapes cannot be multiplied");
  TORCH_CHECK(self.device()==mat1.device() && self.device()==mat2.device(),
              "addmm: all tensors must be on the same device");
  TORCH_CHECK(self.scalar_type()==mat1.scalar_type() && self.scalar_type()==mat2.scalar_type(),
              "addmm: all tensors must have the same dtype");
  const std::vector<int64_t> shape={mat1.size(0),mat2.size(1)};
  const auto bias=self.expand(shape); // Validate broadcasting even when beta is zero.
  if(!mat1.size(0) || !mat2.size(1))return at::empty(shape,mat1.options());
  if(!mat1.size(1))return beta.toFloat()==0 ? at::zeros(shape,mat1.options()) : mul_scalar(bias,beta);

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
  if (alpha.toFloat()!=1.0f) product=mul_scalar(product,alpha);
  // beta == 0 must not propagate NaN/Inf from the ignored bias.
  if (beta.toFloat()==0.0f) return product;
  return add_tensor(product, bias, beta);
}


// aten::mul.Scalar — elementwise scalar multiply via scalar_mul.spv
// Phase 4: cached algorithm
at::Tensor mul_scalar(const at::Tensor& self, const at::Scalar& other) {
  if(self.scalar_type()==at::kComplexFloat)return complex_scale(self,other);
  if(device_copy_supported(self.scalar_type())&&self.scalar_type()!=at::kFloat)return at::mul(self,at::scalar_tensor(other,at::TensorOptions().dtype(at::result_type(self,other))));
  TORCH_CHECK(self.scalar_type() == at::kFloat, "scalar multiply: FP32 only");
  const uint32_t n = static_cast<uint32_t>(self.numel());
  const float scalar = other.toFloat();
  uint32_t bits;
  std::memcpy(&bits, &scalar, sizeof(bits));
  return elementwise_unary(self, "scalar_mul", {n, bits});
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
  TORCH_CHECK(self.dim() <= 2, "t() expects a tensor with <= 2 dimensions, but self is ", self.dim(), "D");
  if(self.dim()<2)return alias_with_strides(self,self.sym_sizes(),self.sym_strides(),self.sym_storage_offset());
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

const at::Tensor& resize_native(const at::Tensor& self,c10::SymIntArrayRef symbolic,std::optional<at::MemoryFormat> format){
  std::vector<int64_t> size;for(auto n:symbolic){auto v=n.expect_int();TORCH_CHECK(v>=0,"negative resize dimension");size.push_back(v);}
  TORCH_CHECK(!format||*format!=at::MemoryFormat::Preserve,"resize_ does not support preserve memory format");
  const auto bytes=at::detail::computeStorageNbytesContiguous(size,self.element_size(),self.storage_offset());
  if(bytes>self.storage().nbytes())resize_vulkan_storage(self.storage().unsafeGetStorageImpl(),bytes);
  at::native::resize_impl_cpu_(self.unsafeGetTensorImpl(),size,std::nullopt,false);
  if(format)self.unsafeGetTensorImpl()->empty_tensor_restride(*format);
  return self;
}

at::Scalar local_scalar_dense(const at::Tensor& self) {
  TORCH_CHECK(self.numel()==1,"a Tensor with ",self.numel()," elements cannot be converted to Scalar");
  return host_mirror(self).item();
}

// Optional diagnostics for unsupported operations and implicit host copies.
// Usage: TORCH_VULKAN_TRACE_FALLBACK=1 python examples/linear_train.py
void cpu_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
  if(std::getenv("TORCH_VULKAN_TRACE_FALLBACK"))std::fprintf(stderr,"PV_FALLBACK_ATTEMPT %s overload=%s\n",op.schema().name().c_str(),op.schema().overload_name().c_str());
  audit::fallback_calls.fetch_add(1);
  TORCH_CHECK(!std::getenv("TORCH_VULKAN_FORBID_CPU_FALLBACK"),"CPU fallback forbidden: ",op.schema());
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
  std::fprintf(stderr, "PV_FALLBACK %s %.3f ms overload=%s\n",
               op.schema().name().c_str(), elapsed_ms,op.schema().overload_name().c_str());
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("resize_",&resize_native);
  m.impl("_local_scalar_dense", &local_scalar_dense);
  m.impl("mm", &mm);
  m.impl("add.Tensor", &add_tensor);
  m.impl("mul.Tensor", &mul_tensor);
  m.impl("relu", &relu);
  m.impl("threshold_backward", &threshold_backward);
  m.impl("empty.memory_format", &empty_memory_format);
  m.impl("empty_strided", &empty_strided_native);
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
  // General division is registered by loss_primitives.cpp.
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
