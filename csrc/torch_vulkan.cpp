// torch_vulkan.cpp — Entry point for the PrivateUse1 Vulkan backend.
//
// When Python does `import torch_vulkan`, this shared library loads and:
// 1. Renames PrivateUse1 -> "vulkan" so you get tensor.vulkan(), tensor.is_vulkan
// 2. Registers the VulkanAllocator for device memory
// 3. Operator registrations happen via TORCH_LIBRARY_IMPL in ops/*.cpp
//
// Usage:
//   import torch
//   import torch_vulkan
//   a = torch.randn(64, 64).vulkan()
//   b = torch.randn(64, 64).vulkan()
//   c = torch.mm(a, b)  # dispatches to matmul.spv on GPU

#include "vulkan_allocator.h"
#include "vulkan_context.h"
#include <c10/core/impl/alloc_cpu.h>
#include <torch/extension.h>
#include <cstring>
#include <ATen/detail/PrivateUse1HooksInterface.h>

namespace torch_vulkan {

// DeviceGuard integration — tells PyTorch how to manage Vulkan device context
struct VulkanGuardImpl final : public c10::impl::DeviceGuardImplInterface {
  c10::DeviceType type() const override {
    return c10::DeviceType::PrivateUse1;
  }

  c10::Device exchangeDevice(c10::Device d) const override {
    // Single-device for now (device 0)
    return c10::Device(c10::DeviceType::PrivateUse1, 0);
  }

  c10::Device getDevice() const override {
    return c10::Device(c10::DeviceType::PrivateUse1, 0);
  }

  void setDevice(c10::Device d) const override {
    // Single device — nothing to do
  }

  void uncheckedSetDevice(c10::Device d) const noexcept override {}

  c10::Stream getStream(c10::Device d) const noexcept override {
    return c10::Stream(c10::Stream::DEFAULT, d);
  }

  c10::Stream getDefaultStream(c10::Device d) const override {
    return c10::Stream(c10::Stream::DEFAULT, d);
  }

  c10::Stream exchangeStream(c10::Stream s) const noexcept override {
    return c10::Stream(c10::Stream::DEFAULT, s.device());
  }

  c10::DeviceIndex deviceCount() const noexcept override {
    return 1; // Single Vulkan device for now
  }
};

// Resize PyTorch storage for fallback outputs. The allocator owns the entire
// Kompute/Vulkan allocation and its backing host pointer, so replace its
// DataPtr rather than reallocating that host pointer behind Kompute's back.
void resize_vulkan_storage(c10::StorageImpl* impl, size_t bytes) {
  TORCH_CHECK(impl != nullptr, "vulkan resize: null storage");
  const size_t old_bytes = impl->nbytes();
  if (bytes <= old_bytes) return;
  auto next = VulkanAllocator::instance().allocate(bytes);
  if (old_bytes && impl->data() != nullptr) {
    std::memcpy(next.get(), impl->data(), old_bytes);
  }
  impl->set_data_ptr(std::move(next));
  impl->set_nbytes(bytes);
  impl->set_allocator(&VulkanAllocator::instance());
  impl->set_resizable(true);
}

// Runtime hooks required by PyTorch 2.14 to execute AutogradPrivateUse1.
// CPU fallback is not a replacement for these hooks: the autograd engine calls
// into the accelerator interface even when all backward operations exist.
struct VulkanHooks final : public at::PrivateUse1HooksInterface {
  bool isBuilt() const override { return true; }
  // The public Python `is_available()` does the full Vulkan-driver check.
  bool isAvailable() const override { return true; }
  bool hasPrimaryContext(c10::DeviceIndex index) const override {
    return index == 0;
  }
  c10::DeviceIndex deviceCount() const override { return 1; }
  c10::DeviceIndex getCurrentDevice() const override { return 0; }
  void setCurrentDevice(c10::DeviceIndex index) const override {
    TORCH_CHECK(index == 0, "Vulkan backend currently supports only device 0");
  }
  c10::Device getDeviceFromPtr(void*) const override {
    return c10::Device(c10::DeviceType::PrivateUse1, 0);
  }
  void init() const override { VulkanContext::instance().manager(); }
  void resizePrivateUse1Bytes(const c10::Storage& storage, size_t bytes) const override {
    resize_vulkan_storage(storage.unsafeGetStorageImpl(), bytes);
  }
};

static VulkanHooks vulkan_hooks;

// Register the guard
C10_REGISTER_GUARD_IMPL(PrivateUse1, VulkanGuardImpl);

// Python bindings
void set_shader_dir(const std::string& path) {
  // Set the shader dir on the Kompute-based VulkanContext (used by all the
  // registered ops). The raw VulkanEngine (used only by the not-yet-wired
  // mm_raw path) is constructed lazily and reads its directory from the
  // TORCH_VULKAN_SHADER_DIR env var, which __init__.py points at this same
  // directory -- so we deliberately do NOT touch VulkanEngine::instance()
  // here, to avoid eagerly creating a second Vulkan device at import time.
  VulkanContext::instance().set_shader_dir(path);
}

std::string device_name() {
  return VulkanContext::instance().device_name();
}

bool is_hardware_accelerated() {
  try {
    const auto ty = VulkanContext::instance().manager().getDeviceProperties().deviceType;
    return ty == vk::PhysicalDeviceType::eDiscreteGpu ||
           ty == vk::PhysicalDeviceType::eIntegratedGpu;
  } catch (...) {
    return false;
  }
}

bool is_available() {
  try {
    VulkanContext::instance().manager();
    return true;
  } catch (...) {
    return false;
  }
}

// Phase 4: Expose cache statistics to Python for benchmarking.
py::dict cache_stats() {
  auto stats = VulkanContext::instance().cache_stats();
  py::dict d;
  d["algo_hits"] = stats.algo_hits;
  d["algo_misses"] = stats.algo_misses;
  d["seq_reuses"] = stats.seq_reuses;
  d["seq_creates"] = stats.seq_creates;
  d["algo_evictions"] = stats.algo_evictions;
  d["bound_algorithms"] = stats.bound_algorithms;
  d["pipeline_hits"] = stats.pipeline_hits;
  d["pipeline_misses"] = stats.pipeline_misses;
  d["pipeline_evictions"] = stats.pipeline_evictions;
  d["pipeline_entries"] = stats.pipeline_entries;
  return d;
}

void clear_algorithm_cache() {
  VulkanContext::instance().clear_algorithm_cache();
}

void synchronize() { VulkanContext::instance().synchronize(); }

py::dict queue_stats() {
  const auto st=VulkanContext::instance().queue_stats();
  py::dict d;
  d["submissions"]=st.submissions;
  d["recorded_ops"]=st.recorded_ops;
  d["largest_batch"]=st.largest_batch;
  return d;
}

py::dict gpu_timing_stats() {
  py::dict res;
  for (const auto& [name,v] : VulkanContext::instance().gpu_timing_stats()) {
    py::dict stats;
    stats["samples"]=v.samples;
    stats["upload_ms"]=v.upload_ms;
    stats["shader_ms"]=v.shader_ms;
    stats["download_ms"]=v.download_ms;
    stats["host_eval_ms"]=v.host_eval_ms;
    res[py::str(name)]=std::move(stats);
  }
  return res;
}

PYBIND11_MODULE(_C, m) {
  m.def("_set_shader_dir", &set_shader_dir);
  m.def("_device_name", &device_name);
  m.def("_is_available", &is_available);
  m.def("_is_hardware_accelerated", &is_hardware_accelerated);
  m.def("_synchronize", &synchronize);
  m.def("_queue_stats", &queue_stats);
  m.def("_gpu_timing_stats", &gpu_timing_stats);
  m.def("_cache_stats", &cache_stats,
        "Return algorithm cache hit/miss statistics (Phase 4)");
  m.def("_clear_algorithm_cache", &clear_algorithm_cache,
        "Clear all cached Vulkan algorithms");

  // Rename PrivateUse1 -> "vulkan"
  // This gives us: tensor.vulkan(), tensor.is_vulkan, torch.device("vulkan")
  c10::register_privateuse1_backend("vulkan");
  at::RegisterPrivateUse1HooksInterface(&vulkan_hooks);

  // Register our allocator for the vulkan device
  c10::SetAllocator(
      c10::DeviceType::PrivateUse1, &VulkanAllocator::instance());
}

} // namespace torch_vulkan
