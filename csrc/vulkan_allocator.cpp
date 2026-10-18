#include "vulkan_allocator.h"
#include "vulkan_context.h"
#include <c10/core/Device.h>
#include <cstring>
#include <cstdlib>
#include <cerrno>
#include <cstdio>

namespace torch_vulkan {

VulkanAllocator& VulkanAllocator::instance() {
  static VulkanAllocator alloc;
  return alloc;
}

c10::DataPtr VulkanAllocator::allocate(size_t nbytes) {
  if (nbytes == 0) {
    return c10::DataPtr(nullptr, c10::Device(c10::DeviceType::PrivateUse1, 0));
  }

  size_t num_floats = (nbytes + sizeof(float) - 1) / sizeof(float);

  std::shared_ptr<kp::TensorT<float>> tensor;

  {
    std::lock_guard<std::mutex> lock(mutex_);

    // Phase 4: Check free list for a reusable buffer of the same size.
    auto it = free_list_.find(num_floats);
    if (it != free_list_.end() && !it->second.empty()) {
      tensor = std::move(it->second.back());
      it->second.pop_back();
      // Zero out the reused buffer for safety
      std::memset(tensor->data(), 0, num_floats * sizeof(float));
    }
  }

  if (!tensor) {
    // Experimental zero-copy path: choose host-visible Vulkan storage to
    // remove staging-buffer transfers. Unlike device-local memory, the GPU
    // reads/writes host system RAM directly, which can be slow on PCIe GPUs.
    // 0 (default): all device-local; MAX: all host-visible; N: <= N bytes host.
    // Treat this as a per-process setting; do not change it while running.
    static const size_t host_max_bytes = []() -> size_t {
      const char* value = std::getenv("TORCH_VULKAN_HOST_VISIBLE_MAX_BYTES");
      if (!value || !*value) return 0;
      char* end = nullptr;
      errno = 0;
      unsigned long long parsed = std::strtoull(value, &end, 10);
      if (errno != 0 || end == value || *end != '\0') {
        throw std::runtime_error("Invalid TORCH_VULKAN_HOST_VISIBLE_MAX_BYTES value");
      }
      return static_cast<size_t>(parsed);
    }();
    auto& mgr = VulkanContext::instance().manager();
    std::vector<float> zeros(num_floats, 0.0f);
    const auto tensor_type = nbytes <= host_max_bytes
        ? kp::Tensor::TensorTypes::eHost
        : kp::Tensor::TensorTypes::eDevice;
    tensor = mgr.tensor(zeros, tensor_type);
  }

  void* ptr = tensor->data();

  {
    std::lock_guard<std::mutex> lock(mutex_);
    tensor_map_[ptr] = tensor;
    residency_[ptr] = Residency::HostDirty;
  }

  return c10::DataPtr(
      ptr,
      ptr,
      &VulkanAllocator::deleter,
      c10::Device(c10::DeviceType::PrivateUse1, 0));
}

c10::DeleterFnPtr VulkanAllocator::raw_deleter() const {
  return &VulkanAllocator::deleter;
}

void VulkanAllocator::deleter(void* ptr) {
  if (!ptr) return;
  auto& alloc = VulkanAllocator::instance();
  std::lock_guard<std::mutex> lock(alloc.mutex_);

  auto it = alloc.tensor_map_.find(ptr);
  if (it == alloc.tensor_map_.end()) return;

  // Phase 4: Instead of destroying the Kompute tensor, move it to the free
  // list keyed by size. The next allocation of the same size will reuse this
  // buffer, getting the same data pointer, which means the algorithm cache
  // in VulkanContext will match and reuse the compiled VkPipeline.
  auto tensor = std::move(it->second);
  size_t num_floats = tensor->size();
  if (alloc.defer_reuse_)
    alloc.deferred_.push_back(std::move(tensor));
  else
    alloc.free_list_[num_floats].push_back(std::move(tensor));
  alloc.tensor_map_.erase(it);
  alloc.residency_.erase(ptr);
}

void VulkanAllocator::defer_reuse(bool enabled) {
  std::lock_guard<std::mutex> lock(mutex_);
  defer_reuse_ = enabled;
  if (!enabled) {
    for (auto& t : deferred_) {
      free_list_[t->size()].push_back(std::move(t));
    }
    deferred_.clear();
  }
}

bool VulkanAllocator::needs_upload(const std::shared_ptr<kp::Tensor>& t) {
  if (t->tensorType() != kp::Tensor::TensorTypes::eDevice) return false;
  std::lock_guard<std::mutex> lock(mutex_);
  auto it=residency_.find(t->rawData());
  // Temporary Kompute buffers that did not come through our allocator
  // are initialized from CPU and always need the upload.
  return it==residency_.end() || it->second==Residency::HostDirty;
}

bool VulkanAllocator::gpu_dirty(const std::shared_ptr<kp::Tensor>& t) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it=residency_.find(t->rawData());
  return it!=residency_.end() && it->second==Residency::GpuDirty;
}

bool VulkanAllocator::needs_download(const std::shared_ptr<kp::Tensor>& t) {
  return t->tensorType()==kp::Tensor::TensorTypes::eDevice && gpu_dirty(t);
}

void VulkanAllocator::mark_gpu_dirty(const std::shared_ptr<kp::Tensor>& t) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it=residency_.find(t->rawData());
  if (it!=residency_.end()) it->second=Residency::GpuDirty;
}

void VulkanAllocator::mark_clean(const std::shared_ptr<kp::Tensor>& t) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it=residency_.find(t->rawData());
  if (it!=residency_.end()) it->second=Residency::Clean;
}

void VulkanAllocator::mark_host_dirty(void* ptr) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it=residency_.find(ptr);
  if (it!=residency_.end()) it->second=Residency::HostDirty;
}

void VulkanAllocator::copy_data(void* dest, const void* src, std::size_t count) const {
  std::memcpy(dest, src, count);
}

std::shared_ptr<kp::TensorT<float>> VulkanAllocator::get_kompute_tensor(void* ptr) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = tensor_map_.find(ptr);
  if (it == tensor_map_.end()) {
    throw std::runtime_error("No Kompute tensor found for data pointer");
  }
  return it->second;
}

} // namespace torch_vulkan
