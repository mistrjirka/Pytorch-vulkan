#pragma once

#include <array>
#include <cstring>
#include <c10/core/Allocator.h>
#include <kompute/Kompute.hpp>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace torch_vulkan {

// Phase 4: Buffer-reusing allocator.
// When a Vulkan tensor is freed, its underlying Kompute tensor (and Vulkan
// buffer) goes onto a size-bucketed free list. The next allocation of the
// same size reuses the existing buffer, which means:
//   1. The data pointer is the same as before, so
//   2. The algorithm cache in VulkanContext can match on tensor pointers and
//   3. The cached Algorithm (VkPipeline + descriptor sets) is reused.
// This eliminates the 3-5ms per-op overhead of pipeline + descriptor set
// recreation during transformer inference.
class VulkanAllocator final : public c10::Allocator {
public:
  static VulkanAllocator& instance();

  c10::DataPtr allocate(size_t nbytes) override;
  c10::DeleterFnPtr raw_deleter() const override;
  void copy_data(void* dest, const void* src, std::size_t count) const override;

  std::shared_ptr<kp::TensorT<float>> get_kompute_tensor(void* ptr);
  // Internal, read-only FP32 identity buffers. Only the first count elements
  // may be read. Never expose these as writable tensors or kernel outputs.
  // Cached device-local buffers participate in ordinary upload tracking.
  std::shared_ptr<kp::TensorT<float>> constant_float_buffer(size_t count, float value);

  // Experimental dirty-bit tracking for device-local buffers. The host-mapped
  // pointer is staging memory, not necessarily the current GPU result.
  enum class Residency { HostDirty, GpuDirty, Clean };
  bool needs_upload(const std::shared_ptr<kp::Tensor>& tensor);
  bool needs_download(const std::shared_ptr<kp::Tensor>& tensor);
  bool gpu_dirty(const std::shared_ptr<kp::Tensor>& tensor);
  void mark_gpu_dirty(const std::shared_ptr<kp::Tensor>& tensor);
  void mark_clean(const std::shared_ptr<kp::Tensor>& tensor);
  void mark_host_dirty(void* raw_storage_ptr);

  // Prevent reuse of storage still referenced by a queued GPU command.
  void defer_reuse(bool enabled);

private:
  VulkanAllocator() = default;

  static void deleter(void* ptr);

  std::mutex mutex_;
  bool defer_reuse_ = false;
  std::vector<std::shared_ptr<kp::TensorT<float>>> deferred_;

  // Active allocations: data_ptr -> Kompute tensor
  std::unordered_map<void*, std::shared_ptr<kp::TensorT<float>>> tensor_map_;
  std::unordered_map<void*, Residency> residency_;

  // Zero/one power-of-two capacity buckets, up to 64K floats each. Including
  // staging memory, all 34 entries have less than 2 MiB of buffer payload
  // (driver allocation granularity is additional). Larger requests use ordinary
  // temporary buffers, so arbitrary shapes remain valid.
  static constexpr size_t kConstantBuckets = 17;
  std::array<std::array<std::shared_ptr<kp::TensorT<float>>, kConstantBuckets>, 2>
      constant_buffers_{};

  // Free list: size_in_floats -> list of reusable Kompute tensors
  // When a tensor is freed, it moves here instead of being destroyed.
  std::unordered_map<size_t, std::vector<std::shared_ptr<kp::TensorT<float>>>> free_list_;
};

} // namespace torch_vulkan
