#include "vulkan_context.h"
#include <fstream>
#include <cstdlib>
#include <cerrno>
#include <limits>
#include "vulkan_allocator.h"
#include <cstring>
#include <algorithm>
#include <c10/util/Exception.h>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace torch_vulkan {

VulkanContext& VulkanContext::instance() {
  static VulkanContext ctx;
  return ctx;
}

VulkanContext::VulkanContext() {
  // Kompute enumerates physical Vulkan devices. This process exposes a single
  // PyTorch logical device, but may select which physical GPU backs it.
  // Important on multi-GPU systems (e.g. RTX 2080 Ti + Tesla V100).
  uint32_t physical_index = 0;
  const char* selected = std::getenv("TORCH_VULKAN_PHYSICAL_DEVICE_INDEX");
  if (selected && *selected) {
    errno = 0;
    char* end = nullptr;
    unsigned long parsed = std::strtoul(selected, &end, 10);
    TORCH_CHECK(errno == 0 && end != selected && *end == '\0' &&
                parsed <= std::numeric_limits<uint32_t>::max(),
                "Invalid TORCH_VULKAN_PHYSICAL_DEVICE_INDEX=", selected);
    physical_index = static_cast<uint32_t>(parsed);
  }
  mgr_ = std::make_unique<kp::Manager>(physical_index);

  // Default shader directory: alongside the shared library
  shader_dir_ = "";
}

VulkanContext::~VulkanContext() {
  try { synchronize(); } catch (...) { /* Shutdown must not throw. */ }
}

bool VulkanContext::deferred_enabled() const {
  static const bool enabled = []() {
    const char* flag = std::getenv("TORCH_VULKAN_DEFER_SUBMISSIONS");
    return flag && std::strcmp(flag, "1") == 0;
  }();
  return enabled;
}

bool VulkanContext::can_enqueue(
    const std::vector<std::shared_ptr<kp::Tensor>>& bindings) const {
  if (!deferred_enabled()) return false;
  return std::all_of(bindings.begin(), bindings.end(), [](const auto& tensor) {
    return tensor->tensorType() == kp::Tensor::TensorTypes::eHost;
  });
}

void VulkanContext::enqueue_compute(
    const std::shared_ptr<kp::Algorithm>& algorithm,
    const std::vector<std::shared_ptr<kp::Tensor>>& bindings,
    const std::vector<std::shared_ptr<kp::Tensor>>& outputs) {
  TORCH_CHECK(deferred_enabled(), "enqueue_compute requires deferred execution");
  for (const auto& binding : bindings) {
    TORCH_CHECK(binding->tensorType() == kp::Tensor::TensorTypes::eHost,
                "Deferred Vulkan commands require host-visible buffers; set "
                "TORCH_VULKAN_HOST_VISIBLE_MAX_BYTES to cover your largest tensor");
  }
  std::lock_guard<std::mutex> lock(deferred_mutex_);
  if (!deferred_) {
    deferred_ = mgr_->sequence();
    VulkanAllocator::instance().defer_reuse(true);
  }
  // Serialize dispatches through this command buffer. Device->device visibility
  // is not automatically established between independent compute dispatches.
  deferred_->record<kp::OpAlgoDispatch>(algorithm);
  deferred_->record<kp::OpMemoryBarrier>(
      outputs,
      vk::AccessFlagBits::eShaderWrite,
      vk::AccessFlagBits::eShaderRead,
      vk::PipelineStageFlagBits::eComputeShader,
      vk::PipelineStageFlagBits::eComputeShader);
  pending_outputs_.insert(pending_outputs_.end(),outputs.begin(),outputs.end());
  ++queue_stats_.recorded_ops;
}

void VulkanContext::synchronize() {
  if (!deferred_enabled()) return;
  std::lock_guard<std::mutex> lock(deferred_mutex_);
  if (!deferred_) return;
  // The final coherent-host visibility barrier is essential on mapped eHost.
  deferred_->record<kp::OpMemoryBarrier>(
      pending_outputs_, vk::AccessFlagBits::eShaderWrite,
      vk::AccessFlagBits::eHostRead,
      vk::PipelineStageFlagBits::eComputeShader,
      vk::PipelineStageFlagBits::eHost);
  const auto count = pending_outputs_.size();
  deferred_->eval();
  deferred_.reset();
  pending_outputs_.clear();
  VulkanAllocator::instance().defer_reuse(false);
  ++queue_stats_.submissions;
  queue_stats_.largest_batch = std::max(queue_stats_.largest_batch,
                                         static_cast<uint64_t>(count));
}

VulkanContext::QueueStats VulkanContext::queue_stats() const {
  std::lock_guard<std::mutex> lock(deferred_mutex_);
  return queue_stats_;
}

kp::Manager& VulkanContext::manager() {
  return *mgr_;
}

void VulkanContext::set_shader_dir(const std::string& path) {
  std::lock_guard<std::mutex> lock(mutex_);
  shader_dir_ = path;
  if (!shader_dir_.empty() && shader_dir_.back() != '/') {
    shader_dir_ += '/';
  }
}

std::vector<uint32_t> VulkanContext::load_shader(const std::string& name) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto it = shader_cache_.find(name);
  if (it != shader_cache_.end()) {
    return it->second;
  }

  std::string path = shader_dir_ + name + ".spv";
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file.is_open()) {
    throw std::runtime_error("Failed to load shader: " + path);
  }

  size_t size = file.tellg();
  file.seekg(0);

  std::vector<uint32_t> spirv(size / sizeof(uint32_t));
  file.read(reinterpret_cast<char*>(spirv.data()), size);

  shader_cache_[name] = spirv;
  return spirv;
}

std::string VulkanContext::device_name() const {
  // Report the actual Vulkan device, including software drivers like Lavapipe.
  return std::string(mgr_->getDeviceProperties().deviceName.data());
}

// ---- Algorithm Cache (Phase 4) ----

std::string VulkanContext::make_cache_key(
    const std::string& shader_name,
    const std::vector<std::shared_ptr<kp::Tensor>>& tensors) {
  // Key format: "shader_name:N:ptr0:ptr1:..."
  // Using raw data pointers ensures we match only when the exact same
  // Vulkan buffers are bound. This is safe because allocator-backed tensors
  // have stable pointers for their entire lifetime.
  std::ostringstream oss;
  oss << shader_name << ':' << tensors.size();
  for (const auto& t : tensors) {
    oss << ':' << reinterpret_cast<uintptr_t>(t->rawData());
  }
  return oss.str();
}

std::shared_ptr<kp::Algorithm> VulkanContext::get_or_create_algorithm(
    const std::string& shader_name,
    const std::vector<std::shared_ptr<kp::Tensor>>& tensors,
    const kp::Workgroup& workgroup,
    const std::vector<uint32_t>& push_constants) {

  std::string key = make_cache_key(shader_name, tensors);

  std::lock_guard<std::mutex> lock(mutex_);

  auto it = algo_cache_.find(key);
  if (it != algo_cache_.end()) {
    // Cache hit: reuse existing algorithm.
    // Update mutable state: push constants and workgroup.
    auto& algo = it->second;
    if (!push_constants.empty()) {
      algo->setPushConstants(
          const_cast<void*>(static_cast<const void*>(push_constants.data())),
          static_cast<uint32_t>(push_constants.size()),
          static_cast<uint32_t>(sizeof(uint32_t)));
    }
    algo->setWorkgroup(workgroup, tensors.size() ? tensors[0]->size() : 1);
    stats_.algo_hits++;
    return algo;
  }

  // Cache miss: create new algorithm and cache it.
  // Check shader cache inline (load_shader would deadlock since we hold mutex_).
  auto spirv = shader_cache_.count(shader_name)
      ? shader_cache_[shader_name]
      : std::vector<uint32_t>{};

  if (spirv.empty()) {
    std::string path = shader_dir_ + shader_name + ".spv";
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
      throw std::runtime_error("Failed to load shader: " + path);
    }
    size_t size = file.tellg();
    file.seekg(0);
    spirv.resize(size / sizeof(uint32_t));
    file.read(reinterpret_cast<char*>(spirv.data()), size);
    shader_cache_[shader_name] = spirv;
  }

  auto algo = mgr_->algorithm<float, uint32_t>(
      tensors, spirv, workgroup,
      std::vector<float>{}, push_constants);

  algo_cache_[key] = algo;
  stats_.algo_misses++;
  return algo;
}

std::shared_ptr<kp::Sequence> VulkanContext::acquire_sequence() {
  std::lock_guard<std::mutex> lock(mutex_);

  if (!seq_pool_.empty()) {
    auto seq = std::move(seq_pool_.back());
    seq_pool_.pop_back();
    stats_.seq_reuses++;
    return seq;
  }

  stats_.seq_creates++;
  return mgr_->sequence();
}

void VulkanContext::release_sequence(std::shared_ptr<kp::Sequence> seq) {
  // Sequences in Kompute create their command pool without the RESET bit,
  // so the command buffer cannot be reused after recording+eval.
  // Just let the sequence drop. The main perf win is from Algorithm caching.
  // (Intentionally left as no-op.)
}

VulkanContext::CacheStats VulkanContext::cache_stats() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return stats_;
}

void VulkanContext::clear_algorithm_cache() {
  std::lock_guard<std::mutex> lock(mutex_);
  algo_cache_.clear();
}

} // namespace torch_vulkan
