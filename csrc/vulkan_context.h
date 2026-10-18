#pragma once

#include <kompute/Kompute.hpp>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include <sstream>
#include <map>
#include <deque>

namespace torch_vulkan {

// Singleton managing the Vulkan instance, device, and shader cache.
// Wraps Kompute to hide Vulkan boilerplate.
class VulkanContext {
public:
  static VulkanContext& instance();

  kp::Manager& manager();

  // Load a pre-compiled SPIR-V shader from the shaders/ directory.
  // Returns the raw SPIR-V bytecode. Caches after first load.
  std::vector<uint32_t> load_shader(const std::string& name);

  // Set the directory where .spv files live.
  void set_shader_dir(const std::string& path);

  std::string device_name() const;

  // Experimental eager, GPU-resident execution. Device-local intermediates
  // remain on the GPU until a real CPU read/fallback requires the host mirror.
  bool gpu_resident_enabled() const;
  void ensure_host_read(void* raw_storage_ptr);
  void run_compute(const std::shared_ptr<kp::Algorithm>& algorithm,
                   const std::string& shader,
                   const std::vector<std::shared_ptr<kp::Tensor>>& inputs,
                   const std::vector<std::shared_ptr<kp::Tensor>>& outputs);

  void enqueue_resident_compute(
      const std::shared_ptr<kp::Algorithm>& algorithm,
      const std::vector<std::shared_ptr<kp::Tensor>>& inputs,
      const std::vector<std::shared_ptr<kp::Tensor>>& outputs);

  // Two dependent kernels with a GPU-only intermediate and one fence.
  // Unlike run_compute_batch (independent updates), this inserts a RAW barrier.
  void run_compute_pair(
      const std::shared_ptr<kp::Algorithm>& first,
      const std::shared_ptr<kp::Algorithm>& second,
      const std::vector<std::shared_ptr<kp::Tensor>>& inputs,
      const std::vector<std::shared_ptr<kp::Tensor>>& intermediate,
      const std::vector<std::shared_ptr<kp::Tensor>>& outputs,
      const std::string& profile_name);

  void run_compute_batch(
      const std::vector<std::shared_ptr<kp::Algorithm>>& algorithms,
      const std::vector<std::vector<std::shared_ptr<kp::Tensor>>>& inputs,
      const std::vector<std::vector<std::shared_ptr<kp::Tensor>>>& outputs);

  // Experimental synchronous-to-deferred execution prototype. Only supported
  // on Kompute eHost (coherent mapped) tensors, so no staging copies are used.
  bool deferred_enabled() const;
  bool can_enqueue(const std::vector<std::shared_ptr<kp::Tensor>>& bindings) const;
  void enqueue_compute(const std::shared_ptr<kp::Algorithm>& algorithm,
                       const std::vector<std::shared_ptr<kp::Tensor>>& bindings,
                       const std::vector<std::shared_ptr<kp::Tensor>>& outputs);
  void synchronize();  // wait for all queued work before any host CPU access
  struct QueueStats {
    uint64_t submissions=0, recorded_ops=0, largest_batch=0;
  };
  QueueStats queue_stats() const;

  // Optional GPU timestamp instrumentation for one-off comparative profiling.
  // Every synchronous sequence timestamps H2D, dispatch and D2H. Disabled
  // by default because query pools add instrumentation overhead.
  bool gpu_timing_enabled() const;
  std::shared_ptr<kp::Sequence> timed_sequence();
  void eval_timed_sequence(const std::shared_ptr<kp::Sequence>& sequence,
                           const std::string& name);
  struct GpuTiming {
    uint64_t samples = 0;
    double upload_ms = 0, shader_ms = 0, download_ms = 0;
    double host_eval_ms = 0;
  };
  std::map<std::string,GpuTiming> gpu_timing_stats() const;

  // ---- Algorithm Cache (Phase 4) ----
  // Get a cached algorithm or create a new one. The cache key is built from
  // the shader name plus the raw data pointers of each tensor -- so the same
  // algorithm object is returned when the exact same shader and Vulkan buffers
  // are dispatched again. Immutable pipeline objects are shared separately by
  // the pinned Kompute patch even if this binding-specific cache misses.
  //
  // IMPORTANT: the caller must NOT destroy the returned algorithm.
  // The cache owns it.
  std::shared_ptr<kp::Algorithm> get_or_create_algorithm(
      const std::string& shader_name,
      const std::vector<std::shared_ptr<kp::Tensor>>& tensors,
      const kp::Workgroup& workgroup,
      const std::vector<uint32_t>& push_constants);

  // Get a reusable Sequence from a small pool instead of creating a new one
  // each call. The sequence is cleared before return.
  std::shared_ptr<kp::Sequence> acquire_sequence();

  // Return a sequence to the pool after eval completes.
  void release_sequence(std::shared_ptr<kp::Sequence> seq);

  // Report cache statistics (for benchmarking).
  struct CacheStats {
    uint64_t algo_hits = 0;
    uint64_t algo_misses = 0;
    uint64_t seq_reuses = 0;
    uint64_t seq_creates = 0;
    uint64_t algo_evictions = 0;
    uint64_t bound_algorithms = 0;
    uint64_t pipeline_hits = 0;
    uint64_t pipeline_misses = 0;
    uint64_t pipeline_evictions = 0;
    uint64_t pipeline_entries = 0;
  };
  CacheStats cache_stats() const;

  // Clear all cached algorithms (e.g. when tensors are freed).
  void clear_algorithm_cache();

private:
  VulkanContext();
  ~VulkanContext();
  VulkanContext(const VulkanContext&) = delete;
  VulkanContext& operator=(const VulkanContext&) = delete;

  // Binding-specific key based on actual VkBuffer identities, not mapped
  // staging pointers. Pipeline caching itself is independent of this key.
  static std::string make_cache_key(
      const std::string& shader_name,
      const std::vector<std::shared_ptr<kp::Tensor>>& tensors);

  mutable std::mutex deferred_mutex_;
  std::shared_ptr<kp::Sequence> deferred_;
  std::vector<std::shared_ptr<kp::Tensor>> pending_outputs_;
  QueueStats queue_stats_;

  std::unique_ptr<kp::Manager> mgr_;
  std::string shader_dir_;
  std::unordered_map<std::string, std::vector<uint32_t>> shader_cache_;

  // Algorithm cache: key -> cached algorithm
  std::unordered_map<std::string, std::shared_ptr<kp::Algorithm>> algo_cache_;
  // Avoid retaining every historical tensor descriptor set indefinitely.
  // In-flight OpAlgoDispatch objects own their algorithms independently.
  std::deque<std::string> algo_cache_fifo_;
  static constexpr size_t kMaxBoundAlgorithms = 512;

  // Sequence pool: reusable sequences
  std::vector<std::shared_ptr<kp::Sequence>> seq_pool_;

  mutable CacheStats stats_;
  mutable std::mutex mutex_;
  std::map<std::string,GpuTiming> gpu_timing_;
};

} // namespace torch_vulkan
