#include "vulkan_context.h"
#include <fstream>
#include <cstdlib>
#include <cerrno>
#include <limits>
#include "vulkan_allocator.h"
#include <cstring>
#include <algorithm>
#include <chrono>
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

bool VulkanContext::gpu_resident_enabled() const {
  static const bool enabled=[](){
    const auto* flag=std::getenv("TORCH_VULKAN_GPU_RESIDENT");
    return flag && std::strcmp(flag,"1")==0;
  }();
  return enabled;
}

void VulkanContext::ensure_host_read(void* raw_storage_ptr) {
  if (!gpu_resident_enabled() || !raw_storage_ptr) return;
  // This is a CPU-read boundary; any deferred shader must finish first.
  synchronize();
  auto tensor=VulkanAllocator::instance().get_kompute_tensor(raw_storage_ptr);
  auto& alloc=VulkanAllocator::instance();
  if (!alloc.needs_download(tensor)) return;
  auto seq=mgr_->sequence();
  seq->record<kp::OpTensorSyncLocal>(
      std::vector<std::shared_ptr<kp::Tensor>>{tensor});
  seq->eval();
  alloc.mark_clean(tensor);
}

void VulkanContext::run_compute(
    const std::shared_ptr<kp::Algorithm>& algo,
    const std::string& shader,
    const std::vector<std::shared_ptr<kp::Tensor>>& inputs,
    const std::vector<std::shared_ptr<kp::Tensor>>& outputs) {
  auto& alloc=VulkanAllocator::instance();
  if (!gpu_resident_enabled()) {
    // Flush older deferred eHost commands before a legacy eager operation.
    synchronize();
    const bool timed = gpu_timing_enabled() &&
        (shader=="mm" || shader=="mm_register" ||
         shader=="addmm" || shader=="addmm_register" ||
         shader.rfind("conv2d_",0)==0 || shader.rfind("batch_norm_",0)==0 || shader.rfind("max_pool2d_",0)==0);
    auto seq = timed ? timed_sequence() : mgr_->sequence();
    seq->record<kp::OpTensorSyncDevice>(inputs);
    seq->record<kp::OpAlgoDispatch>(algo);
    seq->record<kp::OpTensorSyncLocal>(outputs);
    if (timed) eval_timed_sequence(seq,shader);
    else seq->eval();
    return;
  }
  if (deferred_enabled()) {
    enqueue_resident_compute(algo,inputs,outputs);
    return;
  }
  synchronize();
  std::vector<std::shared_ptr<kp::Tensor>> upload, gpu_deps;
  upload.reserve(inputs.size());gpu_deps.reserve(inputs.size());
  for (const auto& t:inputs) {
    if (alloc.needs_upload(t)) upload.push_back(t);
    else if (alloc.gpu_dirty(t)) gpu_deps.push_back(t);
  }
  auto seq=mgr_->sequence();
  if (!upload.empty()) seq->record<kp::OpTensorSyncDevice>(upload);
  if (!gpu_deps.empty()) seq->record<kp::OpMemoryBarrier>(
    gpu_deps,
    vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead,
    vk::PipelineStageFlagBits::eComputeShader,
    vk::PipelineStageFlagBits::eComputeShader);
  seq->record<kp::OpAlgoDispatch>(algo);
  seq->eval();
  for (auto& t:upload) alloc.mark_clean(t);
  for (auto& t:outputs) alloc.mark_gpu_dirty(t);
}

void VulkanContext::enqueue_resident_compute(
    const std::shared_ptr<kp::Algorithm>& algo,
    const std::vector<std::shared_ptr<kp::Tensor>>& inputs,
    const std::vector<std::shared_ptr<kp::Tensor>>& outputs) {
  TORCH_CHECK(gpu_resident_enabled() && deferred_enabled(),
              "enqueue_resident_compute requires GPU residence and deferral");
  auto& alloc=VulkanAllocator::instance();
  std::lock_guard<std::mutex> lock(deferred_mutex_);
  if (!deferred_) {
    deferred_=mgr_->sequence();
    alloc.defer_reuse(true);
  }
  std::vector<std::shared_ptr<kp::Tensor>> upload,deps;
  for (const auto& t:inputs) {
    if (alloc.needs_upload(t)) upload.push_back(t);
    else if (alloc.gpu_dirty(t)) deps.push_back(t);
  }
  if (!upload.empty())deferred_->record<kp::OpTensorSyncDevice>(upload);
  if (!deps.empty())deferred_->record<kp::OpMemoryBarrier>(
      deps,vk::AccessFlagBits::eShaderWrite,vk::AccessFlagBits::eShaderRead,
      vk::PipelineStageFlagBits::eComputeShader,
      vk::PipelineStageFlagBits::eComputeShader);
  deferred_->record<kp::OpAlgoDispatch>(algo);
  deferred_->record<kp::OpMemoryBarrier>(
      outputs,vk::AccessFlagBits::eShaderWrite,vk::AccessFlagBits::eShaderRead,
      vk::PipelineStageFlagBits::eComputeShader,
      vk::PipelineStageFlagBits::eComputeShader);
  pending_outputs_.insert(pending_outputs_.end(),outputs.begin(),outputs.end());
  // GPU writes are logically newer immediately. Actual completion is enforced
  // by synchronize() before host reads or unsupported CPU fallbacks.
  for (const auto& t:upload)alloc.mark_clean(t);
  for (const auto& t:outputs)alloc.mark_gpu_dirty(t);
  ++queue_stats_.recorded_ops;
}

void VulkanContext::run_compute_pair(
    const std::shared_ptr<kp::Algorithm>& first,
    const std::shared_ptr<kp::Algorithm>& second,
    const std::vector<std::shared_ptr<kp::Tensor>>& inputs,
    const std::vector<std::shared_ptr<kp::Tensor>>& intermediate,
    const std::vector<std::shared_ptr<kp::Tensor>>& outputs,
    const std::string& profile_name) {
  if(gpu_resident_enabled() && deferred_enabled()) {
    enqueue_resident_compute(first,inputs,intermediate);
    enqueue_resident_compute(second,intermediate,outputs);
    return;
  }
  synchronize();
  auto& alloc=VulkanAllocator::instance();
  std::vector<std::shared_ptr<kp::Tensor>> upload,deps,download;
  for(const auto& t:inputs) {
    if(!gpu_resident_enabled() || alloc.needs_upload(t)) upload.push_back(t);
    else if(alloc.gpu_dirty(t)) deps.push_back(t);
  }
  if(!gpu_resident_enabled()) download=outputs;
  const bool timed=gpu_timing_enabled();
  auto seq=timed ? mgr_->sequence(0,6) : mgr_->sequence();
  // Kompute rejects empty sync operations. An empty memory barrier is a
  // no-op that keeps timestamp slots stable without uploading scratch data.
  const auto empty_slot=[&]() {
    seq->record<kp::OpMemoryBarrier>(std::vector<std::shared_ptr<kp::Tensor>>{},
        vk::AccessFlagBits::eShaderWrite,vk::AccessFlagBits::eShaderRead,
        vk::PipelineStageFlagBits::eComputeShader,vk::PipelineStageFlagBits::eComputeShader);
  };
  if(!upload.empty()) seq->record<kp::OpTensorSyncDevice>(upload);
  else empty_slot();
  seq->record<kp::OpMemoryBarrier>(deps,
      vk::AccessFlagBits::eShaderWrite,vk::AccessFlagBits::eShaderRead,
      vk::PipelineStageFlagBits::eComputeShader,vk::PipelineStageFlagBits::eComputeShader);
  seq->record<kp::OpAlgoDispatch>(first);
  seq->record<kp::OpMemoryBarrier>(intermediate,
      vk::AccessFlagBits::eShaderWrite,vk::AccessFlagBits::eShaderRead,
      vk::PipelineStageFlagBits::eComputeShader,vk::PipelineStageFlagBits::eComputeShader);
  seq->record<kp::OpAlgoDispatch>(second);
  if(!download.empty()) seq->record<kp::OpTensorSyncLocal>(download);
  else empty_slot();
  const auto start=std::chrono::steady_clock::now();
  seq->eval();
  if(timed) {
    const auto ts=seq->getTimestamps();
    TORCH_CHECK(ts.size()==7,"Vulkan pair profiler expects six operations");
    const double scale=mgr_->getDeviceProperties().limits.timestampPeriod/1e6;
    std::lock_guard<std::mutex> lock(mutex_);
    auto& v=gpu_timing_[profile_name];
    ++v.samples;
    v.host_eval_ms+=std::chrono::duration<double,std::milli>(
        std::chrono::steady_clock::now()-start).count();
    if(std::is_sorted(ts.begin(),ts.end())) {
      v.upload_ms+=(ts[2]-ts[0])*scale;
      v.shader_ms+=(ts[5]-ts[2])*scale;
      v.download_ms+=(ts[6]-ts[5])*scale;
    }
  }
  if(gpu_resident_enabled()) {
    for(const auto& t:upload) alloc.mark_clean(t);
    for(const auto& t:outputs) alloc.mark_gpu_dirty(t);
    for(const auto& t:intermediate) alloc.mark_gpu_dirty(t);
  }
}

void VulkanContext::run_compute_batch(
    const std::vector<std::shared_ptr<kp::Algorithm>>& algorithms,
    const std::vector<std::vector<std::shared_ptr<kp::Tensor>>>& inputs,
    const std::vector<std::vector<std::shared_ptr<kp::Tensor>>>& outputs) {
  TORCH_CHECK(algorithms.size()==inputs.size() && algorithms.size()==outputs.size(),
              "Vulkan batch: size mismatch");
  if (algorithms.empty()) return;
  if (!gpu_resident_enabled()) {
    auto seq=mgr_->sequence();
    for (const auto& v:inputs) seq->record<kp::OpTensorSyncDevice>(v);
    for (const auto& a:algorithms) seq->record<kp::OpAlgoDispatch>(a);
    for (const auto& v:outputs) seq->record<kp::OpTensorSyncLocal>(v);
    seq->eval();
    return;
  }
  if (deferred_enabled()) {
    for (size_t i=0;i<algorithms.size();++i)
      enqueue_resident_compute(algorithms[i],inputs[i],outputs[i]);
    return;
  }
  auto& alloc=VulkanAllocator::instance();
  std::vector<std::shared_ptr<kp::Tensor>> upload,deps;
  for (const auto& vs:inputs) {
    for (const auto& t:vs) {
      if (alloc.needs_upload(t)) upload.push_back(t);
      else if (alloc.gpu_dirty(t)) deps.push_back(t);
    }
  }
  auto seq=mgr_->sequence();
  if (!upload.empty())seq->record<kp::OpTensorSyncDevice>(upload);
  if (!deps.empty())seq->record<kp::OpMemoryBarrier>(
      deps,vk::AccessFlagBits::eShaderWrite,vk::AccessFlagBits::eShaderRead,
      vk::PipelineStageFlagBits::eComputeShader,
      vk::PipelineStageFlagBits::eComputeShader);
  for (const auto& a:algorithms)seq->record<kp::OpAlgoDispatch>(a);
  seq->eval();
  for (const auto& t:upload)alloc.mark_clean(t);
  for (const auto& vs:outputs)
    for (const auto& t:vs)alloc.mark_gpu_dirty(t);
}

bool VulkanContext::gpu_timing_enabled() const {
  static bool enabled=[](){
    const auto* value=std::getenv("TORCH_VULKAN_PROFILE_TIMESTAMPS");
    return value && std::strcmp(value,"1") == 0;
  }();
  return enabled;
}

std::shared_ptr<kp::Sequence> VulkanContext::timed_sequence() {
  return gpu_timing_enabled() ? mgr_->sequence(0, 3) : mgr_->sequence();
}

void VulkanContext::eval_timed_sequence(
    const std::shared_ptr<kp::Sequence>& seq, const std::string& name) {
  auto start=std::chrono::steady_clock::now();
  seq->eval();
  auto elapsed=std::chrono::duration<double,std::milli>(
      std::chrono::steady_clock::now()-start).count();
  if (!gpu_timing_enabled()) return;
  const auto ts=seq->getTimestamps();
  TORCH_CHECK(ts.size()==4,"Vulkan profiler expects three record operations");
  const double ms_per_tick=mgr_->getDeviceProperties().limits.timestampPeriod/1e6;
  std::lock_guard<std::mutex> lock(mutex_);
  auto& v=gpu_timing_[name];
  ++v.samples;
  v.host_eval_ms+=elapsed;
  if (ts[1]>=ts[0] && ts[2]>=ts[1] && ts[3]>=ts[2]) {
    v.upload_ms+=(ts[1]-ts[0])*ms_per_tick;
    v.shader_ms+=(ts[2]-ts[1])*ms_per_tick;
    v.download_ms+=(ts[3]-ts[2])*ms_per_tick;
  }
}

std::map<std::string,VulkanContext::GpuTiming> VulkanContext::gpu_timing_stats() const {
  std::lock_guard<std::mutex> lock(mutex_);
  return gpu_timing_;
}

VulkanContext::~VulkanContext() {
  try { synchronize(); } catch (...) { /* Shutdown must not throw. */ }
#ifdef TORCH_VULKAN_PATCHED_KOMPUTE
  // The cache owns Vulkan pipeline/layout handles. Release while the logical
  // device is still live (mgr_ is destroyed after this destructor body).
  clear_algorithm_cache();
  kp::Algorithm::clearSharedPipelineCache();
#endif
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
  // Mapped eHost needs compute->host visibility; eDevice stays in VRAM and
  // a subsequent explicit readback will provide the transfer barrier.
  if (!gpu_resident_enabled()) {
    deferred_->record<kp::OpMemoryBarrier>(
        pending_outputs_, vk::AccessFlagBits::eShaderWrite,
        vk::AccessFlagBits::eHostRead,
        vk::PipelineStageFlagBits::eComputeShader,
        vk::PipelineStageFlagBits::eHost);
  }
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
  // A mapped data pointer can be recycled for a DIFFERENT VkBuffer. Use the
  // primary VkBuffer handle for correctness; this is NOT a pipeline key.
  std::ostringstream oss;
  oss << shader_name << ':' << tensors.size();
  for (const auto& t : tensors) {
    auto desc = t->constructDescriptorBufferInfo();
    oss << ':' << static_cast<VkBuffer>(desc.buffer)
        << ':' << static_cast<uint64_t>(desc.offset)
        << ':' << static_cast<uint64_t>(desc.range);
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

  if (algo_cache_.size() >= kMaxBoundAlgorithms) {
    algo_cache_.erase(algo_cache_fifo_.front());
    algo_cache_fifo_.pop_front();
    ++stats_.algo_evictions;
  }
  algo_cache_[key] = algo;
  algo_cache_fifo_.push_back(key);
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
  auto result = stats_;
  result.bound_algorithms = algo_cache_.size();
#ifdef TORCH_VULKAN_PATCHED_KOMPUTE
  const auto p = kp::Algorithm::sharedPipelineCacheStats();
  result.pipeline_hits = p.hits;
  result.pipeline_misses = p.misses;
  result.pipeline_evictions = p.evictions;
  result.pipeline_entries = p.entries;
#endif
  return result;
}

void VulkanContext::clear_algorithm_cache() {
  std::lock_guard<std::mutex> lock(mutex_);
  algo_cache_.clear();
  algo_cache_fifo_.clear();
}

} // namespace torch_vulkan
