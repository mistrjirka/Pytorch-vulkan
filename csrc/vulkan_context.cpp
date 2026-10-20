#include "execution_audit.h"
#include "vulkan_context.h"
#include <fstream>
#include <cstdlib>
#include <cerrno>
#include <limits>
#include "vulkan_allocator.h"
#include <cstring>
#include <algorithm>
#include <chrono>
#include <atomic>
#include <c10/util/Exception.h>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace torch_vulkan {
namespace {
using HotClock=std::chrono::steady_clock;
std::atomic<uint64_t> g_algo_hit_ns{0},g_algo_miss_ns{0},g_algo_hit_count{0},g_algo_miss_count{0};
std::atomic<uint64_t> g_enqueue_ns{0},g_enqueue_count{0};
std::atomic<uint64_t> g_alloc_ns{0},g_alloc_fresh_ns{0},g_alloc_reused_ns{0},g_alloc_calls{0},g_alloc_fresh_calls{0},g_alloc_reused_calls{0};
std::atomic<uint64_t> g_alloc_fresh_bytes{0};
uint64_t elapsed_ns(HotClock::time_point begin) {
  return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
    HotClock::now()-begin).count());
}
}
bool host_hotpath_profile_enabled(){
  static bool enabled=[](){auto* v=std::getenv("TORCH_VULKAN_PROFILE_HOST_HOTPATH");
                              return v&&std::strcmp(v,"1")==0;}();
  return enabled;
}
void record_allocator_hotpath(uint64_t nanos,size_t bytes,bool fresh){
  if(!host_hotpath_profile_enabled())return;
  g_alloc_ns.fetch_add(nanos,std::memory_order_relaxed);
  g_alloc_calls.fetch_add(1,std::memory_order_relaxed);
  if(fresh) {
    g_alloc_fresh_ns.fetch_add(nanos,std::memory_order_relaxed);
    g_alloc_fresh_calls.fetch_add(1,std::memory_order_relaxed);
    g_alloc_fresh_bytes.fetch_add(bytes,std::memory_order_relaxed);
  } else {
    g_alloc_reused_ns.fetch_add(nanos,std::memory_order_relaxed);
    g_alloc_reused_calls.fetch_add(1,std::memory_order_relaxed);
  }
}
std::map<std::string,uint64_t> runtime_hotpath_stats(){
 return {{"algo_hit_ns",g_algo_hit_ns.load()},{"algo_miss_ns",g_algo_miss_ns.load()},
  {"algo_hit_count",g_algo_hit_count.load()},{"algo_miss_count",g_algo_miss_count.load()},
  {"enqueue_ns",g_enqueue_ns.load()},{"enqueue_count",g_enqueue_count.load()},
  {"allocator_ns",g_alloc_ns.load()},{"allocator_calls",g_alloc_calls.load()},
  {"allocator_fresh_ns",g_alloc_fresh_ns.load()},{"allocator_fresh_calls",g_alloc_fresh_calls.load()},
  {"allocator_fresh_bytes",g_alloc_fresh_bytes.load()},
  {"allocator_reused_ns",g_alloc_reused_ns.load()},
  {"allocator_reused_calls",g_alloc_reused_calls.load()}};
}


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
  if (!raw_storage_ptr) return;
  audit::host_read(VulkanAllocator::instance().get_kompute_tensor(raw_storage_ptr)->size()*sizeof(float));
  if (!gpu_resident_enabled()) return;
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
    if(!inputs.empty())seq->record<kp::OpTensorSyncDevice>(inputs);
    else if(timed)seq->record<kp::OpMemoryBarrier>(inputs,vk::AccessFlagBits::eShaderWrite,vk::AccessFlagBits::eShaderRead,vk::PipelineStageFlagBits::eComputeShader,vk::PipelineStageFlagBits::eComputeShader);
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
  const bool timed=gpu_timing_enabled();
  auto seq=timed?mgr_->sequence(0,3):mgr_->sequence();
  // In profiling mode retain fixed slots: upload, dependency barrier, dispatch.
  if (!upload.empty()) seq->record<kp::OpTensorSyncDevice>(upload);
  else if(timed)seq->record<kp::OpMemoryBarrier>(upload,vk::AccessFlagBits::eShaderWrite,vk::AccessFlagBits::eShaderRead,vk::PipelineStageFlagBits::eComputeShader,vk::PipelineStageFlagBits::eComputeShader);
  if (timed || !gpu_deps.empty()) seq->record<kp::OpMemoryBarrier>(
    gpu_deps,
    vk::AccessFlagBits::eShaderWrite, vk::AccessFlagBits::eShaderRead,
    vk::PipelineStageFlagBits::eComputeShader,
    vk::PipelineStageFlagBits::eComputeShader);
  seq->record<kp::OpAlgoDispatch>(algo);
  auto start=std::chrono::steady_clock::now();
  seq->eval();
  if(timed){
    auto ts=seq->getTimestamps();
    TORCH_CHECK(ts.size()==4,"resident profiler expects three record operations");
    const double scale=mgr_->getDeviceProperties().limits.timestampPeriod/1e6;
    const double wall=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
    std::lock_guard<std::mutex> lock(mutex_);auto& v=gpu_timing_[shader];++v.samples;v.host_eval_ms+=wall;
    if(ts[1]>=ts[0]&&ts[3]>=ts[2]){v.upload_ms+=(ts[1]-ts[0])*scale;v.shader_ms+=(ts[3]-ts[2])*scale;}
  }
  for (auto& t:upload) alloc.mark_clean(t);
  for (auto& t:outputs) alloc.mark_gpu_dirty(t);
}

void VulkanContext::enqueue_resident_compute(
    const std::shared_ptr<kp::Algorithm>& algo,
    const std::vector<std::shared_ptr<kp::Tensor>>& inputs,
    const std::vector<std::shared_ptr<kp::Tensor>>& outputs) {
  const bool profile=host_hotpath_profile_enabled();
  const auto hot_start=profile?HotClock::now():HotClock::time_point{};
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
  if(profile) {
    g_enqueue_ns.fetch_add(elapsed_ns(hot_start),std::memory_order_relaxed);
    g_enqueue_count.fetch_add(1,std::memory_order_relaxed);
  }
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
      v.pair_first_ms+=(ts[3]-ts[2])*scale;
      v.pair_barrier_ms+=(ts[4]-ts[3])*scale;
      v.pair_second_ms+=(ts[5]-ts[4])*scale;
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

void VulkanContext::submit_deferred_locked() {
  if (!deferred_) return;
  deferred_->record<kp::OpMemoryBarrier>(
      pending_outputs_, vk::AccessFlagBits::eShaderWrite,
      vk::AccessFlagBits::eMemoryRead,
      vk::PipelineStageFlagBits::eComputeShader,
      vk::PipelineStageFlagBits::eAllCommands);
  const auto count = pending_outputs_.size();
  deferred_->evalAsync();
  in_flight_.push_back(std::move(deferred_));
  pending_outputs_.clear();
  ++queue_stats_.submissions;
  queue_stats_.largest_batch=std::max(queue_stats_.largest_batch,uint64_t(count));
}
void VulkanContext::reap_completed_locked() {
  for(auto it=in_flight_.begin();it!=in_flight_.end();) {
    if((*it)->isRunning())(*it)->evalAwait(0);
    if(!(*it)->isRunning())it=in_flight_.erase(it);else ++it;
  }
  if(in_flight_.empty()&&!deferred_)VulkanAllocator::instance().defer_reuse(false);
}
void VulkanContext::synchronize() {
  std::lock_guard<std::mutex> lock(deferred_mutex_);
  submit_deferred_locked();
  for(auto& sequence:in_flight_)if(sequence->isRunning())sequence->evalAwait();
  in_flight_.clear();
  VulkanAllocator::instance().defer_reuse(false);
}
std::shared_ptr<kp::Sequence> VulkanContext::record_event(bool timing) {
  std::lock_guard<std::mutex> lock(deferred_mutex_);
  submit_deferred_locked();
  auto sequence=mgr_->sequence(0,timing?1:0);
  sequence->begin();
  sequence->end();
  sequence->evalAsync();
  in_flight_.push_back(sequence);
  reap_completed_locked();
  return sequence;
}
bool VulkanContext::query_queue() {
  std::lock_guard<std::mutex> lock(deferred_mutex_);
  reap_completed_locked();return !deferred_&&in_flight_.empty();
}
bool VulkanContext::query_event(const std::shared_ptr<kp::Sequence>& sequence) {
  std::lock_guard<std::mutex> lock(deferred_mutex_);
  if(sequence->isRunning())sequence->evalAwait(0);
  bool done=!sequence->isRunning();reap_completed_locked();return done;
}
void VulkanContext::await_event(const std::shared_ptr<kp::Sequence>& sequence) {
  std::lock_guard<std::mutex> lock(deferred_mutex_);
  if(sequence->isRunning())sequence->evalAwait();reap_completed_locked();
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

  const bool profile=host_hotpath_profile_enabled();
  const auto hot_start=profile?HotClock::now():HotClock::time_point{};
  std::string key = make_cache_key(shader_name, tensors);

  std::lock_guard<std::mutex> lock(mutex_);

  auto it = algo_cache_.find(key);
  if (it != algo_cache_.end()) {
    // LRU instead of FIFO: frequently reused buffer tuples survive transient
    // intermediate allocations. Cached descriptors cannot be rewritten while
    // an earlier command using them is in flight; their binding is immutable.
    auto& entry = it->second;
    algo_cache_lru_.splice(algo_cache_lru_.begin(),algo_cache_lru_,entry.lru);
    auto& algo = entry.algorithm;
    if (!push_constants.empty()) {
      algo->setPushConstants(
          const_cast<void*>(static_cast<const void*>(push_constants.data())),
          static_cast<uint32_t>(push_constants.size()),
          static_cast<uint32_t>(sizeof(uint32_t)));
    }
    algo->setWorkgroup(workgroup, tensors.size() ? tensors[0]->size() : 1);
    stats_.algo_hits++;
    if(profile) {
      g_algo_hit_ns.fetch_add(elapsed_ns(hot_start),std::memory_order_relaxed);
      g_algo_hit_count.fetch_add(1,std::memory_order_relaxed);
    }
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

  // A cached Algorithm retains its tensors and descriptor set. Budget
  // distinct underlying VkBuffers, not duplicated bytes for each Algorithm.
  // These are cache residency limits, NOT limits on a live tensor or dispatch.
  static const size_t bound_algo_capacity=[] {
    auto* value=std::getenv("TORCH_VULKAN_BOUND_ALGORITHM_CACHE_CAP");
    if(!value || !*value) return kMaxBoundAlgorithms;
    errno=0;char* end=nullptr;
    auto count=std::strtoull(value,&end,10);
    TORCH_CHECK(errno==0 && end!=value && *end=='\0' &&
                count>=128 && count<=16384,
                "invalid bounded Vulkan algorithm cache capacity");
    return static_cast<size_t>(count);
  }();
  static const uint64_t memory_budget=[] {
    auto* value=std::getenv("TORCH_VULKAN_BOUND_ALGORITHM_CACHE_MB");
    if(!value || !*value) return kMaxBoundCacheBufferBytes;
    errno=0;char* end=nullptr;
    auto count=std::strtoull(value,&end,10);
    TORCH_CHECK(errno==0 && end!=value && *end=='\0' &&
                count<=16384,
                "invalid bounded Vulkan algorithm cache memory budget");
    return static_cast<uint64_t>(count)*1024*1024;
  }();
  std::vector<VkBuffer> unique;
  uint64_t bytes_to_add=0;
  std::unordered_map<VkBuffer,uint64_t> sizes;
  for(const auto& t:tensors) {
    auto desc=t->constructDescriptorBufferInfo();
    const auto buffer=static_cast<VkBuffer>(desc.buffer);
    if(sizes.emplace(buffer,static_cast<uint64_t>(t->size())*sizeof(float)).second) {
      unique.push_back(buffer);
      if(!bound_buffer_refs_.count(buffer)) bytes_to_add+=sizes[buffer];
    }
  }
  // Oversized bindings are still dispatched on the GPU, just not retained
  // indefinitely in the descriptor cache.
  if(memory_budget && bytes_to_add<=memory_budget) {
    while(!algo_cache_.empty() &&
          (algo_cache_.size()>=bound_algo_capacity ||
           bound_buffer_bytes_+bytes_to_add>memory_budget)) {
      evict_bound_algorithm_locked();
      bytes_to_add=0;
      for(auto buffer:unique) if(!bound_buffer_refs_.count(buffer))
        bytes_to_add+=sizes[buffer];
    }
    if(bound_buffer_bytes_+bytes_to_add<=memory_budget) {
      algo_cache_lru_.push_front(key);
      for(auto buffer:unique) {
        auto& entry=bound_buffer_refs_[buffer];
        if(!entry.references) {
          entry.bytes=sizes[buffer];
          bound_buffer_bytes_+=entry.bytes;
        }
        ++entry.references;
      }
      algo_cache_.emplace(key,CachedAlgorithm{algo,algo_cache_lru_.begin(),std::move(unique)});
    }
  }
  stats_.algo_misses++;
  if(profile) {
    g_algo_miss_ns.fetch_add(elapsed_ns(hot_start),std::memory_order_relaxed);
    g_algo_miss_count.fetch_add(1,std::memory_order_relaxed);
  }
  return algo;
}

void VulkanContext::evict_bound_algorithm_locked() {
  TORCH_CHECK(!algo_cache_lru_.empty(),"Empty Vulkan bound algorithm cache");
  auto it=algo_cache_.find(algo_cache_lru_.back());
  TORCH_INTERNAL_ASSERT(it!=algo_cache_.end());
  for(auto buffer:it->second.buffers) {
    auto bound=bound_buffer_refs_.find(buffer);
    TORCH_INTERNAL_ASSERT(bound!=bound_buffer_refs_.end() && bound->second.references);
    if(!--bound->second.references) {
      bound_buffer_bytes_-=bound->second.bytes;
      bound_buffer_refs_.erase(bound);
    }
  }
  algo_cache_.erase(it);
  algo_cache_lru_.pop_back();
  ++stats_.algo_evictions;
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
  result.bound_cached_buffer_bytes = bound_buffer_bytes_;
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
  algo_cache_lru_.clear();
  bound_buffer_refs_.clear();
  bound_buffer_bytes_=0;
}

} // namespace torch_vulkan
