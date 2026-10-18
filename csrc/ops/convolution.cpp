// Generic FP32 NCHW convolution with capability-gated forward/backward
// implementations. Direct shaders remain available for reference and fallback.
#include "../vulkan_context.h"
#include "../vulkan_allocator.h"

#include <ATen/Functions.h>
#include <c10/util/Exception.h>
#include <torch/library.h>
#include <cstdint>
#include <algorithm>
#include <limits>
#include <vector>
#include <cstdlib>
#include <cstring>
#include <array>
#include <chrono>
#include <deque>
#include <map>
#include <mutex>
#include <sstream>
#include <unordered_map>

namespace torch_vulkan::ops {

using TensorBuffer = std::shared_ptr<kp::Tensor>;

// Physical capabilities are immutable for the lifetime of our single-device
// context. Query the driver once, not once per candidate or cache lookup.
static const vk::PhysicalDeviceProperties& conv_device_properties() {
  static const auto properties=VulkanContext::instance().manager().getDeviceProperties();
  return properties;
}


static TensorBuffer buffer(const at::Tensor& t) {
  TORCH_CHECK(t.device().type() == c10::DeviceType::PrivateUse1,
              "Vulkan convolution requires Vulkan tensors");
  TORCH_CHECK(t.is_contiguous() && t.storage_offset() == 0,
              "Vulkan convolution currently requires contiguous, unoffset tensors");
  TORCH_CHECK(t.scalar_type() == at::kFloat, "Vulkan convolution supports FP32 only");
  return VulkanAllocator::instance().get_kompute_tensor(t.data_ptr());
}

struct ConvParams {
  uint32_t n, cin, cout, h, w, kh, kw, oh, ow;
  uint32_t sh, sw, ph, pw, dh, dw, groups;

  std::vector<uint32_t> as_push_constants() const {
    return {n,cin,cout,h,w,kh,kw,oh,ow,sh,sw,ph,pw,dh,dw,groups};
  }
};

static uint32_t checked_uint(int64_t x, const char* name) {
  TORCH_CHECK(x >= 0 && x <= std::numeric_limits<uint32_t>::max(),
              "Vulkan convolution: invalid ", name, "=", x);
  return static_cast<uint32_t>(x);
}

// Normalize layouts on the GPU. Contiguous unoffset inputs need no copy.
// Bind the base allocation, never an interior data_ptr from an offset view.
static at::Tensor packed(const at::Tensor& src) {
  TORCH_CHECK(src.device().type()==c10::DeviceType::PrivateUse1 &&
              src.scalar_type()==at::kFloat,
              "Vulkan convolution requires FP32 Vulkan tensors");
  if(src.is_contiguous() && src.storage_offset()==0) return src;
  TORCH_CHECK(src.dim()<=4,"Vulkan convolution packing supports at most 4 dimensions");
  std::vector<uint32_t> push={checked_uint(src.numel(),"elements"),
                             checked_uint(src.storage_offset(),"storage offset"),
                             1,1,1,1,0,0,0,0};
  uint64_t last=src.storage_offset();
  for(int64_t d=0;d<src.dim();++d) {
    push[2+d]=checked_uint(src.size(d),"dimension");
    push[6+d]=checked_uint(src.stride(d),"stride");
    if(src.size(d)>0) last+=static_cast<uint64_t>(src.size(d)-1)*src.stride(d);
  }
  TORCH_CHECK(last<=UINT32_MAX,"Vulkan convolution view indexing overflow");
  auto dst=at::empty(src.sizes(),src.options());
  if(src.numel()==0) return dst;
  auto input=VulkanAllocator::instance().get_kompute_tensor(src.storage().data_ptr().get());
  auto output=buffer(dst);
  auto& ctx=VulkanContext::instance();
  const auto& limits=conv_device_properties().limits;
  const uint64_t count=(static_cast<uint64_t>(src.numel())+127)/128;
  const uint64_t gx=std::min<uint64_t>(count,limits.maxComputeWorkGroupCount[0]);
  const uint64_t gy=(count+gx-1)/gx;
  TORCH_CHECK(limits.maxComputeWorkGroupInvocations>=128 &&
              limits.maxComputeWorkGroupSize[0]>=128 && gy<=limits.maxComputeWorkGroupCount[1],
              "Vulkan convolution packing exceeds device limits");
  const std::vector<TensorBuffer> bindings={input,output};
  for(const auto& binding:bindings)
    TORCH_CHECK(binding->constructDescriptorBufferInfo().range<=limits.maxStorageBufferRange,
                "Vulkan convolution packing exceeds storage buffer range");
  auto algo=ctx.get_or_create_algorithm("conv2d_pack",bindings,
      kp::Workgroup({static_cast<uint32_t>(gx),static_cast<uint32_t>(gy),1}),push);
  if(ctx.can_enqueue(bindings)) ctx.enqueue_compute(algo,bindings,{output});
  else ctx.run_compute(algo,"conv2d_pack",{input},{output});
  return dst;
}

static ConvParams convolution_params(const at::Tensor& input,
                                     const at::Tensor& weight,
                                     c10::SymIntArrayRef stride,
                                     c10::SymIntArrayRef padding,
                                     c10::SymIntArrayRef dilation,
                                     bool transposed,
                                     c10::SymIntArrayRef output_padding,
                                     c10::SymInt groups) {
  TORCH_CHECK(!transposed, "Vulkan ConvTranspose is not implemented yet");
  TORCH_CHECK(input.dim() == 4 && weight.dim() == 4,
              "Vulkan convolution supports NCHW tensors only");
  TORCH_CHECK(stride.size() == 2 && padding.size() == 2 &&
              dilation.size() == 2 && output_padding.size() == 2,
              "Vulkan convolution requires two spatial dimensions");
  TORCH_CHECK(output_padding[0].expect_int() == 0 &&
              output_padding[1].expect_int() == 0,
              "Vulkan convolution: output_padding requires transposed convolution");
  const int64_t n = input.size(0), cin = input.size(1),
      cout = weight.size(0), h = input.size(2), w = input.size(3),
      kh = weight.size(2), kw = weight.size(3),
      sh = stride[0].expect_int(), sw = stride[1].expect_int(),
      ph = padding[0].expect_int(), pw = padding[1].expect_int(),
      dh = dilation[0].expect_int(), dw = dilation[1].expect_int(),
      group_count = groups.expect_int();
  TORCH_CHECK(n > 0 && cin > 0 && cout > 0 && h > 0 && w > 0 &&
              kh > 0 && kw > 0 && sh > 0 && sw > 0 &&
              dh > 0 && dw > 0 && ph >= 0 && pw >= 0 &&
              group_count > 0 && cin % group_count == 0 &&
              cout % group_count == 0 &&
              weight.size(1) == cin / group_count,
              "Vulkan convolution: invalid shapes or parameters");
  // Shaders use signed coordinates for padding. Bound operands before host
  // products too, so large user scalars cannot overflow shape arithmetic.
  for(const auto value:{h,w,kh,kw,sh,sw,ph,pw,dh,dw})
    TORCH_CHECK(value<=INT32_MAX,"Vulkan convolution: spatial parameters exceed int32 coordinates");
  TORCH_CHECK(dh*(kh-1)<=INT32_MAX && dw*(kw-1)<=INT32_MAX,
              "Vulkan convolution: effective filter exceeds int32 coordinates");
  TORCH_CHECK(h + 2*ph >= dh*(kh-1) + 1 &&
              w + 2*pw >= dw*(kw-1) + 1,
              "Vulkan convolution: effective kernel exceeds padded input");
  const int64_t oh = (h + 2*ph - dh*(kh-1) - 1) / sh + 1;
  const int64_t ow = (w + 2*pw - dw*(kw-1) - 1) / sw + 1;
  TORCH_CHECK(oh > 0 && ow > 0, "Vulkan convolution: empty output");
  TORCH_CHECK((oh-1)*sh+dh*(kh-1)<=INT32_MAX &&
              (ow-1)*sw+dw*(kw-1)<=INT32_MAX &&
              h-1+ph<=INT32_MAX && w-1+pw<=INT32_MAX,
              "Vulkan convolution: spatial indexing exceeds int32 coordinates");
  return {checked_uint(n,"N"),checked_uint(cin,"Cin"),checked_uint(cout,"Cout"),
          checked_uint(h,"H"),checked_uint(w,"W"),
          checked_uint(kh,"Kh"),checked_uint(kw,"Kw"),
          checked_uint(oh,"Oh"),checked_uint(ow,"Ow"),
          checked_uint(sh,"stride_h"),checked_uint(sw,"stride_w"),
          checked_uint(ph,"padding_h"),checked_uint(pw,"padding_w"),
          checked_uint(dh,"dilation_h"),checked_uint(dw,"dilation_w"),
          checked_uint(group_count,"groups")};
}


// A candidate is a reusable implementation, not a model or an exact tensor size.
// Shapes stay runtime constants; only the finite tile menu changes GPU code.
enum class ConvPhase { Forward, Input, Weight, Bias };
enum class ConvKind { Direct, Depthwise, Tiled, Reduce };
struct ConvTile { const char* name; uint32_t m, n, k, rm=1, rn=1; };
// Output tiles and workgroup sizes are distinct: register blocking computes
// multiple outputs per invocation without requiring larger workgroups.
static constexpr ConvTile kTiles[]={{"base",8,16,16},{"wide",8,32,16},
    {"tall",16,16,16},{"register",32,32,16,2,2}};
static constexpr int kTileCount=sizeof(kTiles)/sizeof(kTiles[0]);
struct ConvCandidate {
  ConvKind kind=ConvKind::Direct;
  int tile=0;
  bool operator==(const ConvCandidate& rhs) const { return kind==rhs.kind && tile==rhs.tile; }
};
static const char* phase_name(ConvPhase phase) {
  return std::array<const char*,4>{"forward","dinput","dweight","dbias"}[static_cast<int>(phase)];
}
static std::string candidate_name(ConvCandidate c) {
  switch(c.kind) {
    case ConvKind::Tiled: return std::string("tiled_")+kTiles[c.tile].name;
    case ConvKind::Depthwise: return "depthwise";
    case ConvKind::Reduce: return "reduce";
    default: return "direct";
  }
}
static uint64_t div_up(uint64_t x,uint64_t y) { return (x+y-1)/y; }
static uint64_t conv_count(const ConvParams& p,ConvPhase phase) {
  switch(phase) {
    case ConvPhase::Forward: return uint64_t(p.n)*p.cout*p.oh*p.ow;
    case ConvPhase::Input: return uint64_t(p.n)*p.cin*p.h*p.w;
    case ConvPhase::Weight: return uint64_t(p.cout)*(p.cin/p.groups)*p.kh*p.kw;
    default: return p.cout;
  }
}
static uint64_t split_count(const ConvParams& p) { return div_up(uint64_t(p.n)*p.oh*p.ow,256); }
static uint64_t workspace_bytes(const ConvParams& p,ConvPhase phase,ConvCandidate c) {
  if(phase!=ConvPhase::Weight || c.kind!=ConvKind::Tiled || split_count(p)==1) return 0;
  return conv_count(p,phase)*split_count(p)*sizeof(float);
}
static uint64_t workspace_limit() {
  const char* value=std::getenv("TORCH_VULKAN_CONV_WORKSPACE_MB");
  if(!value || !*value) return 64ull*1024*1024;
  char* end=nullptr;
  const auto mb=std::strtoull(value,&end,10);
  TORCH_CHECK(*value!='-' && end!=value && *end=='\0' && mb<=4096,
      "TORCH_VULKAN_CONV_WORKSPACE_MB must be an integer in [0,4096]");
  return mb*1024*1024;
}
static kp::Workgroup linear_grid(uint64_t groups,const vk::PhysicalDeviceLimits& limits) {
  const uint64_t gx=std::min<uint64_t>(groups,limits.maxComputeWorkGroupCount[0]);
  return kp::Workgroup({static_cast<uint32_t>(gx),static_cast<uint32_t>(div_up(groups,gx)),1});
}
static kp::Workgroup candidate_grid(const ConvParams& p,ConvPhase phase,ConvCandidate c,
                                    const vk::PhysicalDeviceLimits& limits) {
  if(c.kind==ConvKind::Tiled) {
    const auto t=kTiles[c.tile];
    const uint64_t rows=phase==ConvPhase::Weight ? uint64_t(p.cin/p.groups)*p.kh*p.kw :
        uint64_t(p.n)*(phase==ConvPhase::Input ? uint64_t(p.h)*p.w : uint64_t(p.oh)*p.ow);
    const uint32_t channels=phase==ConvPhase::Input ? p.cin/p.groups : p.cout/p.groups;
    return kp::Workgroup({static_cast<uint32_t>(div_up(rows,t.n)),
        static_cast<uint32_t>(div_up(channels,t.m)*p.groups),
        phase==ConvPhase::Weight ? static_cast<uint32_t>(split_count(p)) : 1u});
  }
  return linear_grid(c.kind==ConvKind::Reduce ? conv_count(p,phase) : div_up(conv_count(p,phase),128),limits);
}
// Eligibility is separate from ranking. Every fallback also checks its grid,
// storage/indexing limits and workgroup requirements before being selected.
static bool applicable(const ConvParams& p,ConvPhase phase,ConvCandidate c,uint64_t budget) {
  const auto& limits=conv_device_properties().limits;
  if(c.kind==ConvKind::Depthwise && (phase!=ConvPhase::Forward || p.groups!=p.cin)) return false;
  if(c.kind==ConvKind::Reduce && phase!=ConvPhase::Weight && phase!=ConvPhase::Bias) return false;
  if(c.kind==ConvKind::Tiled && phase==ConvPhase::Bias) return false;
  if(c.kind==ConvKind::Tiled) {
    const auto t=kTiles[c.tile];
    const auto local_x=t.n/t.rn,local_y=t.m/t.rm;
    if(limits.maxComputeWorkGroupInvocations<local_x*local_y || limits.maxComputeWorkGroupSize[0]<local_x ||
       limits.maxComputeWorkGroupSize[1]<local_y || limits.maxComputeSharedMemorySize<(t.m+t.n)*t.k*4) return false;
  } else if(limits.maxComputeWorkGroupInvocations<128 || limits.maxComputeWorkGroupSize[0]<128 ||
            (c.kind==ConvKind::Reduce && limits.maxComputeSharedMemorySize<512)) return false;
  const auto bytes=workspace_bytes(p,phase,c);
  if(bytes>budget || bytes>limits.maxStorageBufferRange || bytes/4>UINT32_MAX) return false;
  const auto grid=candidate_grid(p,phase,c,limits);
  for(int axis=0;axis<3;++axis)
    if(!grid[axis] || grid[axis]>limits.maxComputeWorkGroupCount[axis]) return false;
  if(bytes) {
    const auto finalize=linear_grid(div_up(conv_count(p,phase),128),limits);
    if(finalize[1]>limits.maxComputeWorkGroupCount[1] || limits.maxComputeWorkGroupSize[0]<128) return false;
  }
  return true;
}
static std::vector<ConvCandidate> candidates(const ConvParams& p,ConvPhase phase,uint64_t budget) {
  std::vector<ConvCandidate> result;
  for(const auto kind:{ConvKind::Direct,ConvKind::Depthwise,ConvKind::Reduce}) {
    const ConvCandidate c{kind,0};
    if(applicable(p,phase,c,budget)) result.push_back(c);
  }
  for(int tile=0;tile<kTileCount;++tile) {
    const ConvCandidate c{ConvKind::Tiled,tile};
    if(applicable(p,phase,c,budget)) result.push_back(c);
  }
  TORCH_CHECK(!result.empty(),"Vulkan convolution has no compatible ",phase_name(phase),
      " implementation for this device/geometry/workspace budget");
  return result;
}
struct ConvControls { std::string algorithm; int tile=-1; bool tune=false; uint64_t workspace=0; };
static ConvControls controls(ConvPhase phase) {
  ConvControls c;
  const char* name=phase==ConvPhase::Forward ? "TORCH_VULKAN_CONV_ALGO" : "TORCH_VULKAN_CONV_BACKWARD_ALGO";
  const char* value=std::getenv(name);
  c.algorithm=value && *value ? value : "auto";
  TORCH_CHECK(c.algorithm=="auto" || c.algorithm=="direct" ||
      (phase==ConvPhase::Forward ? c.algorithm=="tiled" || c.algorithm=="depthwise" : c.algorithm=="optimized"),
      name," has an invalid algorithm (forward: auto/direct/tiled/depthwise; backward: auto/direct/optimized)");
  const char* tile=std::getenv("TORCH_VULKAN_CONV_TILE");
  if(tile && *tile && std::strcmp(tile,"auto")) {
    for(int i=0;i<kTileCount;++i) if(!std::strcmp(tile,kTiles[i].name)) c.tile=i;
    TORCH_CHECK(c.tile>=0,"TORCH_VULKAN_CONV_TILE must be auto/base/wide/tall/register");
  }
  const char* tune=std::getenv("TORCH_VULKAN_CONV_AUTOTUNE");
  TORCH_CHECK(!tune || !*tune || !std::strcmp(tune,"0") || !std::strcmp(tune,"1"),
      "TORCH_VULKAN_CONV_AUTOTUNE must be 0 or 1");
  c.tune=tune && !std::strcmp(tune,"1") && c.algorithm=="auto" && c.tile<0;
  c.workspace=workspace_limit();
  return c;
}
static ConvCandidate heuristic(const ConvParams& p,ConvPhase phase,const ConvControls& ctl,
                                const std::vector<ConvCandidate>& available) {
  auto has=[&](ConvCandidate c) { return std::find(available.begin(),available.end(),c)!=available.end(); };
  ConvCandidate preferred;
  const uint64_t rows=uint64_t(p.n)*p.oh*p.ow, cols=uint64_t(p.cin/p.groups)*p.kh*p.kw;
  const ConvCandidate tiled{ConvKind::Tiled,ctl.tile<0 ? 0 : ctl.tile};
  bool strict=false;
  if(ctl.algorithm=="direct") { preferred={ConvKind::Direct,0}; strict=true; }
  else if(ctl.algorithm=="depthwise") { preferred={ConvKind::Depthwise,0}; strict=true; }
  else if(ctl.algorithm=="tiled" || (ctl.algorithm=="optimized" && phase==ConvPhase::Input) ||
          (ctl.algorithm=="optimized" && phase==ConvPhase::Weight && ctl.tile>=0)) {
    preferred=tiled; strict=true;
  } else if(phase==ConvPhase::Forward) {
    if(p.groups==p.cin && p.cin>=8 && has({ConvKind::Depthwise,0})) preferred={ConvKind::Depthwise,0};
    else if(p.cout/p.groups>=8 && cols>=16 && rows>=32) preferred=tiled;
  } else if(phase==ConvPhase::Input) {
    if(p.cin/p.groups>=2 && p.cout/p.groups>=8 && uint64_t(p.n)*p.h*p.w>=32) preferred=tiled;
  } else {
    if(phase==ConvPhase::Weight && p.cout/p.groups>=8 && cols>=16 &&
        (ctl.algorithm=="optimized" || cols*p.cout>4096) && has(tiled)) preferred=tiled;
    else if(rows>=128 || ctl.algorithm=="optimized") preferred={ConvKind::Reduce,0};
  }
  if(has(preferred)) return preferred;
  TORCH_CHECK(!strict,"Forced Vulkan convolution ",phase_name(phase)," candidate ",
      candidate_name(preferred)," exceeds device limits, workspace budget, or operator applicability");
  // Try a smaller compatible tile, then reduction/direct. Never make a model
  // unsupported merely because the preferred performance candidate cannot run.
  if(preferred.kind==ConvKind::Tiled) {
    if(has({ConvKind::Tiled,0})) return {ConvKind::Tiled,0};
    if(has({ConvKind::Reduce,0})) return {ConvKind::Reduce,0};
  }
  if(has({ConvKind::Direct,0})) return {ConvKind::Direct,0};
  return available.front();
}
static std::string shader_name(ConvPhase phase,ConvCandidate c) {
  static const char* direct[]={"conv2d_forward","conv2d_backward_input","conv2d_backward_weight","conv2d_backward_bias"};
  static const char* tiled[]={"conv2d_tiled","conv2d_backward_input_tiled","conv2d_backward_weight_tiled"};
  if(c.kind==ConvKind::Depthwise) return "conv2d_depthwise";
  if(c.kind==ConvKind::Reduce) return phase==ConvPhase::Weight ? "conv2d_backward_weight_reduce" : "conv2d_backward_bias_reduce";
  if(c.kind==ConvKind::Tiled) return std::string(tiled[static_cast<int>(phase)])+(c.tile ? std::string("_")+kTiles[c.tile].name : "");
  return direct[static_cast<int>(phase)];
}
static void execute_candidate(const ConvParams& p,ConvPhase phase,ConvCandidate c,
    const std::vector<TensorBuffer>& inputs,const at::Tensor& result,at::Tensor& scratch) {
  auto& ctx=VulkanContext::instance();
  const auto& limits=conv_device_properties().limits;
  const auto out=buffer(result);
  const auto name=shader_name(phase,c);
  std::vector<TensorBuffer> bindings(inputs);
  if(workspace_bytes(p,phase,c)) {
    if(!scratch.defined()) scratch=at::empty({static_cast<int64_t>(split_count(p)),result.numel()},result.options());
    const auto tmp=buffer(scratch);
    bindings.push_back(tmp);
    auto first=ctx.get_or_create_algorithm(name,bindings,candidate_grid(p,phase,c,limits),p.as_push_constants());
    auto second=ctx.get_or_create_algorithm("conv2d_backward_weight_finalize",{tmp,out},
        linear_grid(div_up(result.numel(),128),limits),p.as_push_constants());
    ctx.run_compute_pair(first,second,inputs,{tmp},{out},std::string("conv2d_backward_weight_split_")+kTiles[c.tile].name);
  } else {
    bindings.push_back(out);
    auto algo=ctx.get_or_create_algorithm(name,bindings,candidate_grid(p,phase,c,limits),p.as_push_constants());
    if(ctx.can_enqueue(bindings)) ctx.enqueue_compute(algo,bindings,{out});
    else ctx.run_compute(algo,name,inputs,{out});
  }
}
// Plans contain only implementation IDs, never tensors, bindings or pointers.
// Process-local storage inherently scopes them to this loaded shader revision.
static std::mutex plan_mutex;
static std::unordered_map<std::string,ConvCandidate> plan_cache;
static std::deque<std::string> plan_fifo;
static std::map<std::string,uint64_t> plan_stats;
static constexpr size_t kMaxConvPlans=256;
std::map<std::string,uint64_t> conv_dispatch_stats() {
  std::lock_guard<std::mutex> lock(plan_mutex);
  auto result=plan_stats;
  for(const auto* key:{"hits","misses","tunes","evictions","candidates_timed"}) result[key]+=0;
  result["entries"]=plan_cache.size();
  return result;
}
void clear_conv_dispatch_cache() {
  std::lock_guard<std::mutex> lock(plan_mutex);
  plan_cache.clear(); plan_fifo.clear(); plan_stats.clear();
}
static std::string plan_key(const ConvParams& p,ConvPhase phase,uint64_t budget,
    const std::vector<TensorBuffer>& inputs,const at::Tensor& result) {
  auto& ctx=VulkanContext::instance();
  const auto& props=conv_device_properties();
  std::ostringstream key;
  key<<props.vendorID<<':'<<props.deviceID<<':'<<props.driverVersion<<':'<<static_cast<int>(phase)<<':'<<budget
     <<':'<<ctx.gpu_resident_enabled()<<':'<<ctx.deferred_enabled()<<':'<<ctx.gpu_timing_enabled();
  // All inputs are normalized FP32 NCHW with zero offset before this point.
  // The descriptor includes exact shape, groups, stride, padding and dilation.
  for(auto v:p.as_push_constants()) key<<':'<<v;
  // Memory class changes transfer/queue behavior without changing semantics.
  // Include classes, never allocation identities or buffer addresses.
  for(const auto& input:inputs) key<<':'<<static_cast<int>(input->tensorType());
  key<<':'<<static_cast<int>(buffer(result)->tensorType());
  return key.str();
}
static void dispatch(const ConvParams& p,ConvPhase phase,const std::vector<TensorBuffer>& inputs,
                     const at::Tensor& result) {
  auto& ctx=VulkanContext::instance();
  const auto& limits=conv_device_properties().limits;
  for(auto direction:{ConvPhase::Forward,ConvPhase::Input,ConvPhase::Weight})
    TORCH_CHECK(conv_count(p,direction)<=UINT32_MAX,"Vulkan convolution: input/weight/output indexing overflow");
  for(const auto& binding:inputs)
    TORCH_CHECK(binding->constructDescriptorBufferInfo().range<=limits.maxStorageBufferRange,
                "Vulkan convolution exceeds physical storage buffer range");
  TORCH_CHECK(buffer(result)->constructDescriptorBufferInfo().range<=limits.maxStorageBufferRange,
              "Vulkan convolution exceeds physical storage buffer range");
  const auto ctl=controls(phase);
  const auto available=candidates(p,phase,ctl.workspace);
  auto selected=heuristic(p,phase,ctl,available);
  at::Tensor scratch;
  if(ctl.tune) {
    const auto key=plan_key(p,phase,ctl.workspace,inputs,result);
    bool found=false;
    {
      std::lock_guard<std::mutex> lock(plan_mutex);
      const auto it=plan_cache.find(key);
      if(it!=plan_cache.end() && std::find(available.begin(),available.end(),it->second)!=available.end()) {
        selected=it->second; found=true; ++plan_stats["hits"];
      } else ++plan_stats["misses"];
    }
    if(!found) {
      // Bound tuning cost: very large scalar-direct candidates are excluded
      // when an optimized candidate exists. This is a shortlist, not exhaustive search.
      std::vector<ConvCandidate> shortlist;
      const long double work=static_cast<long double>(conv_count(p,ConvPhase::Forward))*(p.cin/p.groups)*p.kh*p.kw;
      for(const auto c:available)
        if(c.kind!=ConvKind::Direct || work<=8000000 || c==selected) shortlist.push_back(c);
      ctx.synchronize();
      const auto trial=[&](ConvCandidate c) {
        // Include candidate-specific scratch allocation and release. Input
        // packing/output allocation are shared by all candidates and excluded.
        at::Tensor trial_scratch;
        execute_candidate(p,phase,c,inputs,result,trial_scratch);
        ctx.synchronize();
      };
      for(auto c:shortlist) trial(c);
      std::vector<std::array<double,5>> samples(shortlist.size());
      for(int repeat=0;repeat<5;++repeat) for(size_t j=0;j<shortlist.size();++j) {
        const auto i=(j+repeat)%shortlist.size();
        const auto start=std::chrono::steady_clock::now();
        trial(shortlist[i]);
        samples[i][repeat]=std::chrono::duration<double,std::nano>(std::chrono::steady_clock::now()-start).count();
      }
      for(auto& sample:samples) std::sort(sample.begin(),sample.end());
      const auto incumbent=std::find(shortlist.begin(),shortlist.end(),selected)-shortlist.begin();
      double best=samples[incumbent][2];
      for(size_t i=0;i<shortlist.size();++i) {
        // Keep the heuristic for marginal differences dominated by timing noise.
        if(samples[i][2]<best*0.95) { best=samples[i][2]; selected=shortlist[i]; }
      }
      {
        std::lock_guard<std::mutex> lock(plan_mutex);
        ++plan_stats["tunes"]; plan_stats["candidates_timed"]+=shortlist.size();
        for(const auto c:shortlist)
          ++plan_stats[std::string("candidates_timed_")+phase_name(phase)+"_"+candidate_name(c)];
        if(!plan_cache.count(key)) {
          if(plan_cache.size()>=kMaxConvPlans) {
            plan_cache.erase(plan_fifo.front()); plan_fifo.pop_front(); ++plan_stats["evictions"];
          }
          plan_fifo.push_back(key);
        }
        plan_cache[key]=selected;
      }
    }
  }
  {
    std::lock_guard<std::mutex> lock(plan_mutex);
    ++plan_stats[std::string("selected_")+phase_name(phase)+"_"+candidate_name(selected)];
  }
  // Tuning writes only fresh result/scratch tensors; execute the winner again
  // so the observable result always belongs to the cached plan.
  execute_candidate(p,phase,selected,inputs,result,scratch);
}

static at::Tensor convolution_overrideable(
    const at::Tensor& input, const at::Tensor& weight,
    const std::optional<at::Tensor>& bias,
    c10::SymIntArrayRef stride, c10::SymIntArrayRef padding,
    c10::SymIntArrayRef dilation, bool transposed,
    c10::SymIntArrayRef output_padding, c10::SymInt groups) {
  auto p = convolution_params(input, weight, stride, padding, dilation,
                               transposed, output_padding, groups);
  const auto input_c=packed(input), weight_c=packed(weight);
  auto x = buffer(input_c), w = buffer(weight_c);
  auto& mgr = VulkanContext::instance().manager();
  TensorBuffer b;
  at::Tensor bias_c;
  if (bias && bias->defined()) {
    TORCH_CHECK(bias->numel() == p.cout && bias->scalar_type() == at::kFloat,
                "Vulkan convolution: bias shape/dtype mismatch");
    bias_c=packed(*bias);
    b = buffer(bias_c);
  } else {
    b = mgr.tensor(std::vector<float>(p.cout, 0.0f));
  }
  auto result = at::empty({p.n, p.cout, p.oh, p.ow}, input.options());
  dispatch(p,ConvPhase::Forward,{x,w,b},result);
  return result;
}

static std::tuple<at::Tensor,at::Tensor,at::Tensor> convolution_backward_overrideable(
    const at::Tensor& grad_output, const at::Tensor& input,
    const at::Tensor& weight,
    c10::SymIntArrayRef stride, c10::SymIntArrayRef padding,
    c10::SymIntArrayRef dilation, bool transposed,
    c10::SymIntArrayRef output_padding, c10::SymInt groups,
    std::array<bool,3> output_mask) {
  auto p = convolution_params(input, weight, stride, padding, dilation,
                               transposed, output_padding, groups);
  TORCH_CHECK(grad_output.dim() == 4 && grad_output.size(0) == p.n &&
              grad_output.size(1) == p.cout &&
              grad_output.size(2) == p.oh && grad_output.size(3) == p.ow,
              "Vulkan convolution backward: output gradient shape mismatch");
  const auto grad_c=packed(grad_output);
  const auto go = buffer(grad_c);
  at::Tensor dx, dw, db;
  if (output_mask[0]) {
    dx = at::empty(input.sizes(), input.options());
    dispatch(p,ConvPhase::Input,{go,buffer(packed(weight))},dx);
  }
  if (output_mask[1]) {
    dw = at::empty(weight.sizes(),weight.options());
    dispatch(p,ConvPhase::Weight,{buffer(packed(input)),go},dw);
  }
  if (output_mask[2]) {
    db = at::empty({p.cout},weight.options());
    dispatch(p,ConvPhase::Bias,{go},db);
  }
  return {dx,dw,db};
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("convolution_overrideable", &convolution_overrideable);
  m.impl("convolution_backward_overrideable", &convolution_backward_overrideable);
}

} // namespace torch_vulkan::ops
