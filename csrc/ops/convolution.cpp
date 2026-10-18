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

namespace torch_vulkan::ops {

using TensorBuffer = std::shared_ptr<kp::Tensor>;

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
  const auto limits=ctx.manager().getDeviceProperties().limits;
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
  TORCH_CHECK(h + 2*ph >= dh*(kh-1) + 1 &&
              w + 2*pw >= dw*(kw-1) + 1,
              "Vulkan convolution: effective kernel exceeds padded input");
  const int64_t oh = (h + 2*ph - dh*(kh-1) - 1) / sh + 1;
  const int64_t ow = (w + 2*pw - dw*(kw-1) - 1) / sw + 1;
  TORCH_CHECK(oh > 0 && ow > 0, "Vulkan convolution: empty output");
  return {checked_uint(n,"N"),checked_uint(cin,"Cin"),checked_uint(cout,"Cout"),
          checked_uint(h,"H"),checked_uint(w,"W"),
          checked_uint(kh,"Kh"),checked_uint(kw,"Kw"),
          checked_uint(oh,"Oh"),checked_uint(ow,"Ow"),
          checked_uint(sh,"stride_h"),checked_uint(sw,"stride_w"),
          checked_uint(ph,"padding_h"),checked_uint(pw,"padding_w"),
          checked_uint(dh,"dilation_h"),checked_uint(dw,"dilation_w"),
          checked_uint(group_count,"groups")};
}

enum class ConvAlgorithm { Direct, Depthwise, Tiled, InputTiled, WeightReduce, BiasReduce, WeightTiled, WeightFinalize };

// Capability-aware selection preserves the direct shader as a generic
// correctness fallback. TORCH_VULKAN_CONV_ALGO=direct/depthwise/tiled/auto
// allows A/B benchmarks without changing network architecture.
static ConvAlgorithm select_conv_forward(const ConvParams& p) {
  const char* choice = std::getenv("TORCH_VULKAN_CONV_ALGO");
  bool force_direct=choice && std::strcmp(choice,"direct")==0;
  bool force_depthwise=choice && std::strcmp(choice,"depthwise")==0;
  bool force_tiled=choice && std::strcmp(choice,"tiled")==0;
  TORCH_CHECK(!choice || !*choice || !std::strcmp(choice,"auto") ||
              force_direct || force_depthwise || force_tiled,
              "TORCH_VULKAN_CONV_ALGO must be auto/direct/tiled/depthwise");
  const bool is_depthwise=p.groups==p.cin && p.cout % p.cin==0;
  if (force_depthwise)
    TORCH_CHECK(is_depthwise,
                "Depthwise Conv2D requires groups == input channels");
  if (force_direct) return ConvAlgorithm::Direct;

  auto limits=VulkanContext::instance().manager().getDeviceProperties().limits;
  const uint64_t rows=static_cast<uint64_t>(p.n)*p.oh*p.ow;
  const uint64_t cout_group=p.cout/p.groups;
  const uint64_t tiles_y=((cout_group+7)/8)*p.groups;
  const bool tile_supported=
      limits.maxComputeWorkGroupInvocations>=128 &&
      limits.maxComputeWorkGroupSize[0]>=16 &&
      limits.maxComputeWorkGroupSize[1]>=8 &&
      limits.maxComputeSharedMemorySize>=1536 &&
      (rows+15)/16<=limits.maxComputeWorkGroupCount[0] &&
      tiles_y<=limits.maxComputeWorkGroupCount[1];

  if (force_tiled) {
    TORCH_CHECK(tile_supported,"Tiled Conv2D exceeds physical device limits");
    return ConvAlgorithm::Tiled;
  }
  if (is_depthwise && (force_depthwise || p.cin>=8))
    return ConvAlgorithm::Depthwise;
  const uint64_t reduction=static_cast<uint64_t>(p.cin/p.groups)*p.kh*p.kw;
  if (tile_supported && cout_group>=8 && reduction>=16 && rows>=32)
    return ConvAlgorithm::Tiled;
  return ConvAlgorithm::Direct;
}

// Select each backward phase independently. Optimizations must retain all
// filter, grouping, stride and dilation semantics, including ragged tiles.
static ConvAlgorithm select_conv_backward(const ConvParams& p, int phase) {
  const char* choice=std::getenv("TORCH_VULKAN_CONV_BACKWARD_ALGO");
  const bool direct=choice && !std::strcmp(choice,"direct");
  const bool force=choice && !std::strcmp(choice,"optimized");
  TORCH_CHECK(!choice || !*choice || !std::strcmp(choice,"auto") || direct || force,
              "TORCH_VULKAN_CONV_BACKWARD_ALGO must be auto/direct/optimized");
  if(direct) return ConvAlgorithm::Direct;
  const auto limits=VulkanContext::instance().manager().getDeviceProperties().limits;
  if(phase==0) {
    const uint64_t rows=static_cast<uint64_t>(p.n)*p.h*p.w;
    const uint64_t tiles=((static_cast<uint64_t>(p.cin/p.groups)+7)/8)*p.groups;
    const bool supported=limits.maxComputeWorkGroupInvocations>=128 &&
        limits.maxComputeWorkGroupSize[0]>=16 && limits.maxComputeWorkGroupSize[1]>=8 &&
        limits.maxComputeSharedMemorySize>=1536 &&
        (rows+15)/16<=limits.maxComputeWorkGroupCount[0] &&
        tiles<=limits.maxComputeWorkGroupCount[1];
    TORCH_CHECK(!force || supported,"Tiled convolution dInput exceeds device limits");
    if(supported && (force || (p.cin/p.groups>=2 && p.cout/p.groups>=8 && rows>=32)))
      return ConvAlgorithm::InputTiled;
  } else {
    if(phase==1) {
      const uint64_t cols=static_cast<uint64_t>(p.cin/p.groups)*p.kh*p.kw;
      const uint64_t splits=(static_cast<uint64_t>(p.n)*p.oh*p.ow+255)/256;
      const uint64_t tiles=((static_cast<uint64_t>(p.cout/p.groups)+7)/8)*p.groups;
      const uint64_t bytes=cols*p.cout*splits*sizeof(float);
      const bool tiled=limits.maxComputeWorkGroupInvocations>=128 &&
          limits.maxComputeWorkGroupSize[0]>=16 && limits.maxComputeWorkGroupSize[1]>=8 &&
          limits.maxComputeSharedMemorySize>=1536 &&
          (cols+15)/16<=limits.maxComputeWorkGroupCount[0] &&
          tiles<=limits.maxComputeWorkGroupCount[1] &&
          splits<=limits.maxComputeWorkGroupCount[2] &&
          bytes<=std::min<uint64_t>(64*1024*1024,limits.maxStorageBufferRange);
      if(tiled && p.cout/p.groups>=8 && cols>=16 && (force || cols*p.cout>4096)) return ConvAlgorithm::WeightTiled;
    }
    const bool supported=limits.maxComputeWorkGroupInvocations>=128 &&
        limits.maxComputeWorkGroupSize[0]>=128 && limits.maxComputeSharedMemorySize>=512;
    TORCH_CHECK(!force || supported,"Convolution reduction exceeds device limits");
    const uint64_t reduction=static_cast<uint64_t>(p.n)*p.oh*p.ow;
    if(supported && (force || reduction>=128))
      return phase==1 ? ConvAlgorithm::WeightReduce : ConvAlgorithm::BiasReduce;
  }
  return ConvAlgorithm::Direct;
}

static std::shared_ptr<kp::Algorithm> make_algorithm(const std::string& shader,
                const std::vector<TensorBuffer>& inputs,
                const TensorBuffer& output,
                int64_t output_count,
                const ConvParams& shape,
                ConvAlgorithm selected=ConvAlgorithm::Direct) {
  TORCH_CHECK(output_count >= 0 &&
              static_cast<uint64_t>(output_count)<=UINT32_MAX,
              "Vulkan convolution: output exceeds uint32 indexing");
  TORCH_CHECK(static_cast<uint64_t>(shape.n)*shape.cin*shape.h*shape.w<=UINT32_MAX &&
              static_cast<uint64_t>(shape.n)*shape.cout*shape.oh*shape.ow<=UINT32_MAX &&
              static_cast<uint64_t>(shape.cout)*(shape.cin/shape.groups)*shape.kh*shape.kw<=UINT32_MAX,
              "Vulkan convolution: input/weight/output indexing overflow");
  TORCH_CHECK(output_count>0,"Vulkan convolution dispatch requires nonempty output");
  std::vector<TensorBuffer> bindings(inputs);
  bindings.push_back(output);
  auto& context = VulkanContext::instance();
  auto& mgr = context.manager();
  auto shader_name=shader;
  auto workgroup=kp::Workgroup({static_cast<uint32_t>((output_count+255)/256),1,1});
  if (selected==ConvAlgorithm::Tiled) {
    shader_name="conv2d_tiled";
    const uint32_t tiles_per_group=(shape.cout/shape.groups+7)/8;
    workgroup=kp::Workgroup({
      static_cast<uint32_t>((static_cast<uint64_t>(shape.n)*shape.oh*shape.ow+15)/16),
      tiles_per_group*shape.groups,1});
  } else if (selected==ConvAlgorithm::InputTiled) {
    shader_name="conv2d_backward_input_tiled";
    workgroup=kp::Workgroup({
      static_cast<uint32_t>((static_cast<uint64_t>(shape.n)*shape.h*shape.w+15)/16),
      ((shape.cin/shape.groups+7)/8)*shape.groups,1});
  } else if (selected==ConvAlgorithm::WeightTiled) {
    shader_name="conv2d_backward_weight_tiled";
    const uint64_t cols=static_cast<uint64_t>(shape.cin/shape.groups)*shape.kh*shape.kw;
    workgroup=kp::Workgroup({static_cast<uint32_t>((cols+15)/16),
      ((shape.cout/shape.groups+7)/8)*shape.groups,
      static_cast<uint32_t>((static_cast<uint64_t>(shape.n)*shape.oh*shape.ow+255)/256)});
  } else if (selected==ConvAlgorithm::WeightFinalize) {
    shader_name="conv2d_backward_weight_finalize";
    const auto limits=mgr.getDeviceProperties().limits;
    const uint64_t count=(output_count+127)/128;
    const uint64_t gx=std::min<uint64_t>(count,limits.maxComputeWorkGroupCount[0]);
    const uint64_t gy=(count+gx-1)/gx;
    TORCH_CHECK(gy<=limits.maxComputeWorkGroupCount[1],
                "Vulkan convolution: final reduction exceeds device limits");
    workgroup=kp::Workgroup({static_cast<uint32_t>(gx),static_cast<uint32_t>(gy),1});
  } else if (selected==ConvAlgorithm::WeightReduce || selected==ConvAlgorithm::BiasReduce) {
    shader_name=selected==ConvAlgorithm::WeightReduce ?
        "conv2d_backward_weight_reduce" : "conv2d_backward_bias_reduce";
    const auto limits=mgr.getDeviceProperties().limits;
    const uint64_t gx=std::min<uint64_t>(output_count,limits.maxComputeWorkGroupCount[0]);
    const uint64_t gy=(output_count+gx-1)/gx;
    TORCH_CHECK(gy<=limits.maxComputeWorkGroupCount[1],
                "Vulkan convolution: reduction dispatch exceeds device limits");
    workgroup=kp::Workgroup({static_cast<uint32_t>(gx),static_cast<uint32_t>(gy),1});
  } else if (selected==ConvAlgorithm::Depthwise) {
    shader_name="conv2d_depthwise";
  }
  const auto limits=mgr.getDeviceProperties().limits;
  for(size_t axis=0;axis<3;++axis)
    TORCH_CHECK(workgroup[axis]>0 && workgroup[axis]<=limits.maxComputeWorkGroupCount[axis],
                "Vulkan convolution dispatch exceeds physical workgroup limits");
  if(selected==ConvAlgorithm::Direct || selected==ConvAlgorithm::Depthwise)
    TORCH_CHECK(limits.maxComputeWorkGroupInvocations>=256 &&
                limits.maxComputeWorkGroupSize[0]>=256,
                "Vulkan direct convolution requires 256-invocation workgroups");
  for(const auto& binding:bindings)
    TORCH_CHECK(binding->constructDescriptorBufferInfo().range<=limits.maxStorageBufferRange,
                "Vulkan convolution exceeds physical storage buffer range");
  return context.get_or_create_algorithm(shader_name, bindings,
      workgroup, shape.as_push_constants());
}

static void run(const std::string& shader,
                const std::vector<TensorBuffer>& inputs,
                const TensorBuffer& output, int64_t count, const ConvParams& shape,
                ConvAlgorithm selected=ConvAlgorithm::Direct) {
  auto algo=make_algorithm(shader,inputs,output,count,shape,selected);
  auto& context=VulkanContext::instance();
  std::vector<TensorBuffer> bindings(inputs);
  bindings.push_back(output);
  if(context.can_enqueue(bindings)) {
    context.enqueue_compute(algo,bindings,{output});
    return;
  }
  std::string name=shader;
  switch(selected) {
    case ConvAlgorithm::Tiled: name="conv2d_tiled"; break;
    case ConvAlgorithm::Depthwise: name="conv2d_depthwise"; break;
    case ConvAlgorithm::InputTiled: name="conv2d_backward_input_tiled"; break;
    case ConvAlgorithm::WeightReduce: name="conv2d_backward_weight_reduce"; break;
    case ConvAlgorithm::BiasReduce: name="conv2d_backward_bias_reduce"; break;
    case ConvAlgorithm::WeightTiled: name="conv2d_backward_weight_tiled"; break;
    case ConvAlgorithm::WeightFinalize: name="conv2d_backward_weight_finalize"; break;
    default: break;
  }
  context.run_compute(algo,name,inputs,{output});
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
  run("conv2d_forward", {x,w,b}, buffer(result), result.numel(), p,
      select_conv_forward(p));
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
    run("conv2d_backward_input", {go,buffer(packed(weight))}, buffer(dx),dx.numel(),p,select_conv_backward(p,0));
  }
  if (output_mask[1]) {
    dw = at::empty(weight.sizes(),weight.options());
    const auto selected=select_conv_backward(p,1);
    const int64_t splits=(static_cast<int64_t>(p.n)*p.oh*p.ow+255)/256;
    if(selected==ConvAlgorithm::WeightTiled && splits>1) {
      auto partial=at::empty({splits,dw.numel()},weight.options());
      const auto input_c=packed(input);
      const auto xbuf=buffer(input_c), tmpbuf=buffer(partial), outbuf=buffer(dw);
      auto first=make_algorithm("conv2d_backward_weight_tiled",{xbuf,go},tmpbuf,
          partial.numel(),p,selected);
      auto second=make_algorithm("conv2d_backward_weight_finalize",{tmpbuf},outbuf,
          dw.numel(),p,ConvAlgorithm::WeightFinalize);
      VulkanContext::instance().run_compute_pair(first,second,{xbuf,go},{tmpbuf},{outbuf},
          "conv2d_backward_weight_split");
    } else {
      run("conv2d_backward_weight",{buffer(packed(input)),go},buffer(dw),dw.numel(),p,selected);
    }
  }
  if (output_mask[2]) {
    db = at::empty({p.cout},weight.options());
    run("conv2d_backward_bias",{go},buffer(db),db.numel(),p,select_conv_backward(p,2));
  }
  return {dx,dw,db};
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
  m.impl("convolution_overrideable", &convolution_overrideable);
  m.impl("convolution_backward_overrideable", &convolution_backward_overrideable);
}

} // namespace torch_vulkan::ops
