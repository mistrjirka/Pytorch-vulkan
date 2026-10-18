#include "native_utils.h"
#include <torch/library.h>

namespace torch_vulkan::ops {
namespace {
using namespace native;
using Optional = std::optional<at::Tensor>;
bool present(const Optional& t) { return t && t->defined(); }
void check_vector(const at::Tensor& t,int64_t c) {
  base(t);
  TORCH_CHECK(t.dim()==1 && t.numel()==c,"Vulkan BatchNorm expects a channel vector");
  TORCH_CHECK(uint64_t(u32(t.storage_offset()))+uint64_t(c-1)*u32(t.stride(0))<=UINT32_MAX,
              "Vulkan BatchNorm vector indexing overflow");
}
void check_input(const at::Tensor& x) {
  base(x);
  TORCH_CHECK(x.dim()>=2 && x.dim()<=8 && x.numel()>0,
              "Vulkan BatchNorm requires nonempty rank 2 to 8 input");
  u32(x.numel());
}
std::tuple<at::Tensor,at::Tensor,at::Tensor> batch_norm(
    const at::Tensor& input,const Optional& weight,const Optional& bias,
    const Optional& running_mean,const Optional& running_var,
    bool training,double momentum,double eps) {
  check_input(input);
  auto c=input.size(1),s=input.numel()/input.size(0)/c;
  TORCH_CHECK(training || (present(running_mean) && present(running_var)),
              "Vulkan BatchNorm evaluation requires running statistics");
  for(const auto* t:{&weight,&bias,&running_mean,&running_var})
    if(present(*t))check_vector(**t,c);
  auto x=packed(input);
  at::Tensor w,b;
  if(present(weight))w=packed(*weight);
  if(present(bias))b=packed(*bias);
  auto dummy_tensor=at::empty({1},input.options());
  auto dummy=base(dummy_tensor);
  auto y=at::empty(input.sizes(),input.options());
  auto mean=at::empty({training?c:0},input.options());
  auto inv=at::empty({training?c:0},input.options());
  auto rmb=present(running_mean)?base(*running_mean):dummy;
  auto rvb=present(running_var)?base(*running_var):dummy;
  uint32_t flags=present(weight)|(present(bias)<<1)|(present(running_mean)<<2)|(present(running_var)<<3);
  std::vector<uint32_t> p={u32(input.size(0)),u32(c),u32(s),uint32_t(training),flags,
    present(running_mean)?u32(running_mean->storage_offset()):0,
    present(running_mean)?u32(running_mean->stride(0)):0,
    present(running_var)?u32(running_var->storage_offset()):0,
    present(running_var)?u32(running_var->stride(0)):0,bits(float(eps)),bits(float(momentum))};
  // Running statistics are both inputs and writes. Bind them once, but include
  // them in output residency bookkeeping so later CPU reads see the mutation.
  auto& ctx=VulkanContext::instance();
  std::vector<Buffer> inputs={base(x),w.defined()?base(w):dummy,b.defined()?base(b):dummy,rmb,rvb};
  std::vector<Buffer> outputs={base(y),training?base(mean):dummy,training?base(inv):dummy};
  // Validate capabilities via the same checks as native::run, and dispatch with
  // mutation outputs separately from descriptor order.
  auto bindings=inputs;bindings.insert(bindings.end(),outputs.begin(),outputs.end());
  const auto& l=device_limits();
  TORCH_CHECK(l.maxComputeWorkGroupInvocations>=128 && l.maxComputeWorkGroupSize[0]>=128 &&
      l.maxComputeSharedMemorySize>=512 && bindings.size()<=l.maxPerStageDescriptorStorageBuffers &&
      bindings.size()<=l.maxDescriptorSetStorageBuffers,"Vulkan BatchNorm exceeds device capabilities");
  for(const auto& buffer:bindings)TORCH_CHECK(buffer->constructDescriptorBufferInfo().range<=l.maxStorageBufferRange,
      "Vulkan BatchNorm exceeds storage buffer range");
  auto algo=ctx.get_or_create_algorithm("batch_norm_forward",bindings,grid(c),p);
  if(training && present(running_mean))outputs.push_back(rmb);
  if(training && present(running_var))outputs.push_back(rvb);
  if(ctx.can_enqueue(bindings))ctx.enqueue_compute(algo,bindings,outputs);
  else ctx.run_compute(algo,"batch_norm_forward",inputs,outputs);
  return {y,mean,inv};
}
std::tuple<at::Tensor,at::Tensor,at::Tensor> batch_norm_backward(
    const at::Tensor& grad,const at::Tensor& input,const Optional& weight,
    const Optional& running_mean,const Optional& running_var,
    const Optional& save_mean,const Optional& save_inv,bool training,double eps,
    std::array<bool,3> mask) {
  check_input(input);check_input(grad);
  TORCH_CHECK(input.sizes()==grad.sizes(),"Vulkan BatchNorm gradient shape mismatch");
  const auto c=input.size(1),s=input.numel()/input.size(0)/c;
  const auto& mean=training?save_mean:running_mean;
  const auto& inv=training?save_inv:running_var;
  TORCH_CHECK(present(mean) && present(inv),"Vulkan BatchNorm backward requires statistics");
  check_vector(*mean,c);check_vector(*inv,c);
  if(present(weight))check_vector(*weight,c);
  at::Tensor dx,dw,db;
  if(!mask[0]&&!mask[1]&&!mask[2])return {dx,dw,db};
  auto x=packed(input),g=packed(grad);
  at::Tensor w;
  if(present(weight))w=packed(*weight);
  auto dummy_tensor=at::empty({1},input.options());
  auto dummy=base(dummy_tensor);
  if(mask[0])dx=at::empty(input.sizes(),input.options());
  if(mask[1])dw=at::empty({c},input.options());
  if(mask[2])db=at::empty({c},input.options());
  uint32_t flags=mask[0]|(mask[1]<<1)|(mask[2]<<2)|(present(weight)<<3);
  run("batch_norm_backward",{base(x),base(g),w.defined()?base(w):dummy,base(*mean),base(*inv)},
      {mask[0]?write_base(dx):dummy,mask[1]?write_base(dw):dummy,mask[2]?write_base(db):dummy},
      {u32(input.size(0)),u32(c),u32(s),uint32_t(training),flags,
       u32(mean->storage_offset()),u32(mean->stride(0)),u32(inv->storage_offset()),u32(inv->stride(0)),bits(float(eps))},c,1536);
  return {dx,dw,db};
}
} // namespace
TORCH_LIBRARY_IMPL(aten,PrivateUse1,m) {
  if(!native::batch_norm_resize_enabled())return;
  m.impl("native_batch_norm",&batch_norm);
  m.impl("native_batch_norm_backward",&batch_norm_backward);
}
} // namespace torch_vulkan::ops
