#include "native_utils.h"
#include "device_copy.h"
#include <ATen/WrapDimUtils.h>
#include <torch/library.h>
namespace torch_vulkan::ops {
namespace {
using namespace native;
at::Tensor cat(const at::ITensorListRef& tensor_list,int64_t dim) {
  std::vector<at::Tensor> owned;
  for(const auto& t:tensor_list)owned.push_back(t);
  at::TensorList tensors(owned);
  TORCH_CHECK(!tensors.empty(),"torch.cat(): expected a non-empty list of Tensors");
  const auto device=tensors[0].device();
  bool native_supported=true,general_supported=true;auto dtype=tensors[0].scalar_type();
  for(const auto& t:tensors) {
    TORCH_CHECK(t.device()==device,"Expected all tensors to be on the same device");
    native_supported &= t.scalar_type()==at::kFloat && t.dim()<=8;
    general_supported &= device_copy_supported(t.scalar_type()) && t.dim()<=8;dtype=at::promote_types(dtype,t.scalar_type());
  }
  // Preserve dtype promotion and higher-rank behavior through the old CPU path.
  if(!native_supported&&!general_supported) {
    std::vector<at::Tensor> cpu;for(const auto& t:tensors)cpu.push_back(native::cpu_fallback_tensor(t,__func__));
    return at::cat(cpu,dim).to(device);
  }
  const auto skipped=[](const at::Tensor& t){return t.dim()==1 && t.numel()==0;};
  const at::Tensor* reference=&tensors[0];
  for(const auto& t:tensors)if(!skipped(t)){reference=&t;break;}
  TORCH_CHECK(reference->dim()>0,"zero-dimensional tensor cannot be concatenated");
  dim=at::maybe_wrap_dim(dim,reference->dim());
  auto shape=reference->sizes().vec();shape[dim]=0;
  for(const auto& t:tensors) {
    if(skipped(t))continue;
    TORCH_CHECK(t.dim()==reference->dim(),"Tensors must have same number of dimensions");
    for(int64_t d=0;d<t.dim();++d)
      TORCH_CHECK(d==dim || t.size(d)==reference->size(d),"Sizes of tensors must match except in dimension");
    TORCH_CHECK(t.size(dim)<=UINT32_MAX && uint64_t(shape[dim])+t.size(dim)<=UINT32_MAX,
                "Vulkan cat axis exceeds uint32 indexing");
    shape[dim]+=t.size(dim);
  }
  auto out=at::empty(shape,reference->options().dtype(dtype));
  if(!out.numel())return out;
  u32(out.numel());
  int64_t inner=1;for(int64_t d=dim+1;d<out.dim();++d)inner*=out.size(d);
  int64_t offset=0;
  for(const auto& t:tensors) {
    if(skipped(t))continue;
    if(t.numel()&&!native_supported){out.narrow(dim,offset,t.size(dim)).copy_(t);}
    else if(t.numel()) {
      auto x=packed(t);
      // Each dispatch writes a disjoint region of the destination. The full
      // result stays allocated through all dispatches and deferred completion.
      run("cat_copy",{base(x)},{write_base(out)},
          {u32(t.numel()),u32(inner),u32(t.size(dim)),u32(shape[dim]),u32(offset)},(t.numel()+127)/128);
    }
    offset+=t.size(dim);
  }
  return out;
}
} // namespace
TORCH_LIBRARY_IMPL(aten,PrivateUse1,m) {
  const char* flag=std::getenv("TORCH_VULKAN_DISABLE_NATIVE_CAT");
  if(flag && !std::strcmp(flag,"1"))return;
  m.impl("cat",&cat);
}
} // namespace torch_vulkan::ops
