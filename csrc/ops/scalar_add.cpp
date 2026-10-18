#include "native_utils.h"
#include <ATen/native/BinaryOps.h>
#include <ATen/ExpandUtils.h>
#include <ATen/MemoryOverlap.h>
#include <torch/library.h>

namespace torch_vulkan::ops {
namespace {
using namespace native;
// A conservative non-overlap proof supports permutations and holes while
// keeping ambiguous as_strided layouts on the existing CPU path.
bool supported(const at::Tensor& x,const at::Scalar& value,bool inplace) {
  if(x.dim()>8 || (x.scalar_type()!=at::kFloat && x.scalar_type()!=at::kLong) ||
      at::result_type(x,value)!=x.scalar_type())return false;
  std::vector<std::pair<int64_t,int64_t>> axes;
  for(int64_t d=0;d<x.dim();++d)if(x.size(d)>1)axes.push_back({x.stride(d),x.size(d)});
  std::sort(axes.begin(),axes.end());
  uint64_t span=1;
  for(auto [stride,size]:axes) {
    if(stride<0 || uint64_t(stride)>UINT32_MAX)return false;
    if(inplace && uint64_t(stride)<span)return false;
    span+=uint64_t(size-1)*stride;
    if(span>UINT32_MAX)return false;
  }
  return uint64_t(x.storage_offset())+span-1<=UINT32_MAX;
}
void dispatch(const at::Tensor& x,const at::Tensor& out,const at::Scalar& value,
              const at::Scalar& alpha,bool inplace) {
  at::native::alpha_check(x.scalar_type(),alpha);
  if(!x.numel())return;
  // Rank is padded to eight, so the same shader handles scalars and views.
  std::vector<uint32_t> p(22,0);
  p[0]=u32(x.numel());p[1]=u32(x.storage_offset());p[2]=inplace;
  for(int d=0;d<8;++d) {
    p[4+d]=d<x.dim()?u32(x.size(d)):1;
    p[12+d]=d<x.dim()?u32(x.stride(d)):0;
  }
  const bool integer=x.scalar_type()==at::kLong;
  if(integer) {
    // Unsigned arithmetic gives PyTorch's int64 wraparound without C++ UB.
    const uint64_t delta=uint64_t(value.toLong())*uint64_t(alpha.toLong());
    p[20]=uint32_t(delta);p[21]=uint32_t(delta>>32);
  } else { p[20]=bits(value.toFloat());p[21]=bits(alpha.toFloat()); }
  run(integer?"scalar_add_int64":"scalar_add_float",{base(x,x.scalar_type())},
      {base(out,out.scalar_type())},p,(x.numel()+127)/128);
}
at::Tensor add_scalar(const at::Tensor& x,const at::Scalar& value,const at::Scalar& alpha) {
  if(!supported(x,value,false))return at::add(x.cpu(),value,alpha).to(x.device());
  auto out=at::empty(x.sizes(),x.options());dispatch(x,out,value,alpha,false);return out;
}
at::Tensor& add_scalar_(at::Tensor& x,const at::Scalar& value,const at::Scalar& alpha) {
  if(!supported(x,value,true)) {
    auto cpu=x.cpu();cpu.add_(value,alpha);x.copy_(cpu);return x;
  }
  dispatch(x,x,value,alpha,true);return x;
}
// Indexing uses the destination's logical shape and two independent stride
// sets: arbitrary non-overlapping destination views and broadcast RHS views.
// No packing or host read is needed for the usual accumulation case.
bool tensor_view_supported(const at::Tensor& t,bool inplace) {
  if(t.device().type()!=c10::DeviceType::PrivateUse1 ||
      t.scalar_type()!=at::kFloat || t.dim()>8 || t.numel()>UINT32_MAX ||
      t.storage_offset()<0 || uint64_t(t.storage_offset())>UINT32_MAX)return false;
  if(t.storage().nbytes()>device_limits().maxStorageBufferRange)return false;
  uint64_t last=t.storage_offset();
  std::vector<std::pair<int64_t,int64_t>> axes;
  for(int64_t d=0;d<t.dim();++d) {
    if(t.size(d)>UINT32_MAX || t.stride(d)<0 || uint64_t(t.stride(d))>UINT32_MAX)return false;
    if(t.size(d)>1) {
      const uint64_t extent=uint64_t(t.size(d)-1)*t.stride(d);
      if(extent>UINT32_MAX-last)return false;
      last+=extent;
      axes.push_back({t.stride(d),t.size(d)});
    }
  }
  if(inplace) {
    // Includes transposes and stepped slices, unlike a contiguous-only guard.
    std::sort(axes.begin(),axes.end());
    uint64_t span=1;
    for(auto [stride,size]:axes) {
      if(uint64_t(stride)<span)return false;
      span+=uint64_t(size-1)*stride;
    }
  }
  return true;
}
at::Tensor& add_tensor_(at::Tensor& x,const at::Tensor& other,const at::Scalar& alpha) {
  // Check the original tensors before any copies obscure their aliases.
  at::assert_no_internal_overlap(x);
  at::assert_no_partial_overlap(x,other);
  if(other.device().is_cpu() && other.dim()==0 &&
      at::result_type(x,other)==x.scalar_type())return add_scalar_(x,other.item(),alpha);
  const char* disabled=std::getenv("TORCH_VULKAN_DISABLE_NATIVE_TENSOR_ADD");
  if((!disabled || std::strcmp(disabled,"1")!=0) &&
      tensor_view_supported(x,true) && tensor_view_supported(other,false) &&
      x.device()==other.device()) {
    at::native::alpha_check(x.scalar_type(),alpha);
    const auto shape=at::infer_size(x.sizes(),other.sizes());
    TORCH_CHECK(at::IntArrayRef(shape)==x.sizes(),
                "output with shape ",x.sizes()," doesn't match the broadcast shape ",shape);
    std::vector<uint32_t> p(29,0);
    p[0]=u32(x.numel());p[1]=u32(x.storage_offset());
    p[2]=u32(other.storage_offset());p[3]=bits(alpha.toFloat());
    if(x.is_contiguous() && other.is_contiguous()) {
      if(other.numel()==x.numel())p[28]=1;
      else if(other.numel()==1)p[28]=2;
    }
    std::fill(p.begin()+4,p.begin()+12,1);
    bool same_mapping=x.storage_offset()==other.storage_offset();
    const int64_t pad=x.dim()-other.dim();
    for(int64_t d=0;d<x.dim();++d) {
      p[4+d]=u32(x.size(d));p[12+d]=u32(x.stride(d));
      const int64_t rd=d-pad;
      const uint32_t stride=rd>=0 && other.size(rd)!=1 ? u32(other.stride(rd)) : 0;
      p[20+d]=stride;
      if(x.size(d)>1 && stride!=p[12+d])same_mapping=false;
    }
    // ATen may return TooHard for sliced aliases. Distinct storages are always
    // safe; equal element mappings are also safe because each invocation reads
    // before its own write. Ambiguous different mappings retain the CPU path.
    const bool shared=x.storage().is_alias_of(other.storage());
    if(!shared || same_mapping || at::get_overlap_status(x,other)==at::MemOverlapStatus::No) {
      if(x.numel())run("tensor_add_inplace",{base(x),base(other)},{base(x)},p,(x.numel()+127)/128);
      return x;
    }
  }
  auto cpu=x.cpu();cpu.add_(other.cpu(),alpha);x.copy_(cpu);return x;
}
} // namespace
TORCH_LIBRARY_IMPL(aten,PrivateUse1,m) {
  const char* flag=std::getenv("TORCH_VULKAN_DISABLE_NATIVE_SCALAR_ADD");
  if(flag && !std::strcmp(flag,"1"))return;
  m.impl("add.Scalar",&add_scalar);
  m.impl("add_.Scalar",&add_scalar_);
  m.impl("add_.Tensor",&add_tensor_);
}
} // namespace torch_vulkan::ops
