#include "native_utils.h"
#include <ATen/native/BinaryOps.h>
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
at::Tensor& add_tensor_(at::Tensor& x,const at::Tensor& other,const at::Scalar& alpha) {
  if(other.device().is_cpu() && other.dim()==0 &&
      at::result_type(x,other)==x.scalar_type())return add_scalar_(x,other.item(),alpha);
  // Tensor/tensor mutation retains the established CPU behavior.
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
