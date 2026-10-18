// Original general FP32 Vulkan unary operations for diffusion and other models.
// Mathematical operation definitions; no upstream inference shader source copied.
// Supports arbitrary rank<=8 strides/offsets through native_utils::base.
#include "native_utils.h"
#include "device_copy.h"
#include <torch/library.h>
#include <ATen/Functions.h>
#include <array>

namespace torch_vulkan::ops {
namespace {
using namespace native;

bool enabled() {
  const char* disabled=std::getenv("TORCH_VULKAN_DISABLE_NATIVE_DIFFUSION_UNARY");
  return !disabled || std::strcmp(disabled,"1")!=0;
}

bool eligible(const at::Tensor& x) {
  return enabled() && x.scalar_type()==at::kFloat && can_pack_view(x);
}

at::Tensor cpu_unary(const at::Tensor& x,uint32_t op) {
  auto a=native::cpu_fallback_tensor(x,__func__);
  at::Tensor result;
  switch(op) {
    case 0: result=at::silu(a); break;
    case 1: result=at::sin(a); break;
    case 2: result=at::cos(a); break;
    case 3: result=at::exp(a); break;
    case 4: result=at::sqrt(a); break;
    case 5: result=at::rsqrt(a); break;
    case 6: result=at::tanh(a); break;
    case 7: result=at::neg(a); break;
    default: TORCH_CHECK(false,"Unsupported Vulkan unary primitive");
  }
  return result.to(x.device());
}

at::Tensor unary(const at::Tensor& x,uint32_t op) {
  if(device_copy_supported(x.scalar_type())&&x.scalar_type()!=at::kFloat){
    if(op==7){TORCH_CHECK(x.scalar_type()!=at::kBool,"Negation of a bool tensor is not supported");return at::mul(x,-1);}
    return unary(x.to(at::kFloat),op);
  }
  if(!eligible(x))return cpu_unary(x,op);
  auto out=at::empty(x.sizes(),x.options());
  if(!x.numel())return out;
  std::vector<uint32_t> push(19,0);
  push[0]=u32(x.numel());push[1]=u32(x.storage_offset());push[2]=op;
  for(int d=0;d<8;++d) {
    push[3+d]=d<x.dim()?u32(x.size(d)):1;
    push[11+d]=d<x.dim()?u32(x.stride(d)):0;
  }
  run("diffusion_unary",{base(x)},{write_base(out)},push,
      (uint64_t(x.numel())+127)/128);
  return out;
}

at::Tensor cpu_backward(const at::Tensor& grad,const at::Tensor& x,uint32_t op) {
  auto g=native::cpu_fallback_tensor(grad,__func__),a=native::cpu_fallback_tensor(x,__func__);
  return (op==0?at::silu_backward(g,a):at::tanh_backward(g,a)).to(x.device());
}

at::Tensor backward(const at::Tensor& grad,const at::Tensor& x,uint32_t op) {
  TORCH_CHECK(grad.device()==x.device() && grad.sizes()==x.sizes(),
              "Vulkan unary backward: grad and input must have same shape/device");
  if(!eligible(grad) || !eligible(x))
    return cpu_backward(grad,x,op);
  auto result=at::empty(x.sizes(),x.options());
  if(!x.numel())return result;
  std::vector<uint32_t> push(28,0);
  push[0]=u32(x.numel());
  push[1]=u32(grad.storage_offset());
  push[2]=u32(x.storage_offset());
  push[3]=op;
  for(int d=0;d<8;++d){
    push[4+d]=d<x.dim()?u32(x.size(d)):1;
    push[12+d]=d<x.dim()?u32(grad.stride(d)):0;
    push[20+d]=d<x.dim()?u32(x.stride(d)):0;
  }
  run("diffusion_unary_backward",{base(grad),base(x)},
      {write_base(result)},push,(uint64_t(x.numel())+127)/128);
  return result;
}

at::Tensor silu(const at::Tensor& x){return unary(x,0);}
at::Tensor sin(const at::Tensor& x){return unary(x,1);}
at::Tensor cos(const at::Tensor& x){return unary(x,2);}
at::Tensor exp(const at::Tensor& x){return unary(x,3);}
at::Tensor sqrt(const at::Tensor& x){return unary(x,4);}
at::Tensor rsqrt(const at::Tensor& x){return unary(x,5);}
at::Tensor tanh(const at::Tensor& x){return unary(x,6);}
at::Tensor neg(const at::Tensor& x){return unary(x,7);}
at::Tensor silu_backward(const at::Tensor& g,const at::Tensor& x){return backward(g,x,0);}
at::Tensor tanh_backward(const at::Tensor& g,const at::Tensor& out){return backward(g,out,1);}
} // namespace

TORCH_LIBRARY_IMPL(aten,PrivateUse1,m) {
  m.impl("silu",&silu);
  m.impl("silu_backward",&silu_backward);
  m.impl("sin",&sin);
  m.impl("cos",&cos);
  m.impl("exp",&exp);
  m.impl("sqrt",&sqrt);
  m.impl("rsqrt",&rsqrt);
  m.impl("tanh",&tanh);
  m.impl("tanh_backward",&tanh_backward);
  m.impl("neg",&neg);
}
} // namespace torch_vulkan::ops
