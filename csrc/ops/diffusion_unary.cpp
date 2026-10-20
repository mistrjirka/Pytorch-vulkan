// Original general FP32 Vulkan unary operations for diffusion and other models.
// Mathematical operation definitions; no upstream inference shader source copied.
// Supports arbitrary rank<=8 strides/offsets through native_utils::base.
#include "native_utils.h"
#include "device_copy.h"
#include "fft.h"
#include <torch/library.h>
#include <ATen/Functions.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/native/Resize.h>
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
  if(x.scalar_type()==at::kComplexFloat&&enabled()&&(op==0||op==3||op==7))return complex_math_unary(x,op==0?4:op==3?3:5);
  if(device_copy_supported(x.scalar_type())&&x.scalar_type()!=at::kFloat){
    if(op==7){TORCH_CHECK(x.scalar_type()!=at::kBool,"Negation, the `-` operator, on a bool tensor is not supported. If you are trying to invert a mask, use the `~` or `logical_not()` operator instead.");return at::mul(x,-1);}
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


// Direct-out adapter for the *existing* general unary shader. One shader
// dispatch writes directly to the user's destination instead of materializing
// a separate output followed by an unnecessary GPU copy. Keep the original
// functional+copy route for unsupported layouts, casts and overlapping views.
at::Tensor& unary_out(const at::Tensor& x,at::Tensor& out,uint32_t op) {
  TORCH_CHECK(out.device()==x.device(),"out tensor must be on the same device as the input");
  at::assert_no_internal_overlap(out);
  at::assert_no_partial_overlap(out,x);
  // Output conversion and empty resizing preserve the standard ATen semantics.
  // Do not modify the destination before checking its original alias/shape.
  const bool alias=out.storage().is_alias_of(x.storage());
  if(alias && out.sizes()!=x.sizes())
    TORCH_CHECK(false,"output with shape ",out.sizes(),
                " does not match the broadcast shape ",x.sizes());
  at::native::resize_output(out,x.sizes());
  const bool matching_alias=!alias ||
      (out.sizes()==x.sizes() && out.strides()==x.strides() &&
       out.storage_offset()==x.storage_offset());
  if(direct_unary_out_enabled() &&
     x.scalar_type()==at::kFloat && out.scalar_type()==at::kFloat &&
     eligible(x) && !x.is_neg() && !x.is_conj() &&
     out.is_contiguous() && !out.is_neg() && !out.is_conj() &&
     matching_alias && can_pack_view(out)) {
    if(out.numel()) {
      std::vector<uint32_t> push(19,0);
      push[0]=u32(x.numel());push[1]=u32(x.storage_offset());push[2]=op;
      for(int d=0;d<8;++d) {
        push[3+d]=d<x.dim()?u32(x.size(d)):1;
        push[11+d]=d<x.dim()?u32(x.stride(d)):0;
      }
      // write_base() honors PyTorch's copy-on-write; capture input after that.
      auto destination=write_base(out);
      run("diffusion_unary",{base(x)},{destination},push,
          (uint64_t(x.numel())+127)/128);
    }
    return out;
  }
  auto value=unary(x,op);
  TORCH_CHECK(c10::canCast(value.scalar_type(),out.scalar_type()),
              "result type cannot be cast to the desired output type");
  out.copy_(value);
  return out;
}
at::Tensor& silu_out(const at::Tensor& x,at::Tensor& out){return unary_out(x,out,0);}
at::Tensor& sin_out(const at::Tensor& x,at::Tensor& out){return unary_out(x,out,1);}
at::Tensor& cos_out(const at::Tensor& x,at::Tensor& out){return unary_out(x,out,2);}
at::Tensor& exp_out(const at::Tensor& x,at::Tensor& out){return unary_out(x,out,3);}
at::Tensor& sqrt_out(const at::Tensor& x,at::Tensor& out){return unary_out(x,out,4);}
at::Tensor& rsqrt_out(const at::Tensor& x,at::Tensor& out){return unary_out(x,out,5);}
at::Tensor& tanh_out(const at::Tensor& x,at::Tensor& out){return unary_out(x,out,6);}
at::Tensor& neg_out(const at::Tensor& x,at::Tensor& out){return unary_out(x,out,7);}

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
  m.impl("silu.out",&silu_out);
  m.impl("sin.out",&sin_out);
  m.impl("cos.out",&cos_out);
  m.impl("exp.out",&exp_out);
  m.impl("sqrt.out",&sqrt_out);
  m.impl("rsqrt.out",&rsqrt_out);
  m.impl("tanh.out",&tanh_out);
  m.impl("neg.out",&neg_out);
}
} // namespace torch_vulkan::ops
