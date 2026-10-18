#include "reduce_native.h"
// General FP32 elementwise primitives and global reduction. No model dispatch.
#include "native_utils.h"
#include <ATen/ExpandUtils.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/native/BinaryOps.h>
#include <torch/library.h>
#include <cmath>

namespace torch_vulkan::ops {
namespace {
using namespace native;

bool supported(const at::Tensor& x) {
  if(x.device().type()!=c10::DeviceType::PrivateUse1 ||
     x.scalar_type()!=at::kFloat || x.dim()>8 || x.numel()>UINT32_MAX ||
     x.storage_offset()<0 || uint64_t(x.storage_offset())>UINT32_MAX)return false;
  const auto limit=device_limits().maxStorageBufferRange;
  if(x.storage().nbytes()>limit || uint64_t(x.numel())>limit/sizeof(float))return false;
  uint64_t last=x.storage_offset();
  for(int64_t d=0;d<x.dim();++d) {
    if(x.stride(d)<0 || uint64_t(x.stride(d))>UINT32_MAX ||
       uint64_t(x.size(d))>UINT32_MAX)return false;
    if(x.size(d))last+=uint64_t(x.size(d)-1)*x.stride(d);
    if(last>UINT32_MAX)return false;
  }
  return true;
}
bool independent(const at::Tensor& x,bool ignore_expansion=false) {
  std::vector<std::pair<int64_t,int64_t>> axes;
  for(int64_t d=0;d<x.dim();++d)
    if(x.size(d)>1 && !(ignore_expansion && x.stride(d)==0))
      axes.push_back({x.stride(d),x.size(d)});
  std::sort(axes.begin(),axes.end());
  uint64_t span=1;
  for(auto [stride,size]:axes) {
    if(stride<0 || uint64_t(stride)<span)return false;
    span+=uint64_t(size-1)*stride;
  }
  return true;
}
std::vector<uint32_t> unary_params(const at::Tensor& x,uint32_t op,float scalar=0) {
  std::vector<uint32_t> p(20,0);
  p[0]=u32(x.numel());p[1]=u32(x.storage_offset());p[2]=op;p[3]=bits(scalar);
  for(int d=0;d<8;++d) {
    p[4+d]=d<x.dim()?u32(x.size(d)):1;
    p[12+d]=d<x.dim()?u32(x.stride(d)):0;
  }
  return p;
}
at::Tensor unary(const at::Tensor& x,uint32_t op,float scalar=0) {
  auto out=at::empty(x.sizes(),x.options());
  if(x.numel())run("loss_unary",{base(x)},{base(out)},unary_params(x,op,scalar),
                  (x.numel()+127)/128);
  return out;
}
at::Tensor sigmoid_native(const at::Tensor& x) {
  return supported(x)?unary(x,0):at::sigmoid(x.cpu()).to(x.device());
}
at::Tensor abs_native(const at::Tensor& x) {
  return supported(x)?unary(x,1):at::abs(x.cpu()).to(x.device());
}
at::Tensor sgn_native(const at::Tensor& x) {
  return supported(x)?unary(x,2):at::sgn(x.cpu()).to(x.device());
}
at::Tensor div_scalar(const at::Tensor& x,const at::Scalar& scalar) {
  if(!supported(x) || at::result_type(x,scalar)!=at::kFloat)
    return at::div(x.cpu(),scalar).to(x.device());
  return unary(x,3,scalar.toFloat());
}
// CPU zero-dimensional scalar tensors are legal ATen arithmetic operands.
at::Tensor device_scalar(const at::Tensor& x,const at::Tensor& peer) {
  if(!x.device().is_cpu())return x;
  TORCH_CHECK(x.dim()==0,"Expected all tensors to be on the same device");
  return x.to(peer.device(),at::kFloat);
}
at::Tensor binary(const at::Tensor& aa,const at::Tensor& bb,uint32_t op,
                  const at::Scalar& alpha=1) {
  TORCH_CHECK(aa.device()==bb.device() || (aa.device().is_cpu() && aa.dim()==0) ||
              (bb.device().is_cpu() && bb.dim()==0),
              "Expected all tensors to be on the same device");
  auto dev=aa.device().is_cpu()?bb.device():aa.device();
  if(op==0) {
    at::native::sub_check(aa,bb);
    at::native::alpha_check(at::result_type(aa,bb),alpha);
  }
  if(at::result_type(aa,bb)!=at::kFloat ||
     (!aa.device().is_cpu() && !supported(aa)) ||
     (!bb.device().is_cpu() && !supported(bb))) {
    if(op==0)return at::sub(aa.cpu(),bb.cpu(),alpha).to(dev);
    if(op==1)return at::div(aa.cpu(),bb.cpu()).to(dev);
    return at::sigmoid_backward(aa.cpu(),bb.cpu()).to(dev);
  }
  auto a=device_scalar(aa,bb),b=device_scalar(bb,aa);
  if(!supported(a) || !supported(b)) {
    if(op==0)return at::sub(aa.cpu(),bb.cpu(),alpha).to(dev);
    if(op==1)return at::div(aa.cpu(),bb.cpu()).to(dev);
    return at::sigmoid_backward(aa.cpu(),bb.cpu()).to(dev);
  }
  const auto shape=at::infer_size(a.sizes(),b.sizes());
  uint64_t count=1;
  const uint64_t max_count=std::min<uint64_t>(UINT32_MAX,device_limits().maxStorageBufferRange/sizeof(float));
  bool fits=std::find(shape.begin(),shape.end(),0)!=shape.end();
  if(!fits) {
    fits=true;
    for(const auto size:shape) {
      if(uint64_t(size)>max_count/count){fits=false;break;}
      count*=uint64_t(size);
    }
  }
  if(!fits) {
    if(op==0)return at::sub(aa.cpu(),bb.cpu(),alpha).to(dev);
    if(op==1)return at::div(aa.cpu(),bb.cpu()).to(dev);
    return at::sigmoid_backward(aa.cpu(),bb.cpu()).to(dev);
  }
  auto out=at::empty(shape,a.options());
  if(!out.numel())return out;
  std::vector<uint32_t> p(29,0);
  p[0]=u32(out.numel());p[1]=u32(a.storage_offset());p[2]=u32(b.storage_offset());
  p[3]=op;p[4]=bits(alpha.toFloat());
  for(int d=0;d<8;++d) {
    p[5+d]=d<int(shape.size())?u32(shape[d]):1;
    const int ad=d-int(shape.size())+a.dim(),bd=d-int(shape.size())+b.dim();
    p[13+d]=d<int(shape.size()) && ad>=0 && a.size(ad)!=1?u32(a.stride(ad)):0;
    p[21+d]=d<int(shape.size()) && bd>=0 && b.size(bd)!=1?u32(b.stride(bd)):0;
  }
  run("loss_binary",{base(a),base(b)},{base(out)},p,(out.numel()+127)/128);
  return out;
}
at::Tensor sub_tensor(const at::Tensor& a,const at::Tensor& b,const at::Scalar& alpha) {
  return binary(a,b,0,alpha);
}
at::Tensor div_tensor_native(const at::Tensor& a,const at::Tensor& b) {
  return binary(a,b,1);
}
at::Tensor sigmoid_backward_native(const at::Tensor& grad,const at::Tensor& out) {
  return binary(grad,out,2);
}
// Preserve storage aliases and original overlap semantics on unsupported views.
at::Tensor& mutate_cpu(at::Tensor& x,const at::Scalar& scalar,uint32_t op,const at::Scalar& alpha=1) {
  auto flat=x.as_strided({int64_t(x.storage().nbytes()/x.element_size())},{1},0);
  auto storage=flat.cpu();
  auto view=storage.as_strided(x.sizes(),x.strides(),x.storage_offset());
  if(op==0)view.fill_(scalar);
  else if(op==1)view.div_(scalar);
  else if(op==2)view.sub_(scalar,alpha);
  else view.mul_(scalar);
  flat.copy_(storage);return x;
}
at::Tensor& mutate(at::Tensor& x,const at::Scalar& scalar,uint32_t op,const at::Scalar& alpha=1) {
  const bool fill=op==0;
  if(op==2) {
    at::native::sub_check(x,scalar);
    at::native::alpha_check(at::result_type(x,scalar),alpha);
  }
  if(!fill)at::assert_no_internal_overlap(x);
  if(!supported(x) || (!fill && at::result_type(x,scalar)!=at::kFloat) ||
     (fill && scalar.isComplex()) || !independent(x,fill))
    return mutate_cpu(x,scalar,op,alpha);
  if(!x.numel())return x;
  auto p=unary_params(x,op,scalar.toFloat());
  p.push_back(bits(alpha.toFloat()));
  // fill_ permits expanded tensors: write each distinct element exactly once.
  if(fill) {
    uint64_t count=1;
    for(int d=0;d<8;++d) {if(p[12+d]==0)p[4+d]=1;count*=p[4+d];}
    p[0]=u32(count);
  }
  run("loss_mutate",{base(x)},{base(x)},p,(uint64_t(p[0])+127)/128);
  return x;
}
at::Tensor& fill_scalar(at::Tensor& x,const at::Scalar& s) {return mutate(x,s,0);}
at::Tensor& div_scalar_(at::Tensor& x,const at::Scalar& s) {return mutate(x,s,1);}

// Python scalar mutations can arrive as Tensor overloads with CPU scalars.
// Register both overloads so mutation never falls through an alias-blind copy.
at::Tensor& mutate_tensor(at::Tensor& x,const at::Tensor& other,uint32_t op,
                           const at::Scalar& alpha=1) {
  if(op==2) {
    at::native::sub_check(x,other);
    at::native::alpha_check(at::result_type(x,other),alpha);
  }
  TORCH_CHECK(x.device()==other.device() || (other.device().is_cpu() && other.dim()==0),
              "Expected all tensors to be on the same device");
  at::assert_no_internal_overlap(x);
  at::assert_no_partial_overlap(x,other);
  if(other.device().is_cpu() && other.dim()==0)
    return mutate(x,other.item(),op,alpha);
  const auto shape=at::infer_size(x.sizes(),other.sizes());
  TORCH_CHECK(at::IntArrayRef(shape)==x.sizes(),
              "output with shape ",x.sizes()," doesn't match the broadcast shape ",shape);
  if(supported(x) && supported(other) && independent(x)) {
    std::vector<uint32_t> p(29,0);
    p[0]=u32(x.numel());p[1]=u32(x.storage_offset());p[2]=u32(other.storage_offset());
    p[3]=op+2;p[4]=bits(alpha.toFloat());
    bool same_mapping=x.storage_offset()==other.storage_offset();
    const int64_t pad=x.dim()-other.dim();
    for(int d=0;d<8;++d) {
      p[5+d]=d<x.dim()?u32(x.size(d)):1;
      p[13+d]=d<x.dim()?u32(x.stride(d)):0;
      const int64_t rd=d-pad;
      p[21+d]=d<x.dim() && rd>=0 && other.size(rd)!=1?u32(other.stride(rd)):0;
      if(d<x.dim() && x.size(d)>1 && p[13+d]!=p[21+d])same_mapping=false;
    }
    const bool shared=x.storage().is_alias_of(other.storage());
    if(!shared || same_mapping || at::get_overlap_status(x,other)==at::MemOverlapStatus::No) {
      if(x.numel())run("loss_binary",{base(x),base(other)},{base(x)},p,(x.numel()+127)/128);
      return x;
    }
  }
  auto flat=x.as_strided({int64_t(x.storage().nbytes()/x.element_size())},{1},0);
  auto storage=flat.cpu();
  auto view=storage.as_strided(x.sizes(),x.strides(),x.storage_offset());
  auto rhs=x.scalar_type()==other.scalar_type() && x.storage().is_alias_of(other.storage())?
      storage.as_strided(other.sizes(),other.strides(),other.storage_offset()):other.cpu();
  if(op==1)view.div_(rhs);
  else if(op==2)view.sub_(rhs,alpha);
  else view.mul_(rhs);
  flat.copy_(storage);return x;
}

at::Tensor& div_tensor_(at::Tensor& x,const at::Tensor& other) {
  return mutate_tensor(x,other,1);
}
at::Tensor& sub_tensor_(at::Tensor& x,const at::Tensor& other,const at::Scalar& alpha) {
  return mutate_tensor(x,other,2,alpha);
}
at::Tensor& sub_scalar_(at::Tensor& x,const at::Scalar& other,const at::Scalar& alpha) {
  return mutate(x,other,2,alpha);
}
at::Tensor& mul_tensor_(at::Tensor& x,const at::Tensor& other) {
  return mutate_tensor(x,other,3);
}
at::Tensor& mul_scalar_(at::Tensor& x,const at::Scalar& other) {
  return mutate(x,other,3);
}

std::tuple<at::Tensor,at::Tensor> log_sigmoid_forward(const at::Tensor& x) {
  if(!supported(x)) {
    auto r=at::log_sigmoid_forward(x.cpu());
    return {std::get<0>(r).to(x.device()),std::get<1>(r).to(x.device())};
  }
  auto out=at::empty(x.sizes(),x.options()),buffer=at::empty(x.sizes(),x.options());
  if(x.numel())run("loss_log_sigmoid",{base(x)},{base(out),base(buffer)},
                  unary_params(x,0),(x.numel()+127)/128);
  return {out,buffer};
}
at::Tensor log_sigmoid_backward(const at::Tensor& grad,const at::Tensor& x,
                                const at::Tensor& buffer) {
  if(!supported(x) || !supported(grad) || !supported(buffer) ||
     x.sizes()!=grad.sizes() || x.sizes()!=buffer.sizes())
    return at::log_sigmoid_backward(grad.cpu(),x.cpu(),buffer.cpu()).to(x.device());
  auto a=packed(x),g=packed(grad),b=packed(buffer);
  auto out=at::empty(x.sizes(),x.options());
  if(x.numel())run("loss_log_sigmoid_backward",{base(g),base(a),base(b)},
                  {base(out)},{u32(x.numel())},(x.numel()+127)/128);
  return out;
}
bool all_dimensions(const at::Tensor& x,at::OptionalIntArrayRef dims) {
  if(!dims || dims->empty())return true;
  const int64_t rank=std::max<int64_t>(x.dim(),1);
  std::vector<bool> seen(rank,false);
  for(auto raw:*dims) {
    const auto d=raw<0?raw+rank:raw;
    TORCH_CHECK(d>=0 && d<rank,"mean(): dimension out of range");
    TORCH_CHECK(!seen[d],"mean(): dim appears multiple times in the list of dims");
    seen[d]=true;
  }
  return std::all_of(seen.begin(),seen.end(),[](bool v){return v;});
}
at::Tensor mean_dim(const at::Tensor& x,at::OptionalIntArrayRef dims,bool keepdim,
                     std::optional<at::ScalarType> dtype) {
  if(!supported(x) || dtype.value_or(x.scalar_type())!=at::kFloat)
    return at::mean(x.cpu(),dims,keepdim,dtype).to(x.device());
  if(!all_dimensions(x,dims))return reduce_fp32(x,dims,keepdim,true);
  auto out=at::empty(keepdim?std::vector<int64_t>(x.dim(),1):std::vector<int64_t>{},x.options());
  if(!x.numel()) {fill_scalar(out,std::numeric_limits<float>::quiet_NaN());return out;}
  auto current=packed(x);
  uint32_t count=u32(x.numel());
  while(true) {
    const uint32_t groups=(uint64_t(count)+1023)/1024;
    auto next=groups==1?out:at::empty({int64_t(groups)},x.options());
    run("loss_mean_reduce",{base(current)},{base(next)},
        {count,bits(groups==1?1.0f/float(x.numel()):1.0f)},groups,128*4);
    if(groups==1)break;
    current=next;count=groups;
  }
  return out;
}
at::Tensor mean_all(const at::Tensor& x,std::optional<at::ScalarType> dtype) {
  return mean_dim(x,std::nullopt,false,dtype);
}
}
TORCH_LIBRARY_IMPL(aten,PrivateUse1,m) {
  const char* flag=std::getenv("TORCH_VULKAN_DISABLE_LOSS_PRIMITIVES");
  if(flag && !std::strcmp(flag,"1"))return;
  m.impl("sub.Tensor",&sub_tensor);
  m.impl("div.Tensor",&div_tensor_native);
  m.impl("div.Scalar",&div_scalar);
  m.impl("div_.Scalar",&div_scalar_);
  m.impl("div_.Tensor",&div_tensor_);
  m.impl("sub_.Tensor",&sub_tensor_);
  m.impl("sub_.Scalar",&sub_scalar_);
  m.impl("mul_.Tensor",&mul_tensor_);
  m.impl("mul_.Scalar",&mul_scalar_);
  m.impl("fill_.Scalar",&fill_scalar);
  m.impl("sigmoid",&sigmoid_native);
  m.impl("sigmoid_backward",&sigmoid_backward_native);
  m.impl("abs",&abs_native);
  m.impl("sgn",&sgn_native);
  m.impl("log_sigmoid_forward",&log_sigmoid_forward);
  m.impl("log_sigmoid_backward",&log_sigmoid_backward);
  m.impl("mean",&mean_all);
  m.impl("mean.dim",&mean_dim);
}
} // namespace torch_vulkan::ops
