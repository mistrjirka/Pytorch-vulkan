#include "fft.h"
#include "device_copy.h"
#include "integer_ops.h"
#include "reduce_native.h"
// General FP32 elementwise primitives and global reduction. No model dispatch.
#include "native_utils.h"
#include <ATen/ExpandUtils.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/native/Resize.h>
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
  if(x.numel())run("loss_unary",{base(x)},{write_base(out)},unary_params(x,op,scalar),
                  (x.numel()+127)/128);
  return out;
}
at::Tensor sigmoid_native(const at::Tensor& x) {
  return supported(x)?unary(x,0):at::sigmoid(native::cpu_fallback_tensor(x,__func__)).to(x.device());
}
at::Tensor abs_native(const at::Tensor& x) {
  if(x.scalar_type()==at::kComplexFloat){auto out=at::empty(x.sizes(),x.options().dtype(at::kFloat));if(x.numel())run("complex_magnitude",{base(packed(x,at::kComplexFloat),at::kComplexFloat)},{write_base(out)},{u32(x.numel())},(x.numel()+127)/128);return out;}
  if(integer_type(x.scalar_type()))return at::where(x.lt(0),at::mul(x,-1),x);
  return supported(x)?unary(x,1):at::abs(native::cpu_fallback_tensor(x,__func__)).to(x.device());
}
at::Tensor sgn_native(const at::Tensor& x) {
  if(x.scalar_type()==at::kComplexFloat){auto magnitude=at::abs(x);return at::where(magnitude.eq(0),at::zeros({},x.options()),x.div(magnitude));}
  if(device_copy_supported(x.scalar_type())&&x.scalar_type()!=at::kFloat)return at::sign(x);
  return supported(x)?unary(x,2):at::sgn(native::cpu_fallback_tensor(x,__func__)).to(x.device());
}
at::Tensor div_scalar(const at::Tensor& x,const at::Scalar& scalar) {
  if(device_copy_supported(x.scalar_type())&&x.scalar_type()!=at::kFloat)return div_scalar(x.to(at::kFloat),scalar);
  if(!supported(x) || at::result_type(x,scalar)!=at::kFloat)
    return at::div(native::cpu_fallback_tensor(x,__func__),scalar).to(x.device());
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
  if(device_copy_supported(aa.scalar_type())&&device_copy_supported(bb.scalar_type())){
    auto dtype=at::result_type(aa,bb);
    if(op==0&&integer_type(dtype))return at::add(aa,bb,at::Scalar(int64_t(uint64_t(0)-uint64_t(alpha.toLong()))));
    if((op==1||dtype==at::kFloat)&&(aa.scalar_type()!=at::kFloat||bb.scalar_type()!=at::kFloat))return binary(aa.to(at::kFloat),bb.to(at::kFloat),op,alpha);
  }
  if(at::result_type(aa,bb)!=at::kFloat ||
     (!aa.device().is_cpu() && !supported(aa)) ||
     (!bb.device().is_cpu() && !supported(bb))) {
    if(op==0)return at::sub(native::cpu_fallback_tensor(aa,__func__),native::cpu_fallback_tensor(bb,__func__),alpha).to(dev);
    if(op==1)return at::div(native::cpu_fallback_tensor(aa,__func__),native::cpu_fallback_tensor(bb,__func__)).to(dev);
    return at::sigmoid_backward(native::cpu_fallback_tensor(aa,__func__),native::cpu_fallback_tensor(bb,__func__)).to(dev);
  }
  auto a=device_scalar(aa,bb),b=device_scalar(bb,aa);
  if(!supported(a) || !supported(b)) {
    if(op==0)return at::sub(native::cpu_fallback_tensor(aa,__func__),native::cpu_fallback_tensor(bb,__func__),alpha).to(dev);
    if(op==1)return at::div(native::cpu_fallback_tensor(aa,__func__),native::cpu_fallback_tensor(bb,__func__)).to(dev);
    return at::sigmoid_backward(native::cpu_fallback_tensor(aa,__func__),native::cpu_fallback_tensor(bb,__func__)).to(dev);
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
    if(op==0)return at::sub(native::cpu_fallback_tensor(aa,__func__),native::cpu_fallback_tensor(bb,__func__),alpha).to(dev);
    if(op==1)return at::div(native::cpu_fallback_tensor(aa,__func__),native::cpu_fallback_tensor(bb,__func__)).to(dev);
    return at::sigmoid_backward(native::cpu_fallback_tensor(aa,__func__),native::cpu_fallback_tensor(bb,__func__)).to(dev);
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
  run("loss_binary",{base(a),base(b)},{write_base(out)},p,(out.numel()+127)/128);
  return out;
}
at::Tensor sub_tensor(const at::Tensor& a,const at::Tensor& b,const at::Scalar& alpha) {
  if(a.is_complex()||b.is_complex())return complex_binary(a,b,0,-alpha);
  return binary(a,b,0,alpha);
}
at::Tensor div_tensor_native(const at::Tensor& a,const at::Tensor& b) {
  if(a.is_complex()||b.is_complex())return complex_binary(a,b,2);
  return binary(a,b,1);
}
at::Tensor sigmoid_backward_native(const at::Tensor& grad,const at::Tensor& out) {
  return binary(grad,out,2);
}
// Preserve storage aliases and original overlap semantics on unsupported views.
at::Tensor& mutate_cpu(at::Tensor& x,const at::Scalar& scalar,uint32_t op,const at::Scalar& alpha=1) {
  auto flat=x.as_strided({int64_t(x.storage().nbytes()/x.element_size())},{1},0);
  auto storage=native::cpu_fallback_tensor(flat,__func__);
  auto view=storage.as_strided(x.sizes(),x.strides(),x.storage_offset());
  if(op==0)view.fill_(scalar);
  else if(op==1)view.div_(scalar);
  else if(op==2)view.sub_(scalar,alpha);
  else view.mul_(scalar);
  flat.copy_(storage);return x;
}
at::Tensor& mutate(at::Tensor& x,const at::Scalar& scalar,uint32_t op,const at::Scalar& alpha=1) {
  if(x.scalar_type()==at::kComplexFloat&&op==3){at::assert_no_internal_overlap(x);x.copy_(complex_scale(x,scalar));return x;}
  const bool fill=op==0;
  if(op==2) {
    at::native::sub_check(x,scalar);
    at::native::alpha_check(at::result_type(x,scalar),alpha);
  }
  if(!fill)at::assert_no_internal_overlap(x);
  if(!fill&&device_copy_supported(x.scalar_type())&&x.scalar_type()!=at::kFloat){
    auto value=op==1?at::div(x,scalar):op==2?at::sub(x,scalar,alpha):at::mul(x,scalar);
    TORCH_CHECK(c10::canCast(value.scalar_type(),x.scalar_type()),"result type cannot be cast to output type");x.copy_(value);return x;
  }
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
  run("loss_mutate",{base(x)},{write_base(x)},p,(uint64_t(p[0])+127)/128);
  return x;
}
at::Tensor& fill_scalar(at::Tensor& x,const at::Scalar& s) {
 if(x.scalar_type()==at::kBool||x.scalar_type()==at::kByte||x.scalar_type()==at::kChar||x.scalar_type()==at::kShort){
  if(!x.numel())return x;
  auto shape=x.sizes().vec();for(int d=0;d<x.dim();d++)if(!x.stride(d))shape[d]=1;
  auto distinct=x.as_strided(shape,x.strides(),x.storage_offset());
  distinct.copy_(at::scalar_tensor(s,at::TensorOptions().dtype(x.scalar_type())).to(x.device()));return x;
 }

 if((x.scalar_type()==at::kInt||x.scalar_type()==at::kLong)&&can_pack_view(x)){
  at::assert_no_internal_overlap(x);if(!x.numel())return x;uint64_t value=x.scalar_type()==at::kInt?uint32_t(s.toInt()):uint64_t(s.toLong());
  std::vector<uint32_t> p={u32(x.numel()),u32(x.storage_offset()),uint32_t(x.element_size()/4),uint32_t(value),uint32_t(value>>32)};p.resize(21,0);std::fill(p.begin()+5,p.begin()+13,1);for(int d=0;d<x.dim();d++){p[5+d]=u32(x.size(d));p[13+d]=u32(x.stride(d));}
  run("integer_fill",{base(x,x.scalar_type())},{write_base(x,x.scalar_type())},p,(x.numel()+127)/128);return x;
 }
if(x.scalar_type()==at::kComplexFloat)return complex_fill(x,s);return mutate(x,s,0);}
at::Tensor& zero_native(at::Tensor& x) {return fill_scalar(x,0);}
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
  if(x.is_complex()||other.is_complex()){auto result=complex_binary(x,other,op==1?2:op==2?0:1,op==2?-alpha:at::Scalar(1));TORCH_CHECK(result.sizes()==x.sizes(),"inplace complex broadcast mismatch");x.copy_(result);return x;}
  const auto shape=at::infer_size(x.sizes(),other.sizes());
  TORCH_CHECK(at::IntArrayRef(shape)==x.sizes(),
              "output with shape ",x.sizes()," doesn't match the broadcast shape ",shape);
  if(device_copy_supported(x.scalar_type())&&device_copy_supported(other.scalar_type())&&(x.scalar_type()!=at::kFloat||other.scalar_type()!=at::kFloat)){
    auto value=op==1?at::div(x,other):op==2?at::sub(x,other,alpha):at::mul(x,other);
    TORCH_CHECK(c10::canCast(value.scalar_type(),x.scalar_type()),"result type cannot be cast to output type");x.copy_(value);return x;
  }
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
      if(x.numel())run("loss_binary",{base(x),base(other)},{write_base(x)},p,(x.numel()+127)/128);
      return x;
    }
  }
  auto flat=x.as_strided({int64_t(x.storage().nbytes()/x.element_size())},{1},0);
  auto storage=native::cpu_fallback_tensor(flat,__func__);
  auto view=storage.as_strided(x.sizes(),x.strides(),x.storage_offset());
  auto rhs=x.scalar_type()==other.scalar_type() && x.storage().is_alias_of(other.storage())?
      storage.as_strided(other.sizes(),other.strides(),other.storage_offset()):native::cpu_fallback_tensor(other,__func__);
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
    auto r=at::log_sigmoid_forward(native::cpu_fallback_tensor(x,__func__));
    return {std::get<0>(r).to(x.device()),std::get<1>(r).to(x.device())};
  }
  auto out=at::empty(x.sizes(),x.options()),buffer=at::empty(x.sizes(),x.options());
  if(x.numel())run("loss_log_sigmoid",{base(x)},{write_base(out),write_base(buffer)},
                  unary_params(x,0),(x.numel()+127)/128);
  return {out,buffer};
}
at::Tensor log_sigmoid_backward(const at::Tensor& grad,const at::Tensor& x,
                                const at::Tensor& buffer) {
  if(!supported(x) || !supported(grad) || !supported(buffer) ||
     x.sizes()!=grad.sizes() || x.sizes()!=buffer.sizes())
    return at::log_sigmoid_backward(native::cpu_fallback_tensor(grad,__func__),native::cpu_fallback_tensor(x,__func__),native::cpu_fallback_tensor(buffer,__func__)).to(x.device());
  auto a=packed(x),g=packed(grad),b=packed(buffer);
  auto out=at::empty(x.sizes(),x.options());
  if(x.numel())run("loss_log_sigmoid_backward",{base(g),base(a),base(b)},
                  {write_base(out)},{u32(x.numel())},(x.numel()+127)/128);
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
    return at::mean(native::cpu_fallback_tensor(x,__func__),dims,keepdim,dtype).to(x.device());
  if(!all_dimensions(x,dims))return reduce_fp32(x,dims,keepdim,true);
  auto out=at::empty(keepdim?std::vector<int64_t>(x.dim(),1):std::vector<int64_t>{},x.options());
  if(!x.numel()) {fill_scalar(out,std::numeric_limits<float>::quiet_NaN());return out;}
  auto current=packed(x);
  uint32_t count=u32(x.numel());
  while(true) {
    const uint32_t groups=(uint64_t(count)+1023)/1024;
    auto next=groups==1?out:at::empty({int64_t(groups)},x.options());
    run("loss_mean_reduce",{base(current)},{write_base(next)},
        {count,bits(groups==1?1.0f/float(x.numel()):1.0f)},groups,128*4);
    if(groups==1)break;
    current=next;count=groups;
  }
  return out;
}
at::Tensor mean_all(const at::Tensor& x,std::optional<at::ScalarType> dtype) {
  return mean_dim(x,std::nullopt,false,dtype);
}
at::Tensor mse_native(const at::Tensor& x,const at::Tensor& target,int64_t reduction){
 TORCH_CHECK(x.scalar_type()==at::kFloat&&target.scalar_type()==at::kFloat&&x.device()==target.device()&&reduction>=0&&reduction<=2,"MSE requires FP32 tensors on the same device and a valid reduction");
 auto difference=x.sub(target);auto squares=difference.mul(difference);if(reduction==0)return squares;return reduction==1?squares.mean():squares.sum();
}
at::Tensor mse_backward_native(const at::Tensor& grad,const at::Tensor& x,const at::Tensor& target,int64_t reduction){
 TORCH_CHECK(x.scalar_type()==at::kFloat&&target.scalar_type()==at::kFloat&&grad.scalar_type()==at::kFloat&&x.device()==target.device()&&grad.device()==x.device()&&reduction>=0&&reduction<=2,"MSE backward input mismatch");
 auto difference=x.sub(target);return difference.mul(reduction==1?2.0/difference.numel():2.0).mul(grad);
}
at::Tensor& mse_output(const at::Tensor& value,at::Tensor& out){TORCH_CHECK(out.device()==value.device()&&out.scalar_type()==value.scalar_type(),"MSE output mismatch");at::assert_no_internal_overlap(out);at::native::resize_output(out,value.sizes());out.copy_(value);return out;}
at::Tensor& mse_out(const at::Tensor& x,const at::Tensor& target,int64_t reduction,at::Tensor& out){return mse_output(mse_native(x,target,reduction),out);}
at::Tensor& mse_backward_out(const at::Tensor& grad,const at::Tensor& x,const at::Tensor& target,int64_t reduction,at::Tensor& out){return mse_output(mse_backward_native(grad,x,target,reduction),out);}
}
TORCH_LIBRARY_IMPL(aten,PrivateUse1,m) {
  const char* flag=std::getenv("TORCH_VULKAN_DISABLE_LOSS_PRIMITIVES");
  if(flag && !std::strcmp(flag,"1"))return;
  m.impl("mse_loss",&mse_native);m.impl("mse_loss.out",&mse_out);m.impl("mse_loss_backward",&mse_backward_native);m.impl("mse_loss_backward.grad_input",&mse_backward_out);
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
  m.impl("zero_",&zero_native);
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
