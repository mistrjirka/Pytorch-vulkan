#include "native_utils.h"
#include <ATen/MemoryOverlap.h>
#include <ATen/ExpandUtils.h>
#include <ATen/native/Resize.h>
#include <torch/library.h>
namespace torch_vulkan::ops {
namespace {
void fp32(const at::Tensor& x){TORCH_CHECK(x.scalar_type()==at::kFloat&&native::can_pack_view(x),"Vulkan optimizer primitives require packable FP32 tensors");}
at::Tensor lerp_scalar(const at::Tensor& x,const at::Tensor& end,const at::Scalar& weight){fp32(x);fp32(end);return x.add(end.sub(x).mul(weight));}
at::Tensor lerp_tensor(const at::Tensor& x,const at::Tensor& end,const at::Tensor& weight){fp32(weight);fp32(x);fp32(end);return x.add(end.sub(x).mul(weight));}
at::Tensor addcmul_native(const at::Tensor& x,const at::Tensor& a,const at::Tensor& b,const at::Scalar& value){fp32(x);fp32(a);fp32(b);return x.add(a.mul(b),value);}
at::Tensor addcdiv_native(const at::Tensor& x,const at::Tensor& a,const at::Tensor& b,const at::Scalar& value){fp32(x);fp32(a);fp32(b);return x.add(a.div(b),value);}
at::Tensor& output(at::Tensor& out,const at::Tensor& result){TORCH_CHECK(out.device()==result.device()&&out.scalar_type()==result.scalar_type(),"optimizer output device/dtype mismatch");at::assert_no_internal_overlap(out);at::native::resize_output(out,result.sizes());out.copy_(result);return out;}
at::Tensor extrema(const at::Tensor& x,const at::Tensor& y,bool minimum){
 fp32(x);fp32(y);TORCH_CHECK(x.device()==y.device(),"extrema device mismatch");auto shape=at::infer_size_dimvector(x.sizes(),y.sizes());auto out=at::empty(shape,x.options());if(!out.numel())return out;
 auto a=native::packed(x.expand(shape)),b=native::packed(y.expand(shape));native::run("extrema_fp32",{native::base(a),native::base(b)},{native::base(out)},{native::u32(out.numel()),uint32_t(minimum)},(out.numel()+127)/128);return out;
}
at::Tensor maximum_native(const at::Tensor& x,const at::Tensor& y){return extrema(x,y,false);}
at::Tensor minimum_native(const at::Tensor& x,const at::Tensor& y){return extrema(x,y,true);}
at::Tensor& maximum_out(const at::Tensor& x,const at::Tensor& y,at::Tensor& out){return output(out,maximum_native(x,y));}
at::Tensor& minimum_out(const at::Tensor& x,const at::Tensor& y,at::Tensor& out){return output(out,minimum_native(x,y));}
at::Tensor& lerp_scalar_out(const at::Tensor& x,const at::Tensor& e,const at::Scalar& w,at::Tensor& out){return output(out,lerp_scalar(x,e,w));}
at::Tensor& lerp_tensor_out(const at::Tensor& x,const at::Tensor& e,const at::Tensor& w,at::Tensor& out){return output(out,lerp_tensor(x,e,w));}
at::Tensor& lerp_scalar_inplace(at::Tensor& x,const at::Tensor& e,const at::Scalar& w){at::assert_no_partial_overlap(x,e);return output(x,lerp_scalar(x,e,w));}
at::Tensor& lerp_tensor_inplace(at::Tensor& x,const at::Tensor& e,const at::Tensor& w){at::assert_no_partial_overlap(x,e);at::assert_no_partial_overlap(x,w);return output(x,lerp_tensor(x,e,w));}
at::Tensor& addcmul_out(const at::Tensor& x,const at::Tensor& a,const at::Tensor& b,const at::Scalar& v,at::Tensor& out){return output(out,addcmul_native(x,a,b,v));}
at::Tensor& addcdiv_out(const at::Tensor& x,const at::Tensor& a,const at::Tensor& b,const at::Scalar& v,at::Tensor& out){return output(out,addcdiv_native(x,a,b,v));}
at::Tensor& addcmul_inplace(at::Tensor& x,const at::Tensor& a,const at::Tensor& b,const at::Scalar& v){at::assert_no_partial_overlap(x,a);at::assert_no_partial_overlap(x,b);return output(x,addcmul_native(x,a,b,v));}
at::Tensor& addcdiv_inplace(at::Tensor& x,const at::Tensor& a,const at::Tensor& b,const at::Scalar& v){at::assert_no_partial_overlap(x,a);at::assert_no_partial_overlap(x,b);return output(x,addcdiv_native(x,a,b,v));}
}
TORCH_LIBRARY_IMPL(aten,PrivateUse1,m){
 m.impl("maximum",&maximum_native);m.impl("minimum",&minimum_native);m.impl("maximum.out",&maximum_out);m.impl("minimum.out",&minimum_out);
 m.impl("lerp.Scalar",&lerp_scalar);m.impl("lerp.Tensor",&lerp_tensor);m.impl("lerp.Scalar_out",&lerp_scalar_out);m.impl("lerp.Tensor_out",&lerp_tensor_out);m.impl("lerp_.Scalar",&lerp_scalar_inplace);m.impl("lerp_.Tensor",&lerp_tensor_inplace);
 m.impl("addcmul",&addcmul_native);m.impl("addcdiv",&addcdiv_native);m.impl("addcmul.out",&addcmul_out);m.impl("addcdiv.out",&addcdiv_out);m.impl("addcmul_",&addcmul_inplace);m.impl("addcdiv_",&addcdiv_inplace);
}
}
