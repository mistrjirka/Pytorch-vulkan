#include "native_utils.h"
#include "device_copy.h"
#include <ATen/ExpandUtils.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/WrapDimUtils.h>
#include <ATen/native/Resize.h>
#include <torch/library.h>
#include <cmath>
namespace torch_vulkan::ops {
namespace {
using namespace native;
template<int Op>at::Tensor math_unary(const at::Tensor& raw){
 if constexpr(Op==26)TORCH_CHECK(raw.scalar_type()==at::kFloat,"frac is only implemented for floating-point inputs");
 TORCH_CHECK(device_copy_supported(raw.scalar_type()),"Vulkan pointwise math supports FP32/int32/int64/bool inputs");
 if constexpr(Op>=4&&Op<=7){TORCH_CHECK(raw.scalar_type()!=at::kBool,"rounding operations are not implemented for bool");if(!raw.is_floating_point())return raw.clone();}
 if constexpr(Op==8){if(raw.scalar_type()==at::kBool)return raw.clone();if(!raw.is_floating_point()){auto negative=raw.lt(0),positive=raw.gt(0);return positive.to(raw.scalar_type()).add(negative.to(raw.scalar_type()),-1);}}
 auto x=raw.to(at::kFloat);TORCH_CHECK(can_pack_view(x),"pointwise math exceeds Vulkan indexing limits");auto out=at::empty(x.sizes(),x.options());if(!x.numel())return out;
 run("pointwise_math",{base(packed(x))},{write_base(out)},{u32(x.numel()),Op,bits(1),0},(x.numel()+127)/128);return out;
}

// Reuse existing pointwise_math SPIR-V to write straight into out where the
// source and destination have exact flat FP32 mappings. This removes the
// intermediate tensor+copy used by the generic boxed NativeOut adapter.
// All other input/output dtypes, views, casts and alias cases still take
// that same materialize-then-copy implementation.
template<int Op>at::Tensor& math_unary_out(const at::Tensor& raw,at::Tensor& out) {
  TORCH_CHECK(raw.device()==out.device(),"out tensor must be on the same device as the input");
  at::assert_no_internal_overlap(out);
  at::assert_no_partial_overlap(out,raw);
  const bool alias=out.storage().is_alias_of(raw.storage());
  if(alias && out.sizes()!=raw.sizes())
    TORCH_CHECK(false,"output with shape ",out.sizes(),
                " does not match the broadcast shape ",raw.sizes());
  at::native::resize_output(out,raw.sizes());
  const bool alias_same=!alias ||
      (raw.sizes()==out.sizes() && raw.strides()==out.strides() &&
       raw.storage_offset()==out.storage_offset());
  if(direct_unary_out_enabled() && raw.scalar_type()==at::kFloat &&
     out.scalar_type()==at::kFloat && raw.is_contiguous() && out.is_contiguous() &&
     raw.storage_offset()==0 && out.storage_offset()==0 &&
     !raw.is_neg() && !raw.is_conj() && !out.is_neg() && !out.is_conj() &&
     alias_same && can_pack_view(raw) && can_pack_view(out)) {
    if(out.numel()){
      auto dst=write_base(out);
      run("pointwise_math",{base(raw)},{dst},
          {u32(raw.numel()),Op,bits(1),0},(raw.numel()+127)/128);
    }
    return out;
  }
  auto value=math_unary<Op>(raw);
  TORCH_CHECK(c10::canCast(value.scalar_type(),out.scalar_type()),
              "result type cannot be cast to the desired output type");
  out.copy_(value);
  return out;
}
template<int Op>at::Tensor& math_unary_inplace(at::Tensor& x){at::assert_no_internal_overlap(x);auto value=math_unary<Op>(x);TORCH_CHECK(c10::canCast(value.scalar_type(),x.scalar_type()),"result type cannot be cast to the desired output type");x.copy_(value);return x;}
at::Tensor round_decimals(const at::Tensor& x,int64_t decimals){if(!decimals)return math_unary<7>(x);TORCH_CHECK(x.scalar_type()==at::kFloat,"round decimals requires FP32");auto out=at::empty(x.sizes(),x.options());if(out.numel()){float factor=std::pow(10.,std::abs(double(decimals)));run("pointwise_math",{base(packed(x))},{write_base(out)},{u32(x.numel()),25,bits(factor),uint32_t(decimals<0)},(x.numel()+127)/128);}return out;}
at::Tensor& round_decimals_inplace(at::Tensor& x,int64_t decimals){x.copy_(round_decimals(x,decimals));return x;}
at::Tensor integer_math(const at::Tensor& left,const at::Tensor& right,uint32_t op){
 auto dev=left.is_cpu()?right.device():left.device();TORCH_CHECK(dev.type()==c10::DeviceType::PrivateUse1&&(left.device()==dev||(left.is_cpu()&&left.dim()==0))&&(right.device()==dev||(right.is_cpu()&&right.dim()==0)),"integer math tensors must be on the same device");auto dtype=at::result_type(left,right);TORCH_CHECK(device_copy_supported(dtype)&&dtype!=at::kBool&&dtype!=at::kFloat,"integer math requires integer input");auto compute=dtype==at::kLong?at::kLong:at::kInt;auto shape=at::infer_size_dimvector(left.sizes(),right.sizes());auto result=at::empty(shape,left.options().device(dev).dtype(compute));
 if(result.numel()){auto a=packed(left.to(dev,compute).expand(shape),compute),b=packed(right.to(dev,compute).expand(shape),compute),invalid=at::zeros({},result.options().dtype(at::kInt));run("integer_math",{base(a,compute),base(b,compute)},{write_base(result,compute),write_base(invalid,at::kInt)},{u32(result.numel()),uint32_t(result.element_size()/4),op},(result.numel()+127)/128);if(op!=2&&op<5)TORCH_CHECK(invalid.cpu().item<int>()==0,"ZeroDivisionError");}return result.to(dtype);
}

template<int Op>at::Tensor binary_math(const at::Tensor& left,const at::Tensor& right){
 auto device=left.is_cpu()?right.device():left.device();TORCH_CHECK(device.type()==c10::DeviceType::PrivateUse1&&(left.device()==device||(left.is_cpu()&&left.dim()==0))&&(right.device()==device||(right.is_cpu()&&right.dim()==0)),"binary math tensors must be on the same device");
 auto dtype=at::result_type(left,right);if constexpr(Op<=3)if(dtype!=at::kFloat)return integer_math(left,right,Op);
 TORCH_CHECK(device_copy_supported(dtype),"binary math requires real input");auto shape=at::infer_size_dimvector(left.sizes(),right.sizes());auto out=at::empty(shape,left.options().device(device).dtype(at::kFloat));if(!out.numel())return out;
 auto a=packed(left.to(device,at::kFloat).expand(shape)),b=packed(right.to(device,at::kFloat).expand(shape));run("pointwise_binary",{base(a),base(b)},{write_base(out)},{u32(out.numel()),Op},(out.numel()+127)/128);return out;
}
template<int Op>at::Tensor integer_binary_extra(const at::Tensor& x,const at::Tensor& y){return integer_math(x,y,Op);}
template<int Op>at::Tensor& integer_binary_extra_inplace(at::Tensor& x,const at::Tensor& y){at::assert_no_internal_overlap(x);auto value=integer_math(x,y,Op);TORCH_CHECK(x.sizes()==value.sizes()&&c10::canCast(value.scalar_type(),x.scalar_type()),"inplace result shape/dtype mismatch");x.copy_(value);return x;}
template<int Op>at::Tensor binary_math_scalar(const at::Tensor& x,const at::Scalar& value){return binary_math<Op>(x,at::scalar_tensor(value,at::TensorOptions().dtype(at::result_type(x,value))));}
template<int Op>at::Tensor& binary_math_inplace(at::Tensor& x,const at::Tensor& y){at::assert_no_internal_overlap(x);auto value=binary_math<Op>(x,y);TORCH_CHECK(x.sizes()==value.sizes()&&c10::canCast(value.scalar_type(),x.scalar_type()),"inplace binary result shape/dtype mismatch");x.copy_(value);return x;}
template<int Op>at::Tensor& binary_math_scalar_inplace(at::Tensor& x,const at::Scalar& value){return binary_math_inplace<Op>(x,at::scalar_tensor(value,at::TensorOptions().dtype(at::result_type(x,value))));}
at::Tensor div_mode(const at::Tensor& a,const at::Tensor& b,std::optional<c10::string_view> mode){if(!mode)return at::div(a,b);if(*mode=="trunc"){if(at::result_type(a,b)!=at::kFloat)return integer_math(a,b,4);return at::trunc(at::div(a,b));}TORCH_CHECK(*mode=="floor","div expected rounding_mode to be one of None, 'trunc', or 'floor'");return binary_math<3>(a,b);}
at::Tensor div_scalar_mode(const at::Tensor& a,const at::Scalar& b,std::optional<c10::string_view> mode){return div_mode(a,at::scalar_tensor(b,at::TensorOptions().dtype(at::result_type(a,b))),mode);}
at::Tensor& div_mode_inplace(at::Tensor& a,const at::Tensor& b,std::optional<c10::string_view> mode){auto value=div_mode(a,b,mode);TORCH_CHECK(a.sizes()==value.sizes()&&c10::canCast(value.scalar_type(),a.scalar_type()),"inplace div shape/dtype mismatch");a.copy_(value);return a;}
at::Tensor& div_scalar_mode_inplace(at::Tensor& a,const at::Scalar& b,std::optional<c10::string_view> mode){return div_mode_inplace(a,at::scalar_tensor(b,at::TensorOptions().dtype(at::result_type(a,b))),mode);}

template<int Op>at::Tensor scalar_left_math(const at::Scalar& a,const at::Tensor& b){return binary_math<Op>(at::scalar_tensor(a,at::TensorOptions().dtype(at::result_type(b,a))),b);}
at::Tensor gamma_math(const at::Tensor& raw,uint32_t op,uint32_t order){TORCH_CHECK(device_copy_supported(raw.scalar_type()),"gamma functions require real input");auto x=raw.to(at::kFloat);auto result=at::empty_like(x);if(x.numel())run("special_gamma",{base(packed(x))},{write_base(result)},{u32(x.numel()),op,order},(x.numel()+127)/128);return result;}
at::Tensor lgamma_native(const at::Tensor& x){return gamma_math(x,0,0);}
at::Tensor digamma_native(const at::Tensor& x){return gamma_math(x,1,0);}
at::Tensor polygamma_native(int64_t n,const at::Tensor& x){TORCH_CHECK(n>=0&&n<UINT32_MAX,"polygamma(n, x) does not support negative n or n outside uint32 range");return gamma_math(x,2,uint32_t(n));}
at::Tensor& lgamma_inplace(at::Tensor& x){auto result=lgamma_native(x);TORCH_CHECK(c10::canCast(result.scalar_type(),x.scalar_type()),"lgamma result type cannot be cast");x.copy_(result);return x;}
at::Tensor& digamma_inplace(at::Tensor& x){auto result=digamma_native(x);TORCH_CHECK(c10::canCast(result.scalar_type(),x.scalar_type()),"digamma result type cannot be cast");x.copy_(result);return x;}
at::Tensor& polygamma_inplace(at::Tensor& x,int64_t n){auto result=polygamma_native(n,x);TORCH_CHECK(c10::canCast(result.scalar_type(),x.scalar_type()),"polygamma result type cannot be cast");x.copy_(result);return x;}
void require_real_fp32(const at::Tensor& x){TORCH_CHECK(x.scalar_type()==at::kFloat,"activation requires FP32 input");}
at::Tensor elu_general(const at::Tensor& x,const at::Scalar& alpha,const at::Scalar& scale,const at::Scalar& input_scale){require_real_fp32(x);return at::where(x.gt(0),x.mul(scale),at::expm1(x.mul(input_scale)).mul(alpha).mul(scale));}
at::Tensor elu_grad(const at::Tensor& g,const at::Scalar& alpha,const at::Scalar& scale,const at::Scalar& input_scale,bool result,const at::Tensor& x){require_real_fp32(x);TORCH_CHECK(!result||alpha.toDouble()>=0,"In-place elu backward with negative alpha is not supported");auto slope=result?x.add(alpha.toDouble()*scale.toDouble()).mul(input_scale):at::exp(x.mul(input_scale)).mul(alpha).mul(scale).mul(input_scale);return g.mul(at::where(x.le(0),slope,at::full({},scale,x.options())));}
template<bool Soft>at::Tensor shrink_general(const at::Tensor& x,const at::Scalar& lambda){require_real_fp32(x);if constexpr(Soft)TORCH_CHECK(lambda.toDouble()>=0,"lambda must be in range [0, max] for input dtype float, found ",lambda.toDouble());auto middle=at::logical_and(x.ge(-lambda.toDouble()),x.le(lambda));auto shifted=Soft?at::where(x.gt(0),x.sub(lambda),x.add(lambda)):x;return at::where(middle,at::zeros({},x.options()),shifted);}
at::Tensor shrink_grad(const at::Tensor& g,const at::Tensor& x,const at::Scalar& lambda){return at::where(at::logical_and(x.ge(-lambda.toDouble()),x.le(lambda)),at::zeros({},g.options()),g);}
at::Tensor softplus_general(const at::Tensor& x,const at::Scalar& beta,const at::Scalar& threshold){require_real_fp32(x);auto z=x.mul(beta);return at::where(z.gt(threshold),x,at::log1p(at::exp(z)).div(beta));}
at::Tensor softplus_grad(const at::Tensor& g,const at::Tensor& x,const at::Scalar& beta,const at::Scalar& threshold){auto z=x.mul(beta);return g.mul(at::where(z.gt(threshold),at::ones({},x.options()),at::sigmoid(z)));}
at::Tensor hardsigmoid_general(const at::Tensor& x){require_real_fp32(x);return at::clamp(x.add(3),0,6).div(6);}
at::Tensor hardsigmoid_grad(const at::Tensor& g,const at::Tensor& x){return at::where(at::logical_and(x.gt(-3),x.lt(3)),g.div(6),at::zeros({},g.options()));}
at::Tensor hardswish_general(const at::Tensor& x){require_real_fp32(x);return x.mul(hardsigmoid_general(x));}
at::Tensor hardswish_grad(const at::Tensor& g,const at::Tensor& x){return at::where(x.le(-3),at::zeros({},g.options()),at::where(x.lt(3),g.mul(x.div(3).add(.5)),g));}
at::Tensor mish_general(const at::Tensor& x){require_real_fp32(x);return x.mul(at::tanh(softplus_general(x,1,20)));}
at::Tensor mish_grad(const at::Tensor& g,const at::Tensor& x){auto t=at::tanh(softplus_general(x,1,20));return g.mul(t.add(x.mul(at::sigmoid(x)).mul(at::rsub(t.mul(t),1))));}
at::Tensor threshold_general(const at::Tensor& x,const at::Scalar& threshold,const at::Scalar& value){return at::where(x.le(threshold),at::full({},value,x.options()),x);}
at::Tensor hardtanh_grad(const at::Tensor& g,const at::Tensor& x,const at::Scalar& low,const at::Scalar& high){return at::where(at::logical_or(x.le(low),x.ge(high)),at::zeros({},g.options()),g);}
at::Tensor logit_general(const at::Tensor& raw,std::optional<double> epsilon){auto x=raw.to(at::kFloat);if(epsilon&&*epsilon>=0)x=at::clamp(x,*epsilon,1-*epsilon);return at::log(x.div(at::rsub(x,1)));}
at::Tensor logit_grad(const at::Tensor& g,const at::Tensor& x,std::optional<double> epsilon){auto value=g.div(x.mul(at::rsub(x,1)));if(epsilon&&*epsilon>=0)return at::where(at::logical_or(x.lt(*epsilon),x.gt(1-*epsilon)),at::zeros({},g.options()),value);return at::where(at::logical_or(x.lt(0),x.gt(1)),at::full({},NAN,g.options()),value);}
at::Tensor glu_general(const at::Tensor& x,int64_t raw){require_real_fp32(x);TORCH_CHECK(x.dim()>0,"glu does not support 0-dimensional tensors");auto d=at::maybe_wrap_dim(raw,x.dim()),n=x.size(d);TORCH_CHECK(n%2==0,"Halving dimension must be even");return x.narrow(d,0,n/2).mul(at::sigmoid(x.narrow(d,n/2,n/2)));}
at::Tensor glu_grad(const at::Tensor& g,const at::Tensor& x,int64_t raw){auto d=at::maybe_wrap_dim(raw,x.dim()),n=x.size(d)/2;auto a=x.narrow(d,0,n),s=at::sigmoid(x.narrow(d,n,n));return at::cat({g.mul(s),g.mul(a).mul(s).mul(at::rsub(s,1))},d);}
at::Tensor nan_to_num_general(const at::Tensor& x,std::optional<double> nan,std::optional<double> pos,std::optional<double> neg){if(!x.is_floating_point()&&!x.is_complex())return x.clone();auto data=x.is_complex()?at::view_as_real(x):x;TORCH_CHECK(data.scalar_type()==at::kFloat,"nan_to_num requires FP32 or complex64");auto value=at::where(at::isnan(data),at::full({},nan.value_or(0),data.options()),data);value=at::where(at::eq(data,INFINITY),at::full({},pos.value_or(std::numeric_limits<float>::max()),data.options()),value);value=at::where(at::eq(data,-INFINITY),at::full({},neg.value_or(-std::numeric_limits<float>::max()),data.options()),value);return x.is_complex()?at::view_as_complex(value):value;}
at::Tensor prelu_general(const at::Tensor& x,const at::Tensor& weight){require_real_fp32(x);return at::where(x.gt(0),x,x.mul(weight));}
std::tuple<at::Tensor,at::Tensor> prelu_grad(const at::Tensor& g,const at::Tensor& x,const at::Tensor& weight){auto positive=x.gt(0);return {g.mul(at::where(positive,at::ones({},x.options()),weight)),g.mul(at::where(positive,at::zeros({},x.options()),x))};}
at::Tensor rrelu_general(const at::Tensor& x,at::Tensor& noise,const at::Scalar& lower,const at::Scalar& upper,bool training,std::optional<at::Generator> generator){require_real_fp32(x);TORCH_CHECK(lower.toDouble()<=upper.toDouble(),"rrelu lower must be <= upper");if(!training)return at::leaky_relu(x,(lower.toDouble()+upper.toDouble())*.5);TORCH_CHECK(noise.sizes()==x.sizes()&&noise.device()==x.device()&&noise.scalar_type()==x.scalar_type(),"rrelu noise mismatch");auto slope=at::empty_like(x).uniform_(lower.toDouble(),upper.toDouble(),generator);noise.copy_(at::where(x.le(0),slope,at::ones({},x.options())));return x.mul(noise);}
at::Tensor& rrelu_inplace(at::Tensor& x,at::Tensor& noise,const at::Scalar& low,const at::Scalar& high,bool training,std::optional<at::Generator> generator){at::assert_no_internal_overlap(x);x.copy_(rrelu_general(x,noise,low,high,training,generator));return x;}
at::Tensor rrelu_grad(const at::Tensor& g,const at::Tensor& x,const at::Tensor& noise,const at::Scalar& lower,const at::Scalar& upper,bool training,bool result){if(training&&std::abs(upper.toDouble()-lower.toDouble())>1e-6)return g.mul(noise);return at::leaky_relu_backward(g,x,(lower.toDouble()+upper.toDouble())*.5,result);}
at::Tensor mvlgamma_general(const at::Tensor& raw,int64_t p){auto x=raw.to(at::kFloat);TORCH_CHECK(p>=1,"p has to be greater than or equal to 1");TORCH_CHECK(at::all(x.gt((p-1)*.5)).cpu().item<bool>(),"All elements must be greater than (p-1)/2");auto result=at::zeros_like(x);for(int64_t i=0;i<p;i++)result=result.add(at::lgamma(x.sub(i*.5)));return result.add((p*(p-1)*.25)*std::log(3.141592653589793));}
at::Tensor& elu_general_inplace(at::Tensor& x,const at::Scalar& a,const at::Scalar& b,const at::Scalar& c){at::assert_no_internal_overlap(x);x.copy_(elu_general(x,a,b,c));return x;}
template<int Mode>at::Tensor& activation_inplace(at::Tensor& x){at::assert_no_internal_overlap(x);if constexpr(Mode==0)x.copy_(hardswish_general(x));else if constexpr(Mode==1)x.copy_(hardsigmoid_general(x));else x.copy_(mish_general(x));return x;}
at::Tensor& logit_inplace(at::Tensor& x,std::optional<double> eps){require_real_fp32(x);at::assert_no_internal_overlap(x);x.copy_(logit_general(x,eps));return x;}
at::Tensor& nan_to_num_inplace(at::Tensor& x,std::optional<double> a,std::optional<double> b,std::optional<double> c){at::assert_no_internal_overlap(x);x.copy_(nan_to_num_general(x,a,b,c));return x;}

at::Tensor glu_jvp_general(const at::Tensor& output,const at::Tensor& x,const at::Tensor& dx,int64_t raw){auto d=at::maybe_wrap_dim(raw,x.dim()),n=x.size(d)/2;auto s=at::sigmoid(x.narrow(d,n,n));return dx.narrow(d,0,n).mul(s).add(output.mul(at::rsub(s,1)).mul(dx.narrow(d,n,n)));}
at::Tensor angle_general(const at::Tensor& x){if(x.is_complex()){auto parts=at::view_as_real(at::resolve_neg(at::resolve_conj(x)));return at::atan2(parts.select(-1,1),parts.select(-1,0));}auto value=x.to(at::kFloat);return at::where(at::isnan(value),value,at::where(value.lt(0),at::full({},3.141592653589793,value.options()),at::zeros({},value.options())));}
std::tuple<at::Tensor,at::Tensor> frexp_general(const at::Tensor& x){TORCH_CHECK(x.scalar_type()==at::kFloat,"frexp expects a floating-point input");auto mantissa=at::empty(x.sizes(),x.options()),exponent=at::empty(x.sizes(),x.options().dtype(at::kInt));if(x.numel())run("frexp",{base(packed(x))},{write_base(mantissa),write_base(exponent,at::kInt)},{u32(x.numel())},(x.numel()+127)/128);return {mantissa,exponent};}
std::tuple<at::Tensor&,at::Tensor&> frexp_out_general(const at::Tensor& x,at::Tensor& mantissa,at::Tensor& exponent){TORCH_CHECK(mantissa.device()==x.device()&&exponent.device()==x.device()&&mantissa.scalar_type()==at::kFloat&&exponent.scalar_type()==at::kInt,"frexp output requires float32 mantissa and int32 exponent on the input device");at::assert_no_internal_overlap(mantissa);at::assert_no_internal_overlap(exponent);at::assert_no_partial_overlap(mantissa,x);at::assert_no_overlap(exponent,x);at::assert_no_overlap(mantissa,exponent);auto [m,e]=frexp_general(x);at::native::resize_output(mantissa,m.sizes());at::native::resize_output(exponent,e.sizes());mantissa.copy_(m);exponent.copy_(e);return {mantissa,exponent};}
}
TORCH_LIBRARY_IMPL(aten,PrivateUse1,m){
 m.impl("glu_jvp",&glu_jvp_general);
 m.impl("elu",&elu_general);m.impl("elu_backward",&elu_grad);m.impl("hardshrink",&shrink_general<false>);m.impl("softshrink",&shrink_general<true>);m.impl("hardshrink_backward",&shrink_grad);m.impl("softshrink_backward",&shrink_grad);
 m.impl("softplus",&softplus_general);m.impl("softplus_backward",&softplus_grad);m.impl("hardsigmoid",&hardsigmoid_general);m.impl("hardsigmoid_backward",&hardsigmoid_grad);m.impl("hardswish",&hardswish_general);m.impl("hardswish_backward",&hardswish_grad);m.impl("mish",&mish_general);m.impl("mish_backward",&mish_grad);
 m.impl("threshold",&threshold_general);m.impl("hardtanh_backward",&hardtanh_grad);m.impl("logit",&logit_general);m.impl("logit_backward",&logit_grad);m.impl("glu",&glu_general);m.impl("glu_backward",&glu_grad);m.impl("nan_to_num",&nan_to_num_general);
 m.impl("_prelu_kernel",&prelu_general);m.impl("_prelu_kernel_backward",&prelu_grad);m.impl("rrelu_with_noise",&rrelu_general);m.impl("rrelu_with_noise_",&rrelu_inplace);m.impl("rrelu_with_noise_backward",&rrelu_grad);m.impl("mvlgamma",&mvlgamma_general);
 m.impl("elu_",&elu_general_inplace);m.impl("hardswish_",&activation_inplace<0>);m.impl("hardsigmoid_",&activation_inplace<1>);m.impl("mish_",&activation_inplace<2>);m.impl("logit_",&logit_inplace);m.impl("nan_to_num_",&nan_to_num_inplace);

 m.impl("angle",&angle_general);m.impl("frexp.Tensor",&frexp_general);m.impl("frexp.Tensor_out",&frexp_out_general);m.impl("frac",&math_unary<26>);m.impl("frac_",&math_unary_inplace<26>);m.impl("sinc",&math_unary<27>);m.impl("sinc_",&math_unary_inplace<27>);m.impl("special_entr",&math_unary<28>);
 m.impl("gcd",&integer_binary_extra<5>);m.impl("lcm",&integer_binary_extra<6>);m.impl("gcd_",&integer_binary_extra_inplace<5>);m.impl("lcm_",&integer_binary_extra_inplace<6>);
 m.impl("atan2",&binary_math<10>);m.impl("atan2_",&binary_math_inplace<10>);m.impl("special_xlog1py",&binary_math<11>);m.impl("special_xlog1py.other_scalar",&binary_math_scalar<11>);m.impl("special_xlog1py.self_scalar",&scalar_left_math<11>);
 m.impl("pow.Scalar",&scalar_left_math<2>);m.impl("remainder.Scalar_Tensor",&scalar_left_math<1>);m.impl("xlogy.Scalar_Self",&scalar_left_math<8>);
 m.impl("fmod.Tensor",&binary_math<0>);m.impl("fmod.Scalar",&binary_math_scalar<0>);m.impl("fmod_.Tensor",&binary_math_inplace<0>);m.impl("fmod_.Scalar",&binary_math_scalar_inplace<0>);
 m.impl("remainder.Tensor",&binary_math<1>);m.impl("remainder.Scalar",&binary_math_scalar<1>);m.impl("remainder_.Tensor",&binary_math_inplace<1>);m.impl("remainder_.Scalar",&binary_math_scalar_inplace<1>);
 m.impl("pow.Tensor_Tensor",&binary_math<2>);m.impl("pow_.Tensor",&binary_math_inplace<2>);
 m.impl("floor_divide",&binary_math<3>);m.impl("floor_divide.Scalar",&binary_math_scalar<3>);m.impl("floor_divide_.Tensor",&binary_math_inplace<3>);m.impl("floor_divide_.Scalar",&binary_math_scalar_inplace<3>);
 m.impl("div.Tensor_mode",&div_mode);m.impl("div.Scalar_mode",&div_scalar_mode);m.impl("div_.Tensor_mode",&div_mode_inplace);m.impl("div_.Scalar_mode",&div_scalar_mode_inplace);
 m.impl("hypot",&binary_math<4>);m.impl("hypot_",&binary_math_inplace<4>);m.impl("copysign.Tensor",&binary_math<5>);m.impl("copysign.Scalar",&binary_math_scalar<5>);m.impl("copysign_.Tensor",&binary_math_inplace<5>);m.impl("copysign_.Scalar",&binary_math_scalar_inplace<5>);
 m.impl("logaddexp",&binary_math<6>);m.impl("logaddexp2",&binary_math<7>);m.impl("xlogy.Tensor",&binary_math<8>);m.impl("xlogy.Scalar_Other",&binary_math_scalar<8>);m.impl("xlogy_.Tensor",&binary_math_inplace<8>);m.impl("xlogy_.Scalar_Other",&binary_math_scalar_inplace<8>);m.impl("nextafter",&binary_math<9>);m.impl("nextafter_",&binary_math_inplace<9>);
 m.impl("lgamma",&lgamma_native);m.impl("lgamma_",&lgamma_inplace);m.impl("digamma",&digamma_native);m.impl("digamma_",&digamma_inplace);m.impl("polygamma",&polygamma_native);m.impl("polygamma_",&polygamma_inplace);
#define MATH(name,op) m.impl(name,&math_unary<op>);m.impl(name "_",&math_unary_inplace<op>);m.impl(name ".out",&math_unary_out<op>);
 MATH("reciprocal",0) MATH("log",1) MATH("log1p",2) MATH("expm1",3)
 MATH("floor",4) MATH("ceil",5) MATH("trunc",6) MATH("round",7) MATH("sign",8)
 MATH("log10",9) MATH("log2",10) MATH("exp2",11) MATH("sinh",12) MATH("cosh",13)
 MATH("tan",14) MATH("asin",15) MATH("acos",16) MATH("atan",17)
 MATH("asinh",18) MATH("acosh",19) MATH("atanh",20) MATH("erf",21) MATH("erfc",22)
#undef MATH
 m.impl("round.decimals",&round_decimals);m.impl("round_.decimals",&round_decimals_inplace);
}
}

#include "native_utils.h"
#include "device_copy.h"
#include <ATen/ExpandUtils.h>
#include <ATen/MemoryOverlap.h>
#include <torch/library.h>
namespace torch_vulkan::ops {namespace {
using namespace native;
template<int Op>at::Tensor special_unary(const at::Tensor& raw){TORCH_CHECK(device_copy_supported(raw.scalar_type()),"special functions require real inputs");auto x=raw.to(at::kFloat),out=at::empty(raw.sizes(),x.options());if(out.numel()){auto input=packed(x);auto dummy=VulkanAllocator::instance().constant_float_buffer(1,0);run("special_functions",{base(input),dummy},{write_base(out)},{u32(out.numel()),Op},(out.numel()+127)/128);}return out;}
template<int Op>at::Tensor& special_inplace(at::Tensor& x){at::assert_no_internal_overlap(x);auto out=special_unary<Op>(x);TORCH_CHECK(c10::canCast(out.scalar_type(),x.scalar_type()),"result type cannot be cast to output type");x.copy_(out);return x;}
template<int Op>at::Tensor special_binary(const at::Tensor& left,const at::Tensor& right){auto dev=left.is_cpu()?right.device():left.device();TORCH_CHECK((left.device()==dev||(left.is_cpu()&&left.dim()==0))&&(right.device()==dev||(right.is_cpu()&&right.dim()==0)),"special function inputs must be on the same device");TORCH_CHECK(device_copy_supported(left.scalar_type())&&device_copy_supported(right.scalar_type()),"special functions require real inputs");auto shape=at::infer_size_dimvector(left.sizes(),right.sizes());auto a=packed(left.to(dev,at::kFloat).expand(shape)),b=packed(right.to(dev,at::kFloat).expand(shape)),out=at::empty(shape,a.options());if(out.numel())run(Op>=33?"incomplete_gamma":"special_functions",{base(a),base(b)},{write_base(out)},{u32(out.numel()),Op>=33?Op-33:Op},(out.numel()+127)/128);return out;}
template<int Op>at::Tensor special_right_scalar(const at::Tensor& x,const at::Scalar& n){return special_binary<Op>(x,at::scalar_tensor(n,at::TensorOptions().dtype(at::result_type(x,n))));}
template<int Op>at::Tensor special_left_scalar(const at::Scalar& x,const at::Tensor& n){return special_binary<Op>(at::scalar_tensor(x,at::TensorOptions().dtype(at::result_type(n,x))),n);}
}
TORCH_LIBRARY_IMPL(aten,PrivateUse1,m){
 m.impl("erfinv",&special_unary<0>);
 m.impl("i0",&special_unary<1>);
 m.impl("special_i0e",&special_unary<2>);
 m.impl("special_i1",&special_unary<3>);
 m.impl("special_i1e",&special_unary<4>);
 m.impl("special_ndtri",&special_unary<5>);
 m.impl("special_erfcx",&special_unary<6>);
 m.impl("special_log_ndtr",&special_unary<7>);
 m.impl("special_airy_ai",&special_unary<8>);
 m.impl("special_bessel_j0",&special_unary<9>);
 m.impl("special_bessel_j1",&special_unary<10>);
 m.impl("special_bessel_y0",&special_unary<11>);
 m.impl("special_bessel_y1",&special_unary<12>);
 m.impl("special_spherical_bessel_j0",&special_unary<13>);
 m.impl("special_modified_bessel_i0",&special_unary<14>);
 m.impl("special_modified_bessel_i1",&special_unary<15>);
 m.impl("special_modified_bessel_k0",&special_unary<16>);
 m.impl("special_modified_bessel_k1",&special_unary<17>);
 m.impl("special_scaled_modified_bessel_k0",&special_unary<18>);
 m.impl("special_scaled_modified_bessel_k1",&special_unary<19>);
 m.impl("erfinv_",&special_inplace<0>);m.impl("i0_",&special_inplace<1>);
 m.impl("special_chebyshev_polynomial_t",&special_binary<20>);
 m.impl("special_chebyshev_polynomial_t.n_scalar",&special_right_scalar<20>);m.impl("special_chebyshev_polynomial_t.x_scalar",&special_left_scalar<20>);
 m.impl("special_chebyshev_polynomial_u",&special_binary<21>);
 m.impl("special_chebyshev_polynomial_u.n_scalar",&special_right_scalar<21>);m.impl("special_chebyshev_polynomial_u.x_scalar",&special_left_scalar<21>);
 m.impl("special_chebyshev_polynomial_v",&special_binary<22>);
 m.impl("special_chebyshev_polynomial_v.n_scalar",&special_right_scalar<22>);m.impl("special_chebyshev_polynomial_v.x_scalar",&special_left_scalar<22>);
 m.impl("special_chebyshev_polynomial_w",&special_binary<23>);
 m.impl("special_chebyshev_polynomial_w.n_scalar",&special_right_scalar<23>);m.impl("special_chebyshev_polynomial_w.x_scalar",&special_left_scalar<23>);
 m.impl("special_hermite_polynomial_h",&special_binary<24>);
 m.impl("special_hermite_polynomial_h.n_scalar",&special_right_scalar<24>);m.impl("special_hermite_polynomial_h.x_scalar",&special_left_scalar<24>);
 m.impl("special_hermite_polynomial_he",&special_binary<25>);
 m.impl("special_hermite_polynomial_he.n_scalar",&special_right_scalar<25>);m.impl("special_hermite_polynomial_he.x_scalar",&special_left_scalar<25>);
 m.impl("special_laguerre_polynomial_l",&special_binary<26>);
 m.impl("special_laguerre_polynomial_l.n_scalar",&special_right_scalar<26>);m.impl("special_laguerre_polynomial_l.x_scalar",&special_left_scalar<26>);
 m.impl("special_legendre_polynomial_p",&special_binary<27>);
 m.impl("special_legendre_polynomial_p.n_scalar",&special_right_scalar<27>);m.impl("special_legendre_polynomial_p.x_scalar",&special_left_scalar<27>);
 m.impl("special_shifted_chebyshev_polynomial_t",&special_binary<28>);
 m.impl("special_shifted_chebyshev_polynomial_t.n_scalar",&special_right_scalar<28>);m.impl("special_shifted_chebyshev_polynomial_t.x_scalar",&special_left_scalar<28>);
 m.impl("special_shifted_chebyshev_polynomial_u",&special_binary<29>);
 m.impl("special_shifted_chebyshev_polynomial_u.n_scalar",&special_right_scalar<29>);m.impl("special_shifted_chebyshev_polynomial_u.x_scalar",&special_left_scalar<29>);
 m.impl("special_shifted_chebyshev_polynomial_v",&special_binary<30>);
 m.impl("special_shifted_chebyshev_polynomial_v.n_scalar",&special_right_scalar<30>);m.impl("special_shifted_chebyshev_polynomial_v.x_scalar",&special_left_scalar<30>);
 m.impl("special_shifted_chebyshev_polynomial_w",&special_binary<31>);
 m.impl("special_shifted_chebyshev_polynomial_w.n_scalar",&special_right_scalar<31>);m.impl("special_shifted_chebyshev_polynomial_w.x_scalar",&special_left_scalar<31>);
 m.impl("special_zeta",&special_binary<32>);
 m.impl("special_zeta.other_scalar",&special_right_scalar<32>);m.impl("special_zeta.self_scalar",&special_left_scalar<32>);
}
}

namespace torch_vulkan::ops {
TORCH_LIBRARY_IMPL(aten,PrivateUse1,m){m.impl("igamma",&special_binary<33>);m.impl("igammac",&special_binary<34>);}
}
