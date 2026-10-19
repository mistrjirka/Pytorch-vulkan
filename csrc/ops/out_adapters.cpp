// Adapt out= variants to existing native kernels with shared alias/cast checks.
// Register only kernels with a direct PrivateUse1 functional implementation:
// adapting composite functions that call their own out variant would recurse.
#include "native_utils.h"
#include <ATen/MemoryOverlap.h>
#include <ATen/native/Resize.h>
#include <torch/library.h>
namespace torch_vulkan::ops {
namespace {
struct NativeOut : c10::OperatorKernel {
 std::string name,overload;
 NativeOut(std::string n,std::string o):name(std::move(n)),overload(std::move(o)){}
 void operator()(const c10::OperatorHandle& op,c10::DispatchKeySet,torch::jit::Stack* stack){
  auto count=op.schema().arguments().size(),noutputs=op.schema().returns().size();auto begin=stack->size()-count;
  std::vector<at::Tensor> outputs;for(size_t i=stack->size()-noutputs;i<stack->size();i++){auto out=(*stack)[i].toTensor();at::assert_no_internal_overlap(out);for(const auto& previous:outputs)at::assert_no_overlap(out,previous);outputs.push_back(out);}
  torch::jit::Stack args(stack->begin()+begin,stack->end()-noutputs);
  for(const auto& out:outputs)for(const auto& v:args){
    auto validate=[&](const at::Tensor& t){at::assert_no_partial_overlap(out,t);TORCH_CHECK(t.device()==out.device()||(t.is_cpu()&&t.dim()==0),"out tensor must be on the same device as the input");};
    if(v.isTensor())validate(v.toTensor());else if(v.isTensorList())for(const auto& t:v.toTensorVector())validate(t);
  }
  const auto target=c10::Dispatcher::singleton().findSchemaOrThrow(name.c_str(),overload.c_str());
  TORCH_INTERNAL_ASSERT(target.schema().arguments().size()==args.size());target.callBoxed(&args);
  TORCH_INTERNAL_ASSERT(args.size()==outputs.size());
  for(size_t i=0;i<outputs.size();i++){TORCH_INTERNAL_ASSERT(args[i].isTensor());TORCH_CHECK(c10::canCast(args[i].toTensor().scalar_type(),outputs[i].scalar_type()),"result type cannot be cast to the desired output type");}
  for(size_t i=0;i<outputs.size();i++){auto value=args[i].toTensor();auto& out=outputs[i];at::native::resize_output(out,value.sizes());out.copy_(value);}
  stack->resize(begin);for(const auto& out:outputs)stack->push_back(out);
 }
};
}
TORCH_LIBRARY_IMPL(aten,PrivateUse1,m){
 auto add=[&](const char* out,const char* name,const char* overload=""){
  m.impl(out,torch::CppFunction::makeFromBoxedFunctor(std::make_unique<NativeOut>(std::string("aten::")+name,overload)));
 };
 for(auto name:{"abs","sgn","exp","sqrt","rsqrt","sin","cos","tanh","sigmoid","neg","silu","gelu","mm","bmm","addmm","clamp"})add((std::string(name)+".out").c_str(),name);
 add("bitwise_and.Tensor_out","bitwise_and","Tensor");add("bitwise_or.Tensor_out","bitwise_or","Tensor");add("bitwise_xor.Tensor_out","bitwise_xor","Tensor");add("bitwise_not.out","bitwise_not");
 add("bitwise_and.Scalar_out","bitwise_and","Scalar");add("bitwise_or.Scalar_out","bitwise_or","Scalar");add("bitwise_xor.Scalar_out","bitwise_xor","Scalar");
 for(auto name:{"reciprocal","log","log1p","expm1","floor","ceil","trunc","round","sign","log10","log2","exp2","sinh","cosh","tan","asin","acos","atan","asinh","acosh","atanh","erf","erfc"})add((std::string(name)+".out").c_str(),name);
 add("round.decimals_out","round","decimals");
 add("lu_unpack.out","lu_unpack");add("linalg_lu.out","linalg_lu");
 add("linalg_solve_triangular.out","linalg_solve_triangular");
 add("linalg_lu_factor_ex.out","linalg_lu_factor_ex");add("linalg_lu_solve.out","linalg_lu_solve");add("_linalg_solve_ex.result","_linalg_solve_ex");
 add("linalg_inv_ex.inverse","linalg_inv_ex");add("linalg_cholesky_ex.L","linalg_cholesky_ex");add("cholesky_solve.out","cholesky_solve");add("cholesky_inverse.out","cholesky_inverse");add("std_mean.correction_out","std_mean","correction");
 add("max.unary_out","max");add("min.unary_out","min");add("max.dim_max","max","dim");add("min.dim_min","min","dim");add("argmax.out","argmax");add("argmin.out","argmin");
 add("nll_loss_forward.output","nll_loss_forward");add("nll_loss_backward.grad_input","nll_loss_backward");add("nll_loss2d_forward.output","nll_loss2d_forward");add("nll_loss2d_backward.grad_input","nll_loss2d_backward");
 add("_softmax.out","_softmax");add("_log_softmax.out","_log_softmax");add("fmax.out","fmax");add("fmin.out","fmin");add("aminmax.out","aminmax");add("baddbmm.out","baddbmm");add("linalg_cross.out","linalg_cross");
 add("fmod.Tensor_out","fmod","Tensor");add("fmod.Scalar_out","fmod","Scalar");add("remainder.Tensor_out","remainder","Tensor");add("remainder.Scalar_out","remainder","Scalar");add("pow.Tensor_Tensor_out","pow","Tensor_Tensor");add("floor_divide.out","floor_divide");add("floor_divide.Scalar_out","floor_divide","Scalar");add("div.out_mode","div","Tensor_mode");add("div.Scalar_mode_out","div","Scalar_mode");
 add("hypot.out","hypot");add("copysign.out","copysign","Tensor");add("copysign.Scalar_out","copysign","Scalar");add("logaddexp.out","logaddexp");add("logaddexp2.out","logaddexp2");add("xlogy.OutTensor","xlogy","Tensor");add("xlogy.OutScalar_Other","xlogy","Scalar_Other");add("nextafter.out","nextafter");
 add("_unique2.out","_unique2");add("topk.values","topk");
 add("pow.Scalar_out","pow","Scalar");add("remainder.Scalar_Tensor_out","remainder","Scalar_Tensor");add("xlogy.OutScalar_Self","xlogy","Scalar_Self");
 add("lgamma.out","lgamma");add("digamma.out","digamma");add("polygamma.out","polygamma");
 add("prod.out","prod");add("prod.int_out","prod","dim_int");add("_unique.out","_unique");add("_logcumsumexp.out","_logcumsumexp");add("signbit.out","signbit");add("count_nonzero.dim_IntList_out","count_nonzero","dim_IntList");add("count_nonzero.out","count_nonzero");
 add("unique_dim.out","unique_dim");add("unique_consecutive.out","unique_consecutive");add("unique_dim_consecutive.out","unique_dim_consecutive");
 add("adaptive_max_pool2d.out","adaptive_max_pool2d");add("adaptive_max_pool2d_backward.grad_input","adaptive_max_pool2d_backward");add("adaptive_max_pool3d.out","adaptive_max_pool3d");add("adaptive_max_pool3d_backward.grad_input","adaptive_max_pool3d_backward");add("max_unpool2d.out","max_unpool2d");add("max_unpool3d.out","max_unpool3d");
 for(auto name:{"elu","hardshrink","softshrink","softplus","hardsigmoid","hardswish","mish","threshold","logit","glu","nan_to_num","mvlgamma","nonzero_static","trace","addbmm"})add((std::string(name)+".out").c_str(),name);
 add("elu_backward.grad_input","elu_backward");add("hardshrink_backward.grad_input","hardshrink_backward");add("softshrink_backward.grad_input","softshrink_backward");add("softplus_backward.grad_input","softplus_backward");add("hardsigmoid_backward.grad_input","hardsigmoid_backward");add("hardswish_backward.out","hardswish_backward");add("hardtanh_backward.grad_input","hardtanh_backward");add("logit_backward.grad_input","logit_backward");add("glu_backward.grad_input","glu_backward");add("rrelu_with_noise.out","rrelu_with_noise");add("rrelu_with_noise_backward.out","rrelu_with_noise_backward");
 add("_fft_r2c.out","_fft_r2c");add("_fft_c2c.out","_fft_c2c");add("_fft_c2r.out","_fft_c2r");add("conj_physical.out","_conj_physical");
 add("_linalg_det.result","_linalg_det");add("_linalg_slogdet.sign","_linalg_slogdet");add("_linalg_eigh.eigenvalues","_linalg_eigh");
 add("gelu_backward.grad_input","gelu_backward");add("complex.out","complex");add("polar.out","polar");
 add("geqrf.a","geqrf");add("ormqr.out","ormqr");add("linalg_householder_product.out","linalg_householder_product");add("linalg_qr.out","linalg_qr");
 add("linalg_polar.out","linalg_polar");add("glu_jvp.out","glu_jvp");
 add("searchsorted.Tensor_out","searchsorted","Tensor");add("searchsorted.Scalar_out","searchsorted","Scalar");add("bucketize.Tensor_out","bucketize","Tensor");add("bucketize.Scalar_out","bucketize","Scalar");
 add("angle.out","angle");add("atan2.out","atan2");add("frac.out","frac");add("sinc.out","sinc");add("special_entr.out","special_entr");add("special_xlog1py.out","special_xlog1py");add("special_xlog1py.self_scalar_out","special_xlog1py","self_scalar");add("special_xlog1py.other_scalar_out","special_xlog1py","other_scalar");
 add("upsample_nearest1d.out","upsample_nearest1d");add("_upsample_nearest_exact1d.out","_upsample_nearest_exact1d");add("upsample_linear1d.out","upsample_linear1d");add("upsample_nearest1d_backward.grad_input","upsample_nearest1d_backward");add("_upsample_nearest_exact1d_backward.grad_input","_upsample_nearest_exact1d_backward");add("upsample_linear1d_backward.grad_input","upsample_linear1d_backward");
 for(auto name:{"upsample_nearest2d","upsample_nearest3d","_upsample_nearest_exact2d","_upsample_nearest_exact3d","upsample_bilinear2d","upsample_bicubic2d","upsample_trilinear3d","_upsample_bilinear2d_aa","_upsample_bicubic2d_aa"}){add((std::string(name)+".out").c_str(),name);add((std::string(name)+"_backward.grad_input").c_str(),(std::string(name)+"_backward").c_str());}
 add("cat.out","cat");add("nansum.out","nansum");add("std.correction_out","std","correction");
 for(auto name:{"clamp_min","clamp_max"}){add((std::string(name)+".out").c_str(),name);add((std::string(name)+".Tensor_out").c_str(),name,"Tensor");}
 add("clamp.Tensor_out","clamp","Tensor");add("hardtanh.out","hardtanh");
 add("masked_fill.Scalar_out","masked_fill","Scalar");add("masked_fill.Tensor_out","masked_fill","Tensor");
 add("any.out","any","dim");add("any.all_out","any");add("any.dims_out","any","dims");add("all.dims_out","all","dims");
 add("rsub.Tensor_out","rsub","Tensor");add("rsub.Scalar_out","rsub","Scalar");add("addmv.out","addmv");
 add("add.out","add","Tensor");add("sub.out","sub","Tensor");add("mul.out","mul","Tensor");add("div.out","div","Tensor");
 add("pow.Tensor_Scalar_out","pow","Tensor_Scalar");
 add("sum.IntList_out","sum","dim_IntList");add("sum.out","sum");
 add("mean.out","mean","dim");add("mean.dtype_out","mean");
 add("all.out","all","dim");add("all.all_out","all");
 add("cumprod.out","cumprod");add("cummax.out","cummax");add("cummin.out","cummin");
 add("cumsum.out","cumsum");add("flip.out","flip");
}
}
