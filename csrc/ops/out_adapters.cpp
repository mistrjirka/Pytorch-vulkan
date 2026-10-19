// Adapt out= variants to existing native kernels with shared alias/cast checks.
// Register only kernels with a direct PrivateUse1 functional implementation:
// adapting composite functions that call their own out variant would recurse.
#include "native_utils.h"
#include <ATen/MemoryOverlap.h>
#include <cctype>
#include <ATen/native/Resize.h>
#include <torch/library.h>
#include <ATen/core/grad_mode.h>
namespace torch_vulkan::ops {
namespace {
struct NativeOut : c10::OperatorKernel {
 std::string name,overload;
 NativeOut(std::string n,std::string o):name(std::move(n)),overload(std::move(o)){}
 void operator()(const c10::OperatorHandle& op,c10::DispatchKeySet,torch::jit::Stack* stack){
  auto count=op.schema().arguments().size(),noutputs=op.schema().returns().size();auto begin=stack->size()-count;
  std::vector<at::Tensor> outputs;for(size_t i=stack->size()-noutputs;i<stack->size();i++){auto out=(*stack)[i].toTensor();at::assert_no_internal_overlap(out);for(const auto& previous:outputs)at::assert_no_overlap(out,previous);outputs.push_back(out);}
  if(name=="aten::hash_tensor")for(const auto& out:outputs)TORCH_CHECK(out.scalar_type()==at::kUInt64,"Expected result to be of dtype uint64, but got ",out.scalar_type());
  torch::jit::Stack args(stack->begin()+begin,stack->end()-noutputs);
  if(name=="aten::aminmax"||name=="aten::avg_pool2d"||name=="aten::avg_pool3d"){auto dtype=args[0].toTensor().scalar_type();for(const auto& out:outputs)if(out.scalar_type()!=dtype){auto expected=std::string(c10::toString(dtype)),actual=std::string(c10::toString(out.scalar_type()));for(auto& c:expected)c=std::tolower(c);for(auto& c:actual)c=std::tolower(c);TORCH_CHECK(false,"Expected out tensor to have dtype ",expected,", but got ",actual," instead");}}

  for(const auto& out:outputs)for(const auto& v:args){
    auto validate=[&](const at::Tensor& t){if(!t.defined())return;if(name=="aten::cat"){TORCH_CHECK(!t.is_same(out),"unsupported operation: some elements of the input tensor and the written-to tensor refer to a single memory location.");at::assert_no_overlap(out,t);}else if(name=="aten::kthvalue"||name=="aten::median"||name=="aten::nanmedian"||name=="aten::mode")at::assert_no_overlap(out,t);else at::assert_no_partial_overlap(out,t);TORCH_CHECK(t.device()==out.device()||(t.is_cpu()&&t.dim()==0),"out tensor must be on the same device as the input");};
    if(v.isTensor())validate(v.toTensor());else if(v.isTensorList())for(const auto& t:v.toTensorVector())validate(t);
  }
  const auto target=c10::Dispatcher::singleton().findSchemaOrThrow(name.c_str(),overload.c_str());
  TORCH_INTERNAL_ASSERT(target.schema().arguments().size()==args.size());target.callBoxed(&args);
  TORCH_INTERNAL_ASSERT(args.size()==outputs.size());
  for(size_t i=0;i<outputs.size();i++){TORCH_INTERNAL_ASSERT(args[i].isTensor());if(name=="aten::_chunk_cat"||name=="aten::cat")TORCH_CHECK_TYPE(c10::canCast(args[i].toTensor().scalar_type(),outputs[i].scalar_type()),"torch.cat(): input types can't be cast to the desired output type ",outputs[i].scalar_type());TORCH_CHECK(c10::canCast(args[i].toTensor().scalar_type(),outputs[i].scalar_type()),"result type cannot be cast to the desired output type");}
  for(size_t i=0;i<outputs.size();i++)for(size_t j=begin;j<stack->size()-noutputs;j++){
   auto check=[&](const at::Tensor& input){if(input.defined()&&input.is_same(outputs[i]))TORCH_CHECK(outputs[i].sizes()==args[i].toTensor().sizes(),"output with shape ",outputs[i].sizes()," does not match the broadcast shape ",args[i].toTensor().sizes());};
   if((*stack)[j].isTensor())check((*stack)[j].toTensor());else if((*stack)[j].isTensorList())for(const auto& t:(*stack)[j].toTensorVector())check(t);
  }
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
 add("linalg_eig.out","linalg_eig");add("linalg_eigvals.out","_linalg_eigvals");
 add("_linalg_det.result","_linalg_det");add("_linalg_slogdet.sign","_linalg_slogdet");add("_linalg_eigh.eigenvalues","_linalg_eigh");
 add("gelu_backward.grad_input","gelu_backward");add("complex.out","complex");add("polar.out","polar");
 add("geqrf.a","geqrf");add("ormqr.out","ormqr");add("linalg_householder_product.out","linalg_householder_product");add("linalg_qr.out","linalg_qr");
 add("linalg_polar.out","linalg_polar");add("glu_jvp.out","glu_jvp");
 add("searchsorted.Tensor_out","searchsorted","Tensor");add("searchsorted.Scalar_out","searchsorted","Scalar");add("bucketize.Tensor_out","bucketize","Tensor");add("bucketize.Scalar_out","bucketize","Scalar");
 add("angle.out","angle");add("atan2.out","atan2");add("frac.out","frac");add("sinc.out","sinc");add("special_entr.out","special_entr");add("special_xlog1py.out","special_xlog1py");add("special_xlog1py.self_scalar_out","special_xlog1py","self_scalar");add("special_xlog1py.other_scalar_out","special_xlog1py","other_scalar");
 add("upsample_nearest1d.out","upsample_nearest1d");add("_upsample_nearest_exact1d.out","_upsample_nearest_exact1d");add("upsample_linear1d.out","upsample_linear1d");add("upsample_nearest1d_backward.grad_input","upsample_nearest1d_backward");add("_upsample_nearest_exact1d_backward.grad_input","_upsample_nearest_exact1d_backward");add("upsample_linear1d_backward.grad_input","upsample_linear1d_backward");
 for(auto name:{"upsample_nearest2d","upsample_nearest3d","_upsample_nearest_exact2d","_upsample_nearest_exact3d","upsample_bilinear2d","upsample_bicubic2d","upsample_trilinear3d","_upsample_bilinear2d_aa","_upsample_bicubic2d_aa"}){add((std::string(name)+".out").c_str(),name);add((std::string(name)+"_backward.grad_input").c_str(),(std::string(name)+"_backward").c_str());}
 add("kthvalue.values","kthvalue");add("median.out","median");add("median.dim_values","median","dim");add("nanmedian.out","nanmedian");add("nanmedian.dim_values","nanmedian","dim");add("mode.values","mode");
 add("isin.Tensor_Tensor_out","isin","Tensor_Tensor");add("isin.Tensor_Scalar_out","isin","Tensor_Scalar");add("isin.Scalar_Tensor_out","isin","Scalar_Tensor");add("repeat_interleave.Tensor_out","repeat_interleave","Tensor");add("renorm.out","renorm");add("addr.out","addr");
 add("smooth_l1_loss.out","smooth_l1_loss");add("smooth_l1_loss_backward.grad_input","smooth_l1_loss_backward");add("huber_loss.out","huber_loss");add("huber_loss_backward.out","huber_loss_backward");add("binary_cross_entropy.out","binary_cross_entropy");add("binary_cross_entropy_backward.grad_input","binary_cross_entropy_backward");
 add("native_batch_norm.out","native_batch_norm");add("_native_batch_norm_legit.out","_native_batch_norm_legit");add("_native_batch_norm_legit.no_stats_out","_native_batch_norm_legit","no_stats");add("_native_batch_norm_legit_no_training.out","_native_batch_norm_legit_no_training");add("_batch_norm_with_update.out","_batch_norm_with_update");
 add("erfinv.out","erfinv");
 add("i0.out","i0");
 add("special_i0e.out","special_i0e");
 add("special_i1.out","special_i1");
 add("special_i1e.out","special_i1e");
 add("special_ndtri.out","special_ndtri");
 add("special_erfcx.out","special_erfcx");
 add("special_log_ndtr.out","special_log_ndtr");
 add("special_airy_ai.out","special_airy_ai");
 add("special_bessel_j0.out","special_bessel_j0");
 add("special_bessel_j1.out","special_bessel_j1");
 add("special_bessel_y0.out","special_bessel_y0");
 add("special_bessel_y1.out","special_bessel_y1");
 add("special_spherical_bessel_j0.out","special_spherical_bessel_j0");
 add("special_modified_bessel_i0.out","special_modified_bessel_i0");
 add("special_modified_bessel_i1.out","special_modified_bessel_i1");
 add("special_modified_bessel_k0.out","special_modified_bessel_k0");
 add("special_modified_bessel_k1.out","special_modified_bessel_k1");
 add("special_scaled_modified_bessel_k0.out","special_scaled_modified_bessel_k0");
 add("special_scaled_modified_bessel_k1.out","special_scaled_modified_bessel_k1");
 add("special_chebyshev_polynomial_t.out","special_chebyshev_polynomial_t");
 add("special_chebyshev_polynomial_t.n_scalar_out","special_chebyshev_polynomial_t","n_scalar");
 add("special_chebyshev_polynomial_t.x_scalar_out","special_chebyshev_polynomial_t","x_scalar");
 add("special_chebyshev_polynomial_u.out","special_chebyshev_polynomial_u");
 add("special_chebyshev_polynomial_u.n_scalar_out","special_chebyshev_polynomial_u","n_scalar");
 add("special_chebyshev_polynomial_u.x_scalar_out","special_chebyshev_polynomial_u","x_scalar");
 add("special_chebyshev_polynomial_v.out","special_chebyshev_polynomial_v");
 add("special_chebyshev_polynomial_v.n_scalar_out","special_chebyshev_polynomial_v","n_scalar");
 add("special_chebyshev_polynomial_v.x_scalar_out","special_chebyshev_polynomial_v","x_scalar");
 add("special_chebyshev_polynomial_w.out","special_chebyshev_polynomial_w");
 add("special_chebyshev_polynomial_w.n_scalar_out","special_chebyshev_polynomial_w","n_scalar");
 add("special_chebyshev_polynomial_w.x_scalar_out","special_chebyshev_polynomial_w","x_scalar");
 add("special_hermite_polynomial_h.out","special_hermite_polynomial_h");
 add("special_hermite_polynomial_h.n_scalar_out","special_hermite_polynomial_h","n_scalar");
 add("special_hermite_polynomial_h.x_scalar_out","special_hermite_polynomial_h","x_scalar");
 add("special_hermite_polynomial_he.out","special_hermite_polynomial_he");
 add("special_hermite_polynomial_he.n_scalar_out","special_hermite_polynomial_he","n_scalar");
 add("special_hermite_polynomial_he.x_scalar_out","special_hermite_polynomial_he","x_scalar");
 add("special_laguerre_polynomial_l.out","special_laguerre_polynomial_l");
 add("special_laguerre_polynomial_l.n_scalar_out","special_laguerre_polynomial_l","n_scalar");
 add("special_laguerre_polynomial_l.x_scalar_out","special_laguerre_polynomial_l","x_scalar");
 add("special_legendre_polynomial_p.out","special_legendre_polynomial_p");
 add("special_legendre_polynomial_p.n_scalar_out","special_legendre_polynomial_p","n_scalar");
 add("special_legendre_polynomial_p.x_scalar_out","special_legendre_polynomial_p","x_scalar");
 add("special_shifted_chebyshev_polynomial_t.out","special_shifted_chebyshev_polynomial_t");
 add("special_shifted_chebyshev_polynomial_t.n_scalar_out","special_shifted_chebyshev_polynomial_t","n_scalar");
 add("special_shifted_chebyshev_polynomial_t.x_scalar_out","special_shifted_chebyshev_polynomial_t","x_scalar");
 add("special_shifted_chebyshev_polynomial_u.out","special_shifted_chebyshev_polynomial_u");
 add("special_shifted_chebyshev_polynomial_u.n_scalar_out","special_shifted_chebyshev_polynomial_u","n_scalar");
 add("special_shifted_chebyshev_polynomial_u.x_scalar_out","special_shifted_chebyshev_polynomial_u","x_scalar");
 add("special_shifted_chebyshev_polynomial_v.out","special_shifted_chebyshev_polynomial_v");
 add("special_shifted_chebyshev_polynomial_v.n_scalar_out","special_shifted_chebyshev_polynomial_v","n_scalar");
 add("special_shifted_chebyshev_polynomial_v.x_scalar_out","special_shifted_chebyshev_polynomial_v","x_scalar");
 add("special_shifted_chebyshev_polynomial_w.out","special_shifted_chebyshev_polynomial_w");
 add("special_shifted_chebyshev_polynomial_w.n_scalar_out","special_shifted_chebyshev_polynomial_w","n_scalar");
 add("special_shifted_chebyshev_polynomial_w.x_scalar_out","special_shifted_chebyshev_polynomial_w","x_scalar");
 add("special_zeta.out","special_zeta");
 add("special_zeta.other_scalar_out","special_zeta","other_scalar");
 add("special_zeta.self_scalar_out","special_zeta","self_scalar");
 add("gcd.out","gcd");add("lcm.out","lcm");
 for(auto name:{"bitwise_left_shift","bitwise_right_shift"}){add((std::string(name)+".Tensor_out").c_str(),name,"Tensor");add((std::string(name)+".Tensor_Scalar_out").c_str(),name,"Tensor_Scalar");add((std::string(name)+".Scalar_Tensor_out").c_str(),name,"Scalar_Tensor");}
 add("normal.Tensor_float_out","normal","Tensor_float");add("normal.float_Tensor_out","normal","float_Tensor");add("normal.Tensor_Tensor_out","normal","Tensor_Tensor");add("bernoulli.out","bernoulli");
 for(auto name:{"linalg_pinv","linalg_matrix_rank"}){add((std::string(name)+".out").c_str(),name);add((std::string(name)+".atol_rtol_tensor_out").c_str(),name,"atol_rtol_tensor");add((std::string(name)+".atol_rtol_float_out").c_str(),name,"atol_rtol_float");}
 add("linalg_pinv.out_rcond_tensor","linalg_pinv","rcond_tensor");add("linalg_matrix_rank.out_tol_tensor","linalg_matrix_rank","tol_tensor");
 add("igamma.out","igamma");add("igammac.out","igammac");add("linalg_lstsq.out","linalg_lstsq");
 add("im2col.out","im2col");add("col2im.out","col2im");add("heaviside.out","heaviside");add("linalg_matrix_sqrth.out","linalg_matrix_sqrth");add("linalg_matrix_exp.out","linalg_matrix_exp");add("triangular_solve.X","triangular_solve");add("_cdist_forward.out","_cdist_forward");add("_cdist_backward.out","_cdist_backward");
 for(auto name:{"quantile","nanquantile"}){add((std::string(name)+".out").c_str(),name);add((std::string(name)+".scalar_out").c_str(),name,"scalar");}
 add("multi_margin_loss.out","multi_margin_loss");add("multi_margin_loss_backward.grad_input","multi_margin_loss_backward");add("multilabel_margin_loss_forward.output","multilabel_margin_loss_forward");add("multilabel_margin_loss_backward.grad_input","multilabel_margin_loss_backward");
 add("hash_tensor.out","hash_tensor");
 add("linalg_ldl_factor_ex.out","linalg_ldl_factor_ex");add("linalg_ldl_solve.out","linalg_ldl_solve");
 add("_ctc_loss.out","_ctc_loss");add("_ctc_loss.Tensor_out","_ctc_loss","Tensor");add("_ctc_loss_backward.out","_ctc_loss_backward");
 add("fractional_max_pool2d.output","fractional_max_pool2d");add("fractional_max_pool3d.output","fractional_max_pool3d");add("fractional_max_pool2d_backward.grad_input","fractional_max_pool2d_backward");add("fractional_max_pool3d_backward.grad_input","fractional_max_pool3d_backward");
 add("_pdist_forward.out","_pdist_forward");add("_pdist_backward.out","_pdist_backward");add("log_sigmoid_forward.output","log_sigmoid_forward");add("log_sigmoid_backward.grad_input","log_sigmoid_backward");
 add("histc.out","histc");add("histogram.bins_tensor_out","histogram","bins_tensor");add("histogram.bin_ct_out","histogram","bin_ct");add("_histogramdd_from_bin_cts.out","_histogramdd_from_bin_cts");add("_histogramdd_from_bin_tensors.out","_histogramdd_from_bin_tensors");
 add("avg_pool2d.out","avg_pool2d");add("avg_pool3d.out","avg_pool3d");add("avg_pool2d_backward.grad_input","avg_pool2d_backward");add("avg_pool3d_backward.grad_input","avg_pool3d_backward");
 add("cat.out","cat");add("nansum.out","nansum");add("std.correction_out","std","correction");add("var.correction_out","var","correction");
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
