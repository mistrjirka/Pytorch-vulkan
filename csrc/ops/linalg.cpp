#include "native_utils.h"
#include <ATen/ExpandUtils.h>
#include <ATen/ops/_linalg_check_errors_native.h>
#include <torch/library.h>
#include <ATen/MemoryOverlap.h>
namespace torch_vulkan::ops {
namespace {
using namespace native;
at::Tensor& eye_out(int64_t n,int64_t m,at::Tensor& out){
 TORCH_CHECK(n>=0&&m>=0&&(out.scalar_type()==at::kFloat||out.scalar_type()==at::kInt||out.scalar_type()==at::kLong||out.scalar_type()==at::kComplexFloat),"Vulkan eye requires nonnegative sizes and FP32/complex64/int32/int64 output");
 out.resize_({n,m});at::assert_no_internal_overlap(out);out.zero_();if(std::min(n,m)>0)out.as_strided({std::min(n,m)},{out.stride(0)+out.stride(1)}).fill_(1);return out;
}
at::Tensor& eye_square(int64_t n,at::Tensor& out){return eye_out(n,n,out);}
void check_info(const at::Tensor& info,c10::string_view api,bool matrix){
 // These are per-matrix status integers, never the numeric matrix data.
 at::native::_linalg_check_errors(info.cpu(),api,matrix);
}
std::tuple<at::Tensor,at::Tensor,at::Tensor> lu_factor(const at::Tensor& a,bool pivot,bool check_errors){
 TORCH_CHECK(a.dim()>=2&&a.scalar_type()==at::kFloat&&can_pack_view(a),"Vulkan LU requires rank>=2 packable FP32 matrices");
 auto lu=at::empty(a.sizes(),a.options());int64_t m=a.size(-2),n=a.size(-1),k=std::min(m,n);
 auto batch=a.sizes().vec();batch.resize(a.dim()-2);auto pivshape=batch;pivshape.push_back(k);
 auto piv=at::empty(pivshape,a.options().dtype(at::kInt)),info=at::empty(batch,a.options().dtype(at::kInt));
 if(!info.numel())return {lu,piv,info};
 if(!k){info.zero_();return {lu,piv,info};}
 run("lu_factor",{base(packed(a))},{write_base(lu),write_base(piv,at::kInt),write_base(info,at::kInt)},
  {u32(info.numel()),u32(m),u32(n),uint32_t(pivot)},info.numel(),8);
 if(check_errors)check_info(info,"torch.linalg.lu_factor_ex",a.dim()==2);return {lu,piv,info};
}
at::Tensor lu_solve(const at::Tensor& lu,const at::Tensor& piv,const at::Tensor& b,bool left,bool adjoint){
 TORCH_CHECK(piv.dim()>=1&&lu.dim()>=2&&lu.size(-2)==lu.size(-1)&&b.dim()>=2&&b.scalar_type()==at::kFloat&&lu.scalar_type()==at::kFloat&&piv.scalar_type()==at::kInt&&lu.device()==b.device()&&piv.device()==lu.device(),"LU solve input mismatch");
 TORCH_CHECK(piv.sizes().slice(0,piv.dim()-1)==lu.sizes().slice(0,lu.dim()-2),"LU pivots batch shape mismatch");
 int64_t n=lu.size(-1);TORCH_CHECK((left?b.size(-2):b.size(-1))==n&&piv.dim()==lu.dim()-1&&piv.size(-1)==n,"LU solve shape mismatch");
 auto rhs=left?b:b.transpose(-2,-1);auto lb=lu.sizes().slice(0,lu.dim()-2),bb=rhs.sizes().slice(0,rhs.dim()-2);auto shape=at::infer_size_dimvector(lb,bb);auto ls=shape,ps=shape,bs=shape;ls.push_back(n);ls.push_back(n);ps.push_back(n);bs.push_back(n);bs.push_back(rhs.size(-1));
 auto out=at::empty(bs,b.options());if(!out.numel())return left?out:out.transpose(-2,-1);
 auto l=packed(lu.expand(ls)),p=packed(piv.expand(ps),at::kInt),r=packed(rhs.expand(bs));int64_t cols=rhs.size(-1),batch=out.numel()/(n*cols);
 auto flags=at::empty({std::min<int64_t>(256,(p.numel()+4095)/4096)},lu.options());run("lu_validate",{base(p,at::kInt)},{write_base(flags)},{u32(p.numel()),u32(n)},flags.numel(),4);TORCH_CHECK((flags.numel()==1?flags:flags.sum()).cpu().item<float>()==0,"LU pivot index out of range");
 run("lu_solve",{base(l),base(p,at::kInt),base(r)},{write_base(out)},
  {u32(batch),u32(n),u32(cols),uint32_t(left?adjoint:!adjoint)},(batch*cols+127)/128);
 return left?out:out.transpose(-2,-1);
}
std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> solve_ex(const at::Tensor& a,const at::Tensor& b,bool left,bool check_errors){
 TORCH_CHECK(a.dim()>=2&&a.size(-2)==a.size(-1),"solve requires square matrices");
 const bool vector=b.dim()==1||(b.dim()==a.dim()-1&&b.sizes()==a.sizes().slice(0,a.dim()-1));
 auto [lu,piv,info]=lu_factor(a,true,false);auto rhs=vector?b.unsqueeze(-1):b;
 if(vector)TORCH_CHECK(left,"vector solve requires left=True");
 auto out=lu_solve(lu,piv,rhs,left,false);if(vector)out=out.squeeze(-1);
 if(check_errors)check_info(info,"torch.linalg.solve_ex",a.dim()==2);return {out,lu,piv,info};
}
at::Tensor solve_triangular_native(const at::Tensor& a,const at::Tensor& b,bool upper,bool left,bool unit){
 auto triangle=upper?at::triu(a):at::tril(a);if(unit)triangle.diagonal(0,-2,-1).fill_(1);return std::get<0>(solve_ex(triangle,b,left,false));
}
std::tuple<at::Tensor,at::Tensor> inverse_ex(const at::Tensor& x,bool check_errors){
 TORCH_CHECK(x.dim()>=2&&x.size(-2)==x.size(-1),"linalg.inv requires square matrices");
 auto [lu,piv,info]=lu_factor(x,true,false);auto eye=at::eye(x.size(-1),x.options()).expand(x.sizes());
 auto result=lu_solve(lu,piv,eye,true,false);if(check_errors)check_info(info,"torch.linalg.inv_ex",x.dim()==2);return {result,info};
}
std::tuple<at::Tensor,at::Tensor> cholesky_ex(const at::Tensor& x,bool upper,bool check_errors){
 TORCH_CHECK(x.dim()>=2&&x.size(-2)==x.size(-1)&&x.scalar_type()==at::kFloat&&can_pack_view(x),"Cholesky requires packable FP32 square matrices");
 auto shape=x.sizes().vec();shape.resize(x.dim()-2);auto info=at::zeros(shape,x.options().dtype(at::kInt)),out=at::empty(x.sizes(),x.options());
 if(x.numel())run("cholesky_factor",{base(packed(x))},{write_base(out),write_base(info,at::kInt)},{u32(info.numel()),u32(x.size(-1)),uint32_t(upper)},(info.numel()+127)/128);
 if(check_errors)check_info(info,"torch.linalg.cholesky_ex",x.dim()==2);return {out,info};
}
at::Tensor cholesky_solve_native(const at::Tensor& b,const at::Tensor& factor,bool upper){
 TORCH_CHECK(factor.dim()>=2&&factor.size(-2)==factor.size(-1)&&b.dim()>=2&&b.size(-2)==factor.size(-1),"Cholesky solve shape mismatch");
 TORCH_CHECK(factor.scalar_type()==at::kFloat&&b.scalar_type()==at::kFloat&&factor.device()==b.device(),"Cholesky solve requires FP32 inputs on the same device");
 auto shape=at::infer_size_dimvector(factor.sizes().slice(0,factor.dim()-2),b.sizes().slice(0,b.dim()-2));auto fs=shape,bs=shape;auto n=factor.size(-1),cols=b.size(-1);fs.insert(fs.end(),{n,n});bs.insert(bs.end(),{n,cols});auto out=at::empty(bs,b.options());if(!out.numel())return out;
 auto f=packed(factor.expand(fs)),rhs=packed(b.expand(bs));auto batch=out.numel()/(n*cols);
 run("cholesky_solve",{base(f),base(rhs)},{write_base(out)},{u32(batch),u32(n),u32(cols),uint32_t(upper)},(batch*cols+127)/128);return out;
}
at::Tensor cholesky_inverse_native(const at::Tensor& factor,bool upper){TORCH_CHECK(factor.dim()>=2,"Cholesky inverse requires rank >= 2");return cholesky_solve_native(at::eye(factor.size(-1),factor.options()).expand(factor.sizes()),factor,upper);}

std::tuple<at::Tensor,at::Tensor,at::Tensor> lu_unpack_native(const at::Tensor& lu,const at::Tensor& piv,bool data,bool permutation){
 TORCH_CHECK(lu.dim()>=2&&lu.scalar_type()==at::kFloat,"lu_unpack requires FP32 matrices");auto m=lu.size(-2),n=lu.size(-1),k=std::min(m,n);auto batch=lu.sizes().vec();batch.resize(lu.dim()-2);auto ps=batch;ps.push_back(k);
 TORCH_CHECK(piv.scalar_type()==at::kInt&&piv.device()==lu.device()&&piv.sizes()==at::IntArrayRef(ps),"lu_unpack pivots shape/device/dtype mismatch");
 auto p=at::empty({0},lu.options()),l=at::empty({0},lu.options()),u=at::empty({0},lu.options());
 if(data){l=at::tril(lu.narrow(-1,0,k),-1);l.diagonal(0,-2,-1).fill_(1);u=at::triu(lu.narrow(-2,0,k));}
 if(permutation){auto shape=batch;shape.insert(shape.end(),{m,m});p=at::empty(shape,lu.options());if(p.numel()){
 auto indices=packed(piv,at::kInt);if(piv.numel()){auto flags=at::empty({std::min<int64_t>(256,(piv.numel()+4095)/4096)},lu.options());run("lu_validate",{base(indices,at::kInt)},{write_base(flags)},{u32(piv.numel()),u32(m)},flags.numel(),4);TORCH_CHECK((flags.numel()==1?flags:flags.sum()).cpu().item<float>()==0,"LU pivot index out of range");}
 auto ib=piv.numel()?base(indices,at::kInt):VulkanAllocator::instance().constant_float_buffer(1,0);run("lu_permutation",{ib},{write_base(p)},{u32(p.numel()),u32(m),u32(k)},(p.numel()+127)/128);
 }}return {p,l,u};
}
std::tuple<at::Tensor,at::Tensor,at::Tensor> lu_native(const at::Tensor& x,bool pivot){auto [lu,piv,info]=lu_factor(x,pivot,false);return lu_unpack_native(lu,piv,true,true);}

at::Tensor lu_permutation_sign(const at::Tensor& piv){auto reference=at::arange(1,piv.size(-1)+1,piv.options());auto swaps=at::sum(at::ne(piv,reference),{-1},false,at::kLong);return at::rsub(at::bitwise_and(swaps,1).to(at::kFloat).mul(2),1);}
std::tuple<at::Tensor,at::Tensor,at::Tensor> det_native(const at::Tensor& x){TORCH_CHECK(x.dim()>=2&&x.size(-2)==x.size(-1),"linalg.det requires square matrices");auto [lu,piv,info]=lu_factor(x.is_contiguous()?x.transpose(-2,-1):x,true,false);auto result=at::prod(lu.diagonal(0,-2,-1),-1).mul(lu_permutation_sign(piv));return {result,lu,piv};}
std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> slogdet_native(const at::Tensor& x){TORCH_CHECK(x.dim()>=2&&x.size(-2)==x.size(-1),"linalg.slogdet requires square matrices");auto [lu,piv,info]=lu_factor(x.is_contiguous()?x.transpose(-2,-1):x,true,false);auto diag=lu.diagonal(0,-2,-1);auto sign=at::prod(at::sign(diag),-1).mul(lu_permutation_sign(piv));auto logabs=at::sum(at::log(at::abs(diag)),{-1});return {sign,logabs,lu,piv};}
std::tuple<at::Tensor,at::Tensor> eigh_native(const at::Tensor& x,c10::string_view uplo,bool vectors){TORCH_CHECK(x.dim()>=2&&x.size(-1)==x.size(-2)&&x.scalar_type()==at::kFloat&&can_pack_view(x),"linalg.eigh requires packable FP32 square matrices");TORCH_CHECK(uplo=="L"||uplo=="U"||uplo=="l"||uplo=="u","UPLO argument must be 'L' or 'U'");auto n=x.size(-1);auto shape=x.sizes().vec();shape.pop_back();auto values=at::empty(shape,x.options()),v=at::empty_like(x);if(x.numel()){auto work=at::empty_like(x),status=at::zeros({},x.options().dtype(at::kInt));auto batches=x.numel()/(n*n);run("eigh_jacobi",{base(packed(x))},{write_base(work),write_base(v),write_base(values),write_base(status,at::kInt)},{u32(batches),u32(n),uint32_t(uplo=="U"||uplo=="u")},(batches+127)/128);auto code=status.cpu().item<int>();TORCH_CHECK(code==0,"linalg.eigh failed: non-finite input or Jacobi iteration did not converge (",code,")");}return {values,vectors?v:at::empty({0},x.options())};}

std::tuple<at::Tensor,at::Tensor> geqrf_general(const at::Tensor& x){TORCH_CHECK(x.dim()>=2&&x.scalar_type()==at::kFloat&&can_pack_view(x),"geqrf requires packable FP32 matrices");auto m=x.size(-2),n=x.size(-1),k=std::min(m,n);auto shape=x.sizes().vec();shape.resize(x.dim()-2);int64_t batches=1;for(auto d:shape)batches*=d;shape.push_back(k);auto result=at::empty_like(x),tau=at::empty(shape,x.options());if(x.numel())run("qr_factor",{base(packed(x))},{write_base(result),write_base(tau)},{u32(batches),u32(m),u32(n),u32(k)},(batches+127)/128);return {result,tau};}
at::Tensor ormqr_apply(const at::Tensor& a,const at::Tensor& tau,const at::Tensor& b,bool left,bool transpose){TORCH_CHECK(a.dim()>=2&&b.dim()==a.dim()&&tau.dim()==a.dim()-1,"ormqr input ranks mismatch");TORCH_CHECK(a.scalar_type()==at::kFloat&&b.scalar_type()==at::kFloat&&tau.scalar_type()==at::kFloat&&a.device()==b.device()&&a.device()==tau.device(),"ormqr requires FP32 inputs on the same device");auto dim=left?b.size(-2):b.size(-1),k=tau.size(-1);TORCH_CHECK(a.size(-2)==dim&&a.size(-1)>=k&&dim>=k,"ormqr matrix dimensions mismatch");TORCH_CHECK(a.sizes().slice(0,a.dim()-2)==b.sizes().slice(0,b.dim()-2)&&a.sizes().slice(0,a.dim()-2)==tau.sizes().slice(0,tau.dim()-1),"ormqr batch dimensions mismatch");if(!b.numel()||!k)return b.clone();auto result=at::empty_like(b);auto rows=b.size(-2),cols=b.size(-1),batch=b.numel()/(rows*cols),vectors=left?cols:rows;run("qr_apply",{base(packed(a)),base(packed(tau)),base(packed(b))},{write_base(result)},{u32(batch),u32(dim),u32(a.size(-1)),u32(k),u32(rows),u32(cols),uint32_t(left),uint32_t(transpose)},(batch*vectors+127)/128);return result;}
at::Tensor ormqr_general(const at::Tensor& a,const at::Tensor& tau,const at::Tensor& b,bool left,bool transpose){TORCH_CHECK(a.dim()>=2,"input must have at least 2 dimensions");TORCH_CHECK(b.dim()>=2&&tau.dim()>=1,"ormqr other/tau rank mismatch");TORCH_CHECK(tau.size(-1)==std::min(left?b.size(-2):b.size(-1),a.size(-1)),"tau.shape[-1] must be equal to min(other.shape[-2], input.shape[-1])");return ormqr_apply(a,tau,b,left,transpose);}
at::Tensor householder_general(const at::Tensor& a,const at::Tensor& tau){TORCH_CHECK(a.dim()>=2&&a.size(-2)>=a.size(-1)&&tau.dim()==a.dim()-1&&a.size(-1)>=tau.size(-1),"linalg.householder_product requires m >= n >= k");auto eye=at::eye(a.size(-2),a.size(-1),a.options()).expand(a.sizes());return ormqr_apply(a,tau,eye,true,false);}
std::tuple<at::Tensor,at::Tensor> qr_general(const at::Tensor& x,c10::string_view mode){TORCH_CHECK(mode=="reduced"||mode=="complete"||mode=="r","qr mode must be 'reduced', 'complete' or 'r'");auto [factor,tau]=geqrf_general(x);auto m=x.size(-2),n=x.size(-1),k=std::min(m,n),width=mode=="complete"?m:k;auto r=at::triu(factor.narrow(-2,0,width));auto q=at::empty({0},x.options());if(mode!="r"){auto shape=x.sizes().vec();shape.back()=width;auto b=at::eye(m,width,x.options()).expand(shape);q=ormqr_general(factor,tau,b,true,false);}return {q,r};}

std::tuple<at::Tensor,at::Tensor> polar_general(const at::Tensor& x){TORCH_CHECK(x.dim()>=2,"linalg.polar input must have at least 2 dimensions");TORCH_CHECK(x.size(-2)>=x.size(-1),"linalg.polar: input must have at least as many rows as columns");auto [u,s,vh]=at::linalg_svd(x,false);return {at::matmul(u,vh),at::matmul(vh.transpose(-2,-1).mul(s.unsqueeze(-2)),vh)};}
}
TORCH_LIBRARY_IMPL(aten,PrivateUse1,m){
 m.impl("linalg_polar",&polar_general);
 m.impl("geqrf",&geqrf_general);m.impl("ormqr",&ormqr_general);m.impl("linalg_householder_product",&householder_general);m.impl("linalg_qr",&qr_general);
 m.impl("_linalg_det",&det_native);m.impl("_linalg_slogdet",&slogdet_native);m.impl("_linalg_eigh",&eigh_native);
 m.impl("lu_unpack",&lu_unpack_native);m.impl("linalg_lu",&lu_native);
 m.impl("linalg_solve_triangular",&solve_triangular_native);
 m.impl("linalg_inv_ex",&inverse_ex);m.impl("linalg_cholesky_ex",&cholesky_ex);m.impl("cholesky_solve",&cholesky_solve_native);m.impl("cholesky_inverse",&cholesky_inverse_native);
 m.impl("eye.out",&eye_square);m.impl("eye.m_out",&eye_out);
 m.impl("linalg_lu_factor_ex",&lu_factor);m.impl("linalg_lu_solve",&lu_solve);m.impl("_linalg_solve_ex",&solve_ex);m.impl("_linalg_check_errors",&check_info);
}
}
#include "native_utils.h"
#include <ATen/WrapDimUtils.h>
#include <ATen/ExpandUtils.h>
#include <torch/library.h>
namespace torch_vulkan::ops {namespace {
using namespace native;
using OptionalTolerance=std::optional<at::Tensor>;
void spectral_input(const at::Tensor& x,bool hermitian){TORCH_CHECK(x.dim()>=2&&x.scalar_type()==at::kFloat,"linalg spectral operation expects float32 matrices");TORCH_CHECK(!hermitian||x.size(-2)==x.size(-1),"linalg spectral operation expects square Hermitian matrices");}
at::Tensor tolerance_tensor(const OptionalTolerance& value,const at::Tensor& x,double fallback){if(!value||!value->defined())return at::full({},fallback,x.options());TORCH_CHECK(!value->is_complex(),"tolerance must be real");return value->to(x.device(),at::kFloat);}
at::Tensor spectral_threshold(const at::Tensor& x,const at::Tensor& values,const OptionalTolerance& atol,const OptionalTolerance& rtol){auto a=tolerance_tensor(atol,x,0),r=tolerance_tensor(rtol,x,atol&&atol->defined()?0:std::max(x.size(-1),x.size(-2))*std::numeric_limits<float>::epsilon());auto largest=at::amax(at::abs(values),{-1},false);return at::maximum(a,r.mul(largest)).unsqueeze(-1);}
at::Tensor pinv_tensor_tol(const at::Tensor& x,const OptionalTolerance& atol,const OptionalTolerance& rtol,bool hermitian){spectral_input(x,hermitian);if(!x.size(-1)||!x.size(-2))return x.transpose(-2,-1).clone();at::Tensor vectors,values,right;
 if(hermitian){auto pair=at::linalg_eigh(x,"L");values=std::get<0>(pair);vectors=std::get<1>(pair);right=vectors.transpose(-2,-1);}else{auto triplet=at::linalg_svd(x,false,std::nullopt);vectors=std::get<0>(triplet);values=std::get<1>(triplet);right=std::get<2>(triplet);}
 auto threshold=spectral_threshold(x,values,atol,rtol);auto inverse=at::where(at::abs(values).gt(threshold),at::reciprocal(values),at::zeros({},x.options()));return right.transpose(-2,-1).mul(inverse.unsqueeze(-2)).matmul(vectors.transpose(-2,-1));
}
at::Tensor pinv_float_tol(const at::Tensor& x,std::optional<double> atol,std::optional<double> rtol,bool hermitian){OptionalTolerance a=atol?OptionalTolerance(at::full({},*atol,x.options())):std::nullopt,r=rtol?OptionalTolerance(at::full({},*rtol,x.options())):std::nullopt;return at::linalg_pinv(x,a,r,hermitian);}
at::Tensor pinv_rcond(const at::Tensor& x,double rcond,bool hermitian){return pinv_float_tol(x,0,rcond,hermitian);}
at::Tensor pinv_rcond_tensor(const at::Tensor& x,const at::Tensor& rcond,bool hermitian){return at::linalg_pinv(x,at::zeros({},x.options()),rcond,hermitian);}
at::Tensor matrix_rank_tensor_tol(const at::Tensor& x,const OptionalTolerance& atol,const OptionalTolerance& rtol,bool hermitian){spectral_input(x,hermitian);if(!x.size(-1)||!x.size(-2)){auto shape=x.sizes().slice(0,x.dim()-2);return at::zeros(shape,x.options().dtype(at::kLong));}auto values=hermitian?at::linalg_eigvalsh(x,"L"):at::linalg_svdvals(x,std::nullopt);auto threshold=spectral_threshold(x,values,atol,rtol);return at::sum(at::abs(values).gt(threshold),{-1},false,at::kLong);}
at::Tensor matrix_rank_float_tol(const at::Tensor& x,std::optional<double> atol,std::optional<double> rtol,bool hermitian){OptionalTolerance a=atol?OptionalTolerance(at::full({},*atol,x.options())):std::nullopt,r=rtol?OptionalTolerance(at::full({},*rtol,x.options())):std::nullopt;return matrix_rank_tensor_tol(x,a,r,hermitian);}
at::Tensor matrix_rank_legacy(const at::Tensor& x,double tol,bool hermitian){return matrix_rank_float_tol(x,tol,0,hermitian);}
at::Tensor matrix_rank_legacy_tensor(const at::Tensor& x,const at::Tensor& tol,bool hermitian){return matrix_rank_tensor_tol(x,tol,at::zeros({},x.options()),hermitian);}
}
TORCH_LIBRARY_IMPL(aten,PrivateUse1,m){
 m.impl("linalg_pinv",&pinv_rcond);m.impl("linalg_pinv.atol_rtol_tensor",&pinv_tensor_tol);m.impl("linalg_pinv.atol_rtol_float",&pinv_float_tol);m.impl("linalg_pinv.rcond_tensor",&pinv_rcond_tensor);
 m.impl("linalg_matrix_rank",&matrix_rank_legacy);m.impl("linalg_matrix_rank.tol_tensor",&matrix_rank_legacy_tensor);m.impl("linalg_matrix_rank.atol_rtol_tensor",&matrix_rank_tensor_tol);m.impl("linalg_matrix_rank.atol_rtol_float",&matrix_rank_float_tol);
}
}

namespace torch_vulkan::ops {
// Preserve the composite autograd graph while creating FP32 tolerance tensors.
TORCH_LIBRARY_IMPL(aten,AutogradPrivateUse1,m){
 m.impl("linalg_pinv",&pinv_rcond);m.impl("linalg_pinv.atol_rtol_float",&pinv_float_tol);m.impl("linalg_pinv.rcond_tensor",&pinv_rcond_tensor);
}
}
#include <cctype>
namespace torch_vulkan::ops {namespace {
std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> lstsq_native(const at::Tensor& a,const at::Tensor& b,std::optional<double> rcond,std::optional<c10::string_view> driver){
 TORCH_CHECK(a.dim()>=2,"torch.linalg.lstsq: input must have at least 2 dimensions.");TORCH_CHECK(b.dim()>=1,"torch.linalg.lstsq: other must have at least 1 dimension.");TORCH_CHECK(a.scalar_type()==b.scalar_type(),"torch.linalg.lstsq: Expected input and other to have the same dtype, but got input's dtype ",a.scalar_type()," and other's dtype ",b.scalar_type());
 auto diff=a.dim()-b.dim();TORCH_CHECK(diff>=0&&diff<=1,"torch.linalg.lstsq: input.dim() must be greater or equal to other.dim() and (input.dim() - other.dim()) <= 1");
 bool vector=b.dim()==1||(b.dim()==a.dim()-1&&b.sizes()==a.sizes().slice(0,a.dim()-1));auto rhs=vector?b.unsqueeze(-1):b;TORCH_CHECK(a.size(-2)==rhs.size(-2),vector?"torch.linalg.lstsq: input.size(-2) should match other.size(-1)":"torch.linalg.lstsq: input.size(-2) should match other.size(-2)");TORCH_CHECK(a.device()==b.device(),"torch.linalg.lstsq: Expected all tensors to be on the same device");TORCH_CHECK(a.scalar_type()==at::kFloat,"Vulkan least squares requires FP32 matrices");
 std::string selected=driver?std::string(*driver):"gelsy";for(auto& ch:selected)ch=std::tolower(static_cast<unsigned char>(ch));TORCH_CHECK(selected=="gels"||selected=="gelsy"||selected=="gelsd"||selected=="gelss","torch.linalg.lstsq: parameter `driver` should be one of (gels, gelsy, gelsd, gelss)");
 auto [u,s,vh]=at::linalg_svd(a,false,std::nullopt);auto k=std::min(a.size(-2),a.size(-1));double tolerance=rcond.value_or(std::max(a.size(-2),a.size(-1))*std::numeric_limits<float>::epsilon());if(tolerance<0)tolerance=std::numeric_limits<float>::epsilon();
 auto rank=at::zeros(a.sizes().slice(0,a.dim()-2),a.options().dtype(at::kLong));at::Tensor inverse;
 if(k){auto cutoff=at::amax(s,{-1},true).mul(selected=="gels"?0:tolerance);auto valid=s.gt(cutoff);rank=at::sum(valid,{-1},false,at::kLong);inverse=at::where(valid,at::reciprocal(s),at::zeros({},s.options()));}else inverse=s;
 auto solution=vh.transpose(-2,-1).matmul(inverse.unsqueeze(-1).mul(u.transpose(-2,-1).matmul(rhs)));auto residuals=at::empty({0},a.options());
 if(a.size(-2)>a.size(-1)&&selected!="gelsy"&&(selected=="gels"||at::all(rank.eq(a.size(-1))).cpu().item<bool>())){auto residual=a.matmul(solution).sub(rhs);residuals=at::sum(residual.square(),{-2},false);}
 if(vector)solution=solution.squeeze(-1);
 if(selected=="gels")rank=at::empty({0},a.options().dtype(at::kLong));if(selected=="gels"||selected=="gelsy")s=at::empty({0},a.options());return {solution,residuals,rank,s};
}
}
TORCH_LIBRARY_IMPL(aten,PrivateUse1,m){m.impl("linalg_lstsq",&lstsq_native);}
}
#include <ATen/native/LinearAlgebraUtils.h>
namespace torch_vulkan::ops {namespace {
at::Tensor matrix_sqrth_native(const at::Tensor& x){at::native::squareCheckInputs(x,"linalg.matrix_sqrth");at::native::checkFloatingOrComplex(x,"linalg.matrix_sqrth",false);TORCH_CHECK(x.scalar_type()==at::kFloat,"Vulkan matrix_sqrth requires FP32 input");if(!x.size(-1))return x.clone();auto [values,vectors]=at::linalg_eigh(x,"L");auto result=vectors.mul(at::sqrt(at::clamp_min(values,0)).unsqueeze(-2)).matmul(vectors.transpose(-2,-1));return result.add(result.transpose(-2,-1)).mul(.5);}
at::Tensor matrix_exp_native(const at::Tensor& x){at::native::squareCheckInputs(x,"linalg.matrix_exp");at::native::checkFloatingOrComplex(x,"linalg.matrix_exp",false);TORCH_CHECK(x.scalar_type()==at::kFloat,"Vulkan matrix_exp requires FP32 input");if(!x.numel())return x.clone();if(x.size(-1)==1)return at::exp(x);auto norm=at::amax(at::sum(at::abs(x),{-2},false)).cpu().item<float>();int scale=std::isfinite(norm)&&norm>.5f?std::max(0,int(std::ceil(std::log2(double(norm)/.5)))):0;auto scaled=x.mul(std::ldexp(1.,-scale));auto result=at::eye(x.size(-1),x.options()).expand(x.sizes()).clone(),term=result.clone();for(int k=1;k<=32;k++){term=term.matmul(scaled).mul(1./k);result=result.add(term);}for(int k=0;k<scale;k++)result=result.matmul(result);return result;}
std::tuple<at::Tensor,at::Tensor> triangular_solve_legacy(const at::Tensor& b,const at::Tensor& a,bool upper,bool transpose,bool unit){auto matrix=transpose?a.transpose(-2,-1):a;auto solution=at::linalg_solve_triangular(matrix,b,transpose?!upper:upper,true,unit);auto batch=at::infer_size_dimvector(a.sizes().slice(0,a.dim()-2),b.sizes().slice(0,b.dim()-2));batch.push_back(a.size(-2));batch.push_back(a.size(-1));return {solution,a.expand(batch).clone()};}
}
TORCH_LIBRARY_IMPL(aten,PrivateUse1,m){m.impl("linalg_matrix_sqrth",&matrix_sqrth_native);m.impl("linalg_matrix_exp",&matrix_exp_native);m.impl("triangular_solve",&triangular_solve_legacy);}
}
