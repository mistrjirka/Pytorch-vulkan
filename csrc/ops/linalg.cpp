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

}
TORCH_LIBRARY_IMPL(aten,PrivateUse1,m){
 m.impl("lu_unpack",&lu_unpack_native);m.impl("linalg_lu",&lu_native);
 m.impl("linalg_solve_triangular",&solve_triangular_native);
 m.impl("linalg_inv_ex",&inverse_ex);m.impl("linalg_cholesky_ex",&cholesky_ex);m.impl("cholesky_solve",&cholesky_solve_native);m.impl("cholesky_inverse",&cholesky_inverse_native);
 m.impl("eye.out",&eye_square);m.impl("eye.m_out",&eye_out);
 m.impl("linalg_lu_factor_ex",&lu_factor);m.impl("linalg_lu_solve",&lu_solve);m.impl("_linalg_solve_ex",&solve_ex);m.impl("_linalg_check_errors",&check_info);
}
}
