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
 run("lu_factor",{base(packed(a))},{base(lu),base(piv,at::kInt),base(info,at::kInt)},
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
 auto flags=at::empty({std::min<int64_t>(256,(p.numel()+4095)/4096)},lu.options());run("lu_validate",{base(p,at::kInt)},{base(flags)},{u32(p.numel()),u32(n)},flags.numel(),4);TORCH_CHECK((flags.numel()==1?flags:flags.sum()).cpu().item<float>()==0,"LU pivot index out of range");
 run("lu_solve",{base(l),base(p,at::kInt),base(r)},{base(out)},
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
}
TORCH_LIBRARY_IMPL(aten,PrivateUse1,m){
 m.impl("eye.out",&eye_square);m.impl("eye.m_out",&eye_out);
 m.impl("linalg_lu_factor_ex",&lu_factor);m.impl("linalg_lu_solve",&lu_solve);m.impl("_linalg_solve_ex",&solve_ex);m.impl("_linalg_check_errors",&check_info);
}
}
