#include "native_utils.h"
#include <ATen/native/Resize.h>
#include <ATen/MemoryOverlap.h>
#include <torch/library.h>
namespace torch_vulkan::ops {
namespace {
using namespace native;
std::tuple<at::Tensor,at::Tensor,at::Tensor> svd_native(const at::Tensor& x,bool full,bool uv,std::optional<c10::string_view> driver){
 TORCH_CHECK(!driver,"torch.linalg.svd: keyword argument `driver=` is only supported on CUDA inputs with cuSOLVER backend.");
 TORCH_CHECK(x.dim()>=2&&x.scalar_type()==at::kFloat&&can_pack_view(x),"Vulkan SVD requires packable FP32 matrices");
 int64_t m=x.size(-2),n=x.size(-1),k=std::min(m,n),l=std::max(m,n),qcols=full?l:k,batches=1;auto batch=x.sizes().slice(0,x.dim()-2).vec();for(auto v:batch)batches*=v;
 auto us=uv?batch:std::vector<int64_t>{},ss=batch,vs=us;ss.push_back(k);if(uv){us.insert(us.end(),{m,full?m:k});vs.insert(vs.end(),{full?n:k,n});}else{us={0};vs={0};}
 auto u=at::empty(us,x.options()),s=at::empty(ss,x.options()),vh=at::empty(vs,x.options());if(!batches)return {u,s,vh};
 if(!k){if(uv){if(u.numel())u.copy_(at::eye(m,x.options()).expand(us));if(vh.numel())vh.copy_(at::eye(n,x.options()).expand(vs));}return {u,s,vh};}
 auto z=at::empty({batches,l,qcols},x.options()),v=at::empty({batches,k,k},x.options()),status=at::zeros({1},x.options().dtype(at::kInt));
 auto placeholder=at::empty({1},x.options());auto dummy=write_base(placeholder);
 run("svd_jacobi",{base(packed(x))},{write_base(z),write_base(v),uv?write_base(u):dummy,write_base(s),uv?write_base(vh):dummy,write_base(status,at::kInt)},
  {u32(m),u32(n),u32(k),u32(l),u32(qcols),uint32_t(full),uint32_t(uv)},batches);
 auto error=status.cpu().item<int>();TORCH_CHECK(error==0,"linalg.svd: ",error==1?"The algorithm failed to converge because the input matrix contained non-finite values.":"Jacobi iteration failed to converge.");
 return {u,s,vh};
}
std::tuple<at::Tensor&,at::Tensor&,at::Tensor&> svd_out(const at::Tensor& x,bool full,bool uv,std::optional<c10::string_view> driver,at::Tensor& u,at::Tensor& s,at::Tensor& vh){
 for(auto* t:{&u,&s,&vh}){TORCH_CHECK(t->device()==x.device()&&t->scalar_type()==at::kFloat,"SVD output device/dtype mismatch");at::assert_no_internal_overlap(*t);at::assert_no_partial_overlap(*t,x);}
 auto [uu,ss,vv]=svd_native(x,full,uv,driver);at::native::resize_output(u,uu.sizes());at::native::resize_output(s,ss.sizes());at::native::resize_output(vh,vv.sizes());u.copy_(uu);s.copy_(ss);vh.copy_(vv);return {u,s,vh};
}
}
TORCH_LIBRARY_IMPL(aten,PrivateUse1,m){m.impl("_linalg_svd",&svd_native);m.impl("_linalg_svd.U",&svd_out);}
}
