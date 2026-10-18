#include "native_utils.h"
#include <torch/library.h>
namespace torch_vulkan::ops {
namespace {
using namespace native;
std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> bag_forward(const at::Tensor& weight,const at::Tensor& indices,const at::Tensor& offsets,bool frequency,int64_t mode,bool sparse,const std::optional<at::Tensor>& per_sample,bool last,int64_t padding){
 TORCH_CHECK(!sparse,"Sparse embedding-bag gradients are not implemented");TORCH_CHECK(weight.dim()==2&&weight.scalar_type()==at::kFloat&&indices.dim()==1&&offsets.dim()==1&&indices.device()==weight.device()&&offsets.device()==weight.device(),"embedding_bag input mismatch");
 TORCH_CHECK((indices.scalar_type()==at::kInt||indices.scalar_type()==at::kLong)&&(offsets.scalar_type()==at::kInt||offsets.scalar_type()==at::kLong),"embedding_bag requires integer indices/offsets");
 TORCH_CHECK(mode>=0&&mode<=2&&!(mode==2&&frequency),"embedding_bag invalid mode/frequency");TORCH_CHECK(padding>=-1&&padding<weight.size(0),"embedding_bag padding out of range");
 int64_t n=indices.numel(),bags=offsets.numel()-int(last),features=weight.size(1);TORCH_CHECK(bags>=0&&(bags>0||n==0),"embedding_bag invalid offsets");
 const bool weighted=per_sample&&per_sample->defined();if(weighted)TORCH_CHECK(mode==0&&per_sample->sizes()==indices.sizes()&&per_sample->scalar_type()==at::kFloat&&per_sample->device()==weight.device(),"embedding_bag per-sample weights mismatch");
 auto out=at::empty({bags,features},weight.options()),mapping=at::empty({n},indices.options().dtype(at::kLong)),sizes=at::empty({bags},mapping.options()),maxima=at::empty(mode==2?std::vector<int64_t>{bags,features}:std::vector<int64_t>{bags},mapping.options());
 auto dummy=VulkanAllocator::instance().constant_float_buffer(1,0);auto idx=n?packed(indices,indices.scalar_type()):at::Tensor(),off=offsets.numel()?packed(offsets,offsets.scalar_type()):at::Tensor();auto ib=idx.defined()?base(idx,idx.scalar_type()):dummy,ob=off.defined()?base(off,off.scalar_type()):dummy;
 uint32_t iw=indices.element_size()/4,ow=offsets.element_size()/4;
 if(n||offsets.numel()){auto flags=at::empty({std::min<int64_t>(256,(std::max(n,offsets.numel())+4095)/4096)},weight.options());run("embedding_bag_validate",{ib,ob},{base(flags)},{u32(n),u32(offsets.numel()),u32(weight.size(0)),iw,ow,uint32_t(last)},flags.numel(),4);TORCH_CHECK((flags.numel()==1?flags:flags.sum()).cpu().item<float>()==0,"embedding_bag invalid indices or offsets");}
 if(n||bags)run("embedding_bag_meta",{ib,ob},{n?base(mapping,at::kLong):dummy,bags?base(sizes,at::kLong):dummy},{u32(n),u32(bags),iw,ow,uint32_t(last),uint32_t(padding)},(std::max(n,bags)+127)/128);
 if(out.numel()){
  auto w=weight.numel()?packed(weight):at::Tensor(),ps=weighted&&n?packed(*per_sample):at::Tensor();run("embedding_bag",{w.defined()?base(w):dummy,ib,ob,ps.defined()?base(ps):dummy},{base(out),base(maxima,at::kLong)},
   {u32(n),u32(bags),u32(features),iw,ow,uint32_t(last),uint32_t(padding),uint32_t(mode),uint32_t(weighted)},(out.numel()+127)/128);
 }return {out,mapping,sizes,maxima};
}
at::Tensor bag_backward(const at::Tensor& grad,const at::Tensor& indices,const at::Tensor& mapping,const at::Tensor& sizes,const at::Tensor& maxima,c10::SymInt rows,bool frequency,int64_t mode,const std::optional<at::Tensor>& per_sample,int64_t padding){
 TORCH_CHECK(grad.dim()==2&&grad.scalar_type()==at::kFloat&&indices.device()==grad.device(),"embedding_bag backward mismatch");auto out=at::empty({rows.expect_int(),grad.size(1)},grad.options());if(!out.numel())return out;if(!indices.numel()){out.zero_();return out;}
 auto g=packed(grad),i=packed(indices,indices.scalar_type()),mp=packed(mapping,at::kLong),bs=packed(sizes,at::kLong),mx=packed(maxima,at::kLong),ps=per_sample&&per_sample->defined()?packed(*per_sample):at::Tensor();auto dummy=VulkanAllocator::instance().constant_float_buffer(1,0);
 run("embedding_bag_backward",{base(g),base(i,i.scalar_type()),base(mp,at::kLong),base(bs,at::kLong),base(mx,at::kLong),ps.defined()?base(ps):dummy,dummy},{base(out)},
 {u32(indices.numel()),u32(grad.size(0)),u32(grad.size(1)),u32(rows.expect_int()),uint32_t(indices.element_size()/4),uint32_t(padding),uint32_t(mode),uint32_t(ps.defined()),uint32_t(frequency),0},(out.numel()+127)/128);return out;
}
at::Tensor bag_backward_entry(const at::Tensor& g,const at::Tensor& i,const at::Tensor&,const at::Tensor& mp,const at::Tensor& bs,const at::Tensor& mx,c10::SymInt rows,bool freq,int64_t mode,bool sparse,const std::optional<at::Tensor>& ps,int64_t pad){TORCH_CHECK(!sparse,"Sparse embedding-bag gradients are not implemented");return bag_backward(g,i,mp,bs,mx,rows,freq,mode,ps,pad);}
at::Tensor bag_ps_backward(const at::Tensor& grad,const at::Tensor& weight,const at::Tensor& indices,const at::Tensor&,const at::Tensor& mapping,int64_t mode,int64_t padding){
 TORCH_CHECK(mode==0,"per-sample gradient requires sum mode");auto out=at::empty(indices.sizes(),weight.options());if(!out.numel())return out;auto g=packed(grad),w=packed(weight),i=packed(indices,indices.scalar_type()),mp=packed(mapping,at::kLong);auto dummy=VulkanAllocator::instance().constant_float_buffer(1,0);
 run("embedding_bag_backward",{base(g),base(i,i.scalar_type()),base(mp,at::kLong),dummy,dummy,dummy,base(w)},{base(out)},
 {u32(indices.numel()),u32(grad.size(0)),u32(grad.size(1)),u32(weight.size(0)),uint32_t(indices.element_size()/4),uint32_t(padding),0,0,0,1},(out.numel()+127)/128);return out;
}
}
TORCH_LIBRARY_IMPL(aten,PrivateUse1,m){m.impl("_embedding_bag",&bag_forward);m.impl("_embedding_bag_backward",&bag_backward_entry);m.impl("_embedding_bag_forward_only",&bag_forward);m.impl("_embedding_bag_dense_backward",&bag_backward);m.impl("_embedding_bag_per_sample_weights_backward",&bag_ps_backward);}
}
