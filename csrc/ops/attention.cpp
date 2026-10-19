#include "native_utils.h"
#include "../vulkan_generator.h"
#include <torch/library.h>
#include <cmath>
namespace torch_vulkan::ops {
namespace {
using namespace native;
std::vector<uint32_t> params(const at::Tensor& q,const at::Tensor& k,const at::Tensor& v,const at::Tensor& mask,double drop,bool causal,double scale,uint32_t mode,at::IntArrayRef state){
 TORCH_CHECK(q.dim()==3&&k.dim()==3&&v.dim()==3&&q.size(0)==k.size(0)&&k.size(0)==v.size(0)&&q.size(2)==k.size(2)&&k.size(1)==v.size(1),"attention dimensions mismatch");
 TORCH_CHECK(q.device()==k.device()&&k.device()==v.device()&&mask.device()==q.device(),"attention devices mismatch");TORCH_CHECK(drop>=0&&drop<=1,"invalid attention dropout");
 TORCH_CHECK(uint64_t(q.size(0))*q.size(1)<=UINT32_MAX/uint64_t(std::max<int64_t>(1,k.size(1))),"attention score indexing exceeds uint32");
 uint32_t mt=mask.numel()?(mask.scalar_type()==at::kBool?2:1):0;
 TORCH_CHECK(!mt||(mask.sizes()==at::IntArrayRef({q.size(0),q.size(1),k.size(1)})&&(mask.scalar_type()==at::kFloat||mask.scalar_type()==at::kBool)),"attention mask shape/dtype mismatch");
 return {u32(q.size(0)),u32(q.size(1)),u32(k.size(1)),u32(q.size(2)),u32(v.size(2)),mode,mt,uint32_t(causal),bits(float(scale)),bits(float(drop)),uint32_t(state[0]),uint32_t(state[1]),uint32_t(state[2]),uint32_t(state[3])};
}
std::tuple<at::Tensor,at::Tensor,at::Tensor> attention_forward(const at::Tensor& query,const at::Tensor& key,const at::Tensor& value,const at::Tensor& mask,double drop,bool causal,double scale){
 std::vector<int64_t> state(4,0);if(drop>0&&drop<1){auto s=random_state(std::nullopt,(uint64_t(query.size(0))*query.size(1)*key.size(1)+3)/4);state.assign(s.begin(),s.end());}
 auto rng=at::empty({4},at::TensorOptions().dtype(at::kLong).device(at::kCPU));std::copy(state.begin(),state.end(),rng.data_ptr<int64_t>());
 auto p=params(query,key,value,mask,drop,causal,scale,0,state);auto out=at::empty({query.size(0),query.size(1),value.size(2)},query.options()),lse=at::empty({query.size(0),query.size(1)},query.options());
 if(!lse.numel())return {out,lse,rng};if(!out.numel()){lse.zero_();return {out,lse,rng};}if(!key.size(1)){out.zero_();lse.fill_(-INFINITY);return {out,lse,rng};}
 auto q=packed(query),k=packed(key),v=packed(value),m=mask.numel()?packed(mask,mask.scalar_type()):at::Tensor();auto dummy=VulkanAllocator::instance().constant_float_buffer(1,0);
 run("sdpa_training",{base(q),base(k),base(v),m.defined()?base(m,m.scalar_type()):dummy,dummy,dummy,dummy},{write_base(out),write_base(lse)},p,lse.numel(),2048);return {out,lse,rng};
}
std::tuple<at::Tensor,at::Tensor,at::Tensor,at::Tensor> attention_backward(const at::Tensor& grad,const at::Tensor& query,const at::Tensor& key,const at::Tensor& value,const at::Tensor& mask,const at::Tensor& output,const at::Tensor& logsum,double drop,bool causal,double scale,const at::Tensor& rng){
 TORCH_CHECK(rng.device().is_cpu()&&rng.scalar_type()==at::kLong&&rng.numel()==4&&rng.is_contiguous(),"Invalid attention RNG metadata");at::IntArrayRef state(rng.const_data_ptr<int64_t>(),4);
 TORCH_CHECK(grad.sizes()==output.sizes()&&output.dim()==3&&output.size(0)==query.size(0)&&output.size(1)==query.size(1)&&output.size(2)==value.size(2)&&logsum.sizes()==query.sizes().slice(0,2),"attention backward shape mismatch");
 auto p=params(query,key,value,mask,drop,causal,scale,1,state);auto dq=at::empty(query.sizes(),query.options()),dk=at::empty(key.sizes(),key.options()),dv=at::empty(value.sizes(),value.options()),dm=at::empty({query.size(0),query.size(1),key.size(1)},query.options());
 if(!query.numel()||!key.numel()||!value.numel()){dq.zero_();dk.zero_();dv.zero_();dm.zero_();return {dq,dk,dv,dm};}
 auto q=packed(query),k=packed(key),v=packed(value),g=packed(grad),o=packed(output),l=packed(logsum),m=mask.numel()?packed(mask,mask.scalar_type()):at::Tensor();auto dummy=VulkanAllocator::instance().constant_float_buffer(1,0);
 std::vector<Buffer> inputs={base(q),base(k),base(v),m.defined()?base(m,m.scalar_type()):dummy,base(g),base(o),base(l)};
 run("sdpa_training",inputs,{write_base(dq),write_base(dm)},p,query.size(0)*query.size(1),2048);p[5]=2;run("sdpa_training",inputs,{write_base(dk),write_base(dv)},p,key.size(0)*key.size(1),2048);return {dq,dk,dv,dm};
}
}
TORCH_LIBRARY_FRAGMENT(torch_vulkan,m){m.def("attention_forward(Tensor q, Tensor k, Tensor v, Tensor mask, float dropout, bool causal, float scale) -> (Tensor, Tensor, Tensor)");m.def("attention_backward(Tensor grad, Tensor q, Tensor k, Tensor v, Tensor mask, Tensor output, Tensor lse, float dropout, bool causal, float scale, Tensor state) -> (Tensor, Tensor, Tensor, Tensor)");}
TORCH_LIBRARY_IMPL(torch_vulkan,PrivateUse1,m){m.impl("attention_forward",&attention_forward);m.impl("attention_backward",&attention_backward);}
}
namespace torch_vulkan::ops {namespace {
at::Tensor cpu_flash_mask(const at::Tensor& q,const at::Tensor& k,const std::optional<at::Tensor>& mask){if(!mask||!mask->defined())return at::empty({0},q.options());TORCH_CHECK(mask->scalar_type()==at::kFloat,"scaled_dot_product_attention_flash_attention: Attention mask is the same data type as query");TORCH_CHECK(mask->dim()==2||mask->dim()==4,"scaled_dot_product_attention_flash_attention: Attention mask dim in {2, 4}");TORCH_CHECK(mask->device()==q.device(),"attention mask device mismatch");return mask->expand({q.size(0),q.size(1),q.size(2),k.size(2)}).reshape({q.size(0)*q.size(1),q.size(2),k.size(2)});}
void cpu_flash_check(const at::Tensor& q,const at::Tensor& k,const at::Tensor& v,double drop){TORCH_CHECK(q.dim()==4&&k.dim()==4&&v.dim()==4,"scaled_dot_product_attention_flash_attention: Accept only 4 dims inputs shape of {B, H, T, K}");TORCH_CHECK(q.scalar_type()==at::kFloat&&k.scalar_type()==at::kFloat&&v.scalar_type()==at::kFloat,"Vulkan attention requires FP32 input");TORCH_CHECK(drop==0,"scaled_dot_product_attention_flash_attention: Currently do not support dropout > 0");TORCH_CHECK(q.size(3)==v.size(3)&&k.size(3)==v.size(3),"scaled_dot_product_attention_flash_attention: Q/K/V should have the same head size");TORCH_CHECK(q.sizes().slice(0,2)==k.sizes().slice(0,2)&&q.sizes().slice(0,2)==v.sizes().slice(0,2)&&k.size(2)==v.size(2),"attention batch/head/sequence mismatch");}
std::tuple<at::Tensor,at::Tensor> cpu_flash_forward_native(const at::Tensor& q,const at::Tensor& k,const at::Tensor& v,double drop,bool causal,const std::optional<at::Tensor>& mask,std::optional<double> scale){cpu_flash_check(q,k,v,drop);auto rows=q.size(0)*q.size(1);auto [output,lse,state]=attention_forward(q.reshape({rows,q.size(2),q.size(3)}),k.reshape({rows,k.size(2),k.size(3)}),v.reshape({rows,v.size(2),v.size(3)}),cpu_flash_mask(q,k,mask),0,causal,scale.value_or(1./std::sqrt(double(q.size(3)))));lse=at::where(at::isneginf(lse),at::zeros({},lse.options()),lse);return {output.reshape(q.sizes()),lse.reshape({q.size(0),q.size(1),q.size(2)})};}
std::tuple<at::Tensor,at::Tensor,at::Tensor> cpu_flash_backward_native(const at::Tensor& grad,const at::Tensor& q,const at::Tensor& k,const at::Tensor& v,const at::Tensor& out,const at::Tensor& lse,double drop,bool causal,const std::optional<at::Tensor>& mask,std::optional<double> scale){if(!grad.defined())return {at::Tensor(),at::Tensor(),at::Tensor()};cpu_flash_check(q,k,v,drop);auto rows=q.size(0)*q.size(1);auto qs=q.reshape({rows,q.size(2),q.size(3)}),ks=k.reshape({rows,k.size(2),k.size(3)}),vs=v.reshape({rows,v.size(2),v.size(3)});auto state=at::zeros({4},at::TensorOptions().dtype(at::kLong).device(at::kCPU));auto [dq,dk,dv,dm]=attention_backward(grad.reshape(qs.sizes()),qs,ks,vs,cpu_flash_mask(q,k,mask),out.reshape(qs.sizes()),lse.reshape({rows,q.size(2)}),0,causal,scale.value_or(1./std::sqrt(double(q.size(3)))),state);return {dq.reshape(q.sizes()),dk.reshape(k.sizes()),dv.reshape(v.sizes())};}
}
TORCH_LIBRARY_IMPL(aten,PrivateUse1,m){m.impl("_scaled_dot_product_flash_attention_for_cpu",&cpu_flash_forward_native);m.impl("_scaled_dot_product_flash_attention_for_cpu_backward",&cpu_flash_backward_native);}
}
