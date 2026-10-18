#include "native_utils.h"
#include <ATen/MemoryOverlap.h>
#include <ATen/native/Resize.h>
#include <torch/library.h>
namespace torch_vulkan::ops {
namespace {
using namespace native;
int64_t axis_for(const at::Tensor& x,int64_t d){int64_t rank=std::max<int64_t>(1,x.dim());if(d<0)d+=rank;TORCH_CHECK(d>=0&&d<rank,"index dimension out of range");return d;}
void validate_index(const at::Tensor& idx,int64_t bound){
 TORCH_CHECK(idx.scalar_type()==at::kLong||idx.scalar_type()==at::kInt,"index must be int32 or int64");if(!idx.numel())return;
 auto flags=at::empty({std::min<int64_t>(256,(idx.numel()+4095)/4096)},idx.options().dtype(at::kFloat));
 run("embedding_validate",{base(idx,idx.scalar_type())},{write_base(flags)},
  {u32(idx.numel()),u32(bound),idx.scalar_type()==at::kLong?2u:1u},flags.numel(),512);
 TORCH_CHECK_INDEX((flags.numel()==1?flags:flags.sum()).cpu().item<float>()==0,"index out of range");
}
at::Tensor indexed(const at::Tensor& input,int64_t raw,const at::Tensor& index,const at::Tensor& source,uint32_t mode,float value=0){
 const int64_t d=axis_for(input,raw);auto x=input.dim()?input:input.view({1});auto idx=index.dim()?index:index.view({1});
 TORCH_CHECK(input.scalar_type()==at::kFloat&&can_pack_view(x)&&can_pack_view(idx)&&x.device()==idx.device(),"indexing requires FP32 data and integer indices on same device");
 TORCH_CHECK(idx.scalar_type()==at::kLong||idx.scalar_type()==at::kInt,"index dtype must be int32 or int64");
 TORCH_CHECK(idx.dim()==x.dim(),"indexing rank mismatch");
 for(int a=0;a<x.dim();a++)if(a!=d)TORCH_CHECK(idx.size(a)<=x.size(a),"index shape exceeds input");
 auto out=at::empty(mode==0?index.sizes():input.sizes(),input.options());
 at::Tensor src=source;
 if(source.defined()){
  TORCH_CHECK(source.scalar_type()==at::kFloat&&source.device()==x.device()&&source.dim()==index.dim(),"scatter source mismatch");
  if(!src.dim())src=src.view({1});for(int a=0;a<idx.dim();a++){TORCH_CHECK(src.size(a)>=idx.size(a),"scatter source shape too small");src=src.narrow(a,0,idx.size(a));}
 }
 if(!idx.numel()){if(mode)out.copy_(input);return out;}
 idx=packed(idx,idx.scalar_type());validate_index(idx,x.size(d));if(!out.numel())return out;
 auto xc=packed(x);if(src.defined())src=packed(src);
 std::vector<uint32_t> p={u32(out.numel()),mode,u32(d),u32(x.dim()),idx.scalar_type()==at::kLong?2u:1u,bits(value)};p.resize(22,1);
 for(int a=0;a<x.dim();a++){p[6+a]=u32(x.size(a));p[14+a]=u32(idx.size(a));}
 auto sb=src.defined()?base(src):VulkanAllocator::instance().constant_float_buffer(1,0);
 run("indexed",{base(xc),base(idx,idx.scalar_type()),sb},{write_base(out)},p,(out.numel()+127)/128);return out;
}
at::Tensor gather_native(const at::Tensor& x,int64_t d,const at::Tensor& i,bool sparse){TORCH_CHECK(!sparse,"Vulkan sparse gather gradients are not implemented");return indexed(x,d,i,{},0);}
at::Tensor scatter_add_native(const at::Tensor& x,int64_t d,const at::Tensor& i,const at::Tensor& s){return indexed(x,d,i,s,1);}
at::Tensor scatter_src_native(const at::Tensor& x,int64_t d,const at::Tensor& i,const at::Tensor& s){return indexed(x,d,i,s,2);}
at::Tensor scatter_value_native(const at::Tensor& x,int64_t d,const at::Tensor& i,const at::Scalar& s){return indexed(x,d,i,{},3,s.toFloat());}
at::Tensor& assign(at::Tensor& out,const at::Tensor& value){TORCH_CHECK(out.device()==value.device()&&out.scalar_type()==value.scalar_type(),"index output dtype/device mismatch");at::assert_no_internal_overlap(out);at::native::resize_output(out,value.sizes());out.copy_(value);return out;}
at::Tensor& gather_out_native(const at::Tensor& x,int64_t d,const at::Tensor& i,bool sparse,at::Tensor& out){return assign(out,gather_native(x,d,i,sparse));}
at::Tensor& scatter_add_out_native(const at::Tensor& x,int64_t d,const at::Tensor& i,const at::Tensor& s,at::Tensor& out){return assign(out,scatter_add_native(x,d,i,s));}
at::Tensor& scatter_src_out_native(const at::Tensor& x,int64_t d,const at::Tensor& i,const at::Tensor& s,at::Tensor& out){return assign(out,scatter_src_native(x,d,i,s));}
at::Tensor& scatter_value_out_native(const at::Tensor& x,int64_t d,const at::Tensor& i,const at::Scalar& s,at::Tensor& out){return assign(out,scatter_value_native(x,d,i,s));}
at::Tensor& scatter_add__native(at::Tensor& x,int64_t d,const at::Tensor& i,const at::Tensor& s){at::assert_no_internal_overlap(x);at::assert_no_partial_overlap(x,s);return assign(x,scatter_add_native(x,d,i,s));}
at::Tensor& scatter_src__native(at::Tensor& x,int64_t d,const at::Tensor& i,const at::Tensor& s){at::assert_no_internal_overlap(x);at::assert_no_partial_overlap(x,s);return assign(x,scatter_src_native(x,d,i,s));}
at::Tensor& scatter_value__native(at::Tensor& x,int64_t d,const at::Tensor& i,const at::Scalar& s){at::assert_no_internal_overlap(x);return assign(x,scatter_value_native(x,d,i,s));}
at::Tensor index_select_native(const at::Tensor& x,int64_t raw,const at::Tensor& index){
 int64_t d=axis_for(x,raw);TORCH_CHECK(index.dim()<=1,"index_select requires vector indices");
 auto shape=x.sizes().vec();if(shape.empty()){TORCH_CHECK(index.numel()==1,"scalar index_select requires one index");return gather_native(x,d,index.reshape({}),false);}
 shape[d]=index.numel();std::vector<int64_t> view(x.dim(),1);view[d]=index.numel();auto idx=index.reshape(view).expand(shape);return gather_native(x,d,idx,false);
}
at::Tensor& index_select_out_native(const at::Tensor& x,int64_t d,const at::Tensor& i,at::Tensor& out){return assign(out,index_select_native(x,d,i));}
at::Tensor index_add_native(const at::Tensor& x,int64_t raw,const at::Tensor& index,const at::Tensor& source,const at::Scalar& alpha){
 int64_t d=axis_for(x,raw);TORCH_CHECK(index.dim()<=1&&source.dim()==x.dim(),"index_add rank mismatch");
 if(!x.dim())return scatter_add_native(x,d,index.reshape({}),source.mul(alpha));
 TORCH_CHECK(source.size(d)==index.numel(),"index_add axis size mismatch");for(int a=0;a<x.dim();a++)if(a!=d)TORCH_CHECK(source.size(a)==x.size(a),"index_add shape mismatch");
 std::vector<int64_t> view(x.dim(),1);view[d]=index.numel();return scatter_add_native(x,d,index.reshape(view).expand(source.sizes()),source.mul(alpha));
}
at::Tensor& index_add__native(at::Tensor& x,int64_t d,const at::Tensor& i,const at::Tensor& s,const at::Scalar& a){at::assert_no_internal_overlap(x);at::assert_no_partial_overlap(x,s);return assign(x,index_add_native(x,d,i,s,a));}
at::Tensor& index_add_out_native(const at::Tensor& x,int64_t d,const at::Tensor& i,const at::Tensor& s,const at::Scalar& a,at::Tensor& out){return assign(out,index_add_native(x,d,i,s,a));}
at::Tensor embedding_backward(const at::Tensor& grad,const at::Tensor& indices,c10::SymInt nw_,c10::SymInt pad_,bool frequency){
 int64_t nw=nw_.expect_int(),pad=pad_.expect_int();TORCH_CHECK(nw>=0&&grad.dim()==indices.dim()+1&&grad.scalar_type()==at::kFloat&&grad.device()==indices.device()&&can_pack_view(grad)&&can_pack_view(indices),"embedding backward invalid shapes/device/dtype");
 for(int d=0;d<indices.dim();d++)TORCH_CHECK(grad.size(d)==indices.size(d),"embedding backward shape mismatch");
 auto out=at::empty({nw,grad.size(-1)},grad.options());if(!indices.numel()){out.zero_();return out;}
 auto idx=packed(indices,indices.scalar_type());validate_index(idx,nw);if(!out.numel())return out;
 auto heads=at::empty({nw},indices.options().dtype(at::kInt)),next=at::empty({indices.numel()},heads.options());
 run("embedding_heads",{},{write_base(heads,at::kInt)},{u32(nw)},(nw+127)/128);
 // heads is both read and written by the link pass. Bind it as an input and
 // output so resident transfers and dirty tracking preserve both dependencies.
 auto& ctx=VulkanContext::instance();auto ib=base(idx,idx.scalar_type()),hb=base(heads,at::kInt),nb=base(next,at::kInt);
 auto alg=ctx.get_or_create_algorithm("embedding_links",{ib,hb,nb},grid((idx.numel()+127)/128),{u32(idx.numel()),idx.scalar_type()==at::kLong?2u:1u,uint32_t(at::globalContext().deterministicAlgorithms())});
 if(ctx.can_enqueue({ib,hb,nb}))ctx.enqueue_compute(alg,{ib,hb,nb},{hb,nb});else ctx.run_compute(alg,"embedding_links",{ib,hb},{hb,nb});
 run("embedding_gradient",{base(packed(grad)),hb,nb},{write_base(out)},{u32(out.numel()),u32(grad.size(-1)),uint32_t(pad),uint32_t(frequency)},(out.numel()+127)/128);
 return out;
}
}
TORCH_LIBRARY_IMPL(aten,PrivateUse1,m){
 m.impl("gather",&gather_native);m.impl("gather.out",&gather_out_native);
 m.impl("scatter_add",&scatter_add_native);m.impl("scatter_add_",&scatter_add__native);m.impl("scatter_add.out",&scatter_add_out_native);
 m.impl("scatter.src",&scatter_src_native);m.impl("scatter_.src",&scatter_src__native);m.impl("scatter.src_out",&scatter_src_out_native);
 m.impl("scatter.value",&scatter_value_native);m.impl("scatter_.value",&scatter_value__native);m.impl("scatter.value_out",&scatter_value_out_native);
 m.impl("index_select",&index_select_native);m.impl("index_select.out",&index_select_out_native);
 m.impl("index_add",&index_add_native);m.impl("index_add_",&index_add__native);m.impl("index_add.out",&index_add_out_native);
 m.impl("embedding_dense_backward",&embedding_backward);
}
}
