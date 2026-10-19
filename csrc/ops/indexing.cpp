#include "native_utils.h"
#include "device_copy.h"
#include <ATen/MemoryOverlap.h>
#include <cmath>
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
 if(!index.numel()){auto out=at::empty(mode==0?index.sizes():input.sizes(),input.options());if(mode)out.copy_(input);return out;}
 TORCH_CHECK(device_copy_supported(input.scalar_type())&&can_pack_view(x)&&can_pack_view(idx)&&x.device()==idx.device(),"indexing requires FP32 data and integer indices on same device");
 TORCH_CHECK(idx.scalar_type()==at::kLong||idx.scalar_type()==at::kInt,"index dtype must be int32 or int64");
 TORCH_CHECK(idx.dim()==x.dim(),"Index tensor must have the same number of dimensions as input tensor");
 for(int a=0;a<x.dim();a++)if(a!=d)TORCH_CHECK(idx.size(a)<=x.size(a),"Size does not match at dimension ",a);
 auto out=at::empty(mode==0?index.sizes():input.sizes(),input.options());
 at::Tensor src=source;
 if(source.defined()){
  TORCH_CHECK(source.scalar_type()==input.scalar_type()&&source.device()==x.device()&&source.dim()==index.dim(),"scatter source mismatch");
  if(!src.dim())src=src.view({1});for(int a=0;a<idx.dim();a++){TORCH_CHECK(src.size(a)>=idx.size(a),"scatter source shape too small");src=src.narrow(a,0,idx.size(a));}
 }
 if(!idx.numel()){if(mode)out.copy_(input);return out;}
 idx=packed(idx,idx.scalar_type());validate_index(idx,x.size(d));if(!out.numel())return out;
 if(input.scalar_type()!=at::kFloat){
  if(mode>=4){
   if(input.element_size()<4)return indexed(input.to(at::kInt),raw,index,source.to(at::kInt),mode,value).to(input.scalar_type());
   auto xc=packed(x,x.scalar_type());src=packed(src,src.scalar_type());
   std::vector<uint32_t> p={u32(out.numel()),mode,u32(d),u32(x.dim()),idx.scalar_type()==at::kLong?2u:1u,uint32_t(value!=0)};p.resize(22,1);p.push_back(u32(input.element_size()/4));
   for(int a=0;a<x.dim();a++){p[6+a]=u32(x.size(a));p[14+a]=u32(idx.size(a));}
   run("indexed_integer",{base(xc,xc.scalar_type()),base(idx,idx.scalar_type()),base(src,src.scalar_type())},{write_base(out,out.scalar_type())},p,(out.numel()+127)/128);return out;
  }
  if(input.scalar_type()!=at::kBool&&input.element_size()<4)return indexed(input.to(at::kInt),raw,index,source.defined()?source.to(at::kInt):source,mode,value).to(input.scalar_type());
  c10::List<std::optional<at::Tensor>> coordinates;
  for(int a=0;a<x.dim();a++){if(a==d)coordinates.push_back(idx);else{std::vector<int64_t> view(x.dim(),1);view[a]=idx.size(a);coordinates.push_back(at::arange(idx.size(a),idx.options().dtype(at::kLong)).reshape(view).expand(idx.sizes()));}}
  if(mode==0)return at::index(x,coordinates).reshape(index.sizes());
  auto result=x.clone();at::_index_put_impl_(result,coordinates,src,mode==1,false);return result.reshape(input.sizes());
 }
 auto xc=packed(x);if(src.defined())src=packed(src);
 std::vector<uint32_t> p={u32(out.numel()),mode,u32(d),u32(x.dim()),idx.scalar_type()==at::kLong?2u:1u,bits(value)};p.resize(22,1);
 for(int a=0;a<x.dim();a++){p[6+a]=u32(x.size(a));p[14+a]=u32(idx.size(a));}
 auto sb=src.defined()?base(src):VulkanAllocator::instance().constant_float_buffer(1,0);
 run("indexed",{base(xc),base(idx,idx.scalar_type()),sb},{write_base(out)},p,(out.numel()+127)/128);return out;
}
at::Tensor gather_native(const at::Tensor& x,int64_t d,const at::Tensor& i,bool sparse){axis_for(x,d);if(!i.numel())return at::empty(i.sizes(),x.options());return indexed(x,d,i,{},0);}
at::Tensor scatter_add_native(const at::Tensor& x,int64_t d,const at::Tensor& i,const at::Tensor& s){return indexed(x,d,i,s,1);}
at::Tensor scatter_src_native(const at::Tensor& x,int64_t d,const at::Tensor& i,const at::Tensor& s){return indexed(x,d,i,s,2);}
at::Tensor scatter_value_native(const at::Tensor& x,int64_t d,const at::Tensor& i,const at::Scalar& s){if(x.scalar_type()!=at::kFloat)return indexed(x,d,i,at::full(i.sizes(),s,x.options()),2);return indexed(x,d,i,{},3,s.toFloat());}
at::Tensor& assign(at::Tensor& out,const at::Tensor& value){TORCH_CHECK(out.device()==value.device()&&out.scalar_type()==value.scalar_type(),"Expected out tensor to have dtype matching input tensor and device");at::assert_no_internal_overlap(out);at::native::resize_output(out,value.sizes());out.copy_(value);return out;}
at::Tensor& gather_out_native(const at::Tensor& x,int64_t d,const at::Tensor& i,bool sparse,at::Tensor& out){at::assert_no_internal_overlap(out);at::assert_no_overlap(out,x);at::assert_no_overlap(out,i);auto value=gather_native(x,d,i,sparse);TORCH_CHECK(out.scalar_type()==x.scalar_type()&&out.device()==x.device(),"Expected out tensor to have dtype matching input tensor and device");at::native::resize_output(out,value.sizes());at::assert_no_overlap(out,x);at::assert_no_overlap(out,i);return assign(out,value);}
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
at::Tensor& embedding_renorm_native(at::Tensor& weight,const at::Tensor& indices,double maximum,double order){
 TORCH_CHECK(weight.dim()==2,"'weight' must be 2-D");
 TORCH_CHECK(weight.scalar_type()==at::kFloat&&can_iterate_fp32(weight),"Vulkan embedding renorm requires FP32 strided weight within device indexing limits");
 TORCH_CHECK(indices.device()==weight.device()&&(indices.scalar_type()==at::kInt||indices.scalar_type()==at::kLong),"embedding_renorm_: indices must be Int or Long on the weight device");
 at::assert_no_internal_overlap(weight);if(!indices.numel())return weight;
 auto raw=indices.reshape({-1}).to(at::kLong);
 // CPU embedding_renorm_ indexes rows through select, which accepts negative
 // indices. Normalize those indices before the common GPU bounds validation.
 auto normalized=at::where(raw.lt(0),raw.add(weight.size(0)),raw);
 validate_index(normalized,weight.size(0));if(!weight.size(1)){TORCH_CHECK(!(order<0||std::isinf(order)),"linalg.vector_norm cannot compute the ",order," norm on an empty tensor because the operation does not have an identity");return weight;}
 auto sorted=std::get<0>(at::sort(normalized,0,false));
 auto wb=write_base(weight);auto ib=base(sorted,at::kLong);
 run("embedding_renorm",{wb,ib},{wb},{u32(sorted.numel()),u32(weight.size(1)),u32(weight.storage_offset()),u32(weight.stride(0)),u32(weight.stride(1)),bits(float(maximum)),bits(float(order))},sorted.numel(),1028);
 return weight;
}
uint32_t reduction_mode(c10::string_view reduce,bool scatter){
 if(reduce=="sum"&&scatter)return 4;if(reduce=="prod")return 5;if(reduce=="mean")return 6;if(reduce=="amax")return 7;if(reduce=="amin")return 8;
 TORCH_CHECK(false,"reduce must be ",scatter?"sum, prod, mean, amax or amin":"prod, mean, amax or amin");
}
at::Tensor scatter_reduce_native(const at::Tensor& x,int64_t d,const at::Tensor& index,const at::Tensor& src,c10::string_view reduce,bool include){return indexed(x,d,index,src,reduction_mode(reduce,true),include?1.f:0.f);}
at::Tensor& scatter_reduce_out(const at::Tensor& x,int64_t d,const at::Tensor& index,const at::Tensor& src,c10::string_view reduce,bool include,at::Tensor& out){at::assert_no_partial_overlap(out,x);at::assert_no_partial_overlap(out,src);at::assert_no_overlap(out,index);return assign(out,scatter_reduce_native(x,d,index,src,reduce,include));}
at::Tensor& scatter_reduce_inplace(at::Tensor& x,int64_t d,const at::Tensor& index,const at::Tensor& src,c10::string_view reduce,bool include){at::assert_no_internal_overlap(x);at::assert_no_partial_overlap(x,src);at::assert_no_overlap(x,index);return assign(x,scatter_reduce_native(x,d,index,src,reduce,include));}
at::Tensor expanded_axis_index(const at::Tensor& x,int64_t d,const at::Tensor& index,const at::Tensor& source){
 TORCH_CHECK(index.dim()<=1&&(source.dim()==x.dim()||(!x.dim()&&source.numel()==1)),"index reduction rank mismatch");
 if(!x.dim()){TORCH_CHECK(index.numel()==1,"scalar index reduction requires one index");return index.reshape({});}
 TORCH_CHECK(source.size(d)==index.numel(),"index length must equal source size along dimension");
 for(int a=0;a<x.dim();a++)if(a!=d)TORCH_CHECK(source.size(a)==x.size(a),"source and destination slices must have the same shape");
 std::vector<int64_t> shape(x.dim(),1);shape[d]=index.numel();return index.reshape(shape).expand(source.sizes());
}
at::Tensor index_reduce_native(const at::Tensor& x,int64_t raw,const at::Tensor& index,const at::Tensor& src,c10::string_view reduce,bool include){auto mode=reduction_mode(reduce,false);auto d=axis_for(x,raw);return indexed(x,d,expanded_axis_index(x,d,index,src),src,mode,include?1.f:0.f);}
at::Tensor& index_reduce_out(const at::Tensor& x,int64_t d,const at::Tensor& index,const at::Tensor& src,c10::string_view reduce,bool include,at::Tensor& out){at::assert_no_partial_overlap(out,x);at::assert_no_partial_overlap(out,src);at::assert_no_overlap(out,index);return assign(out,index_reduce_native(x,d,index,src,reduce,include));}
at::Tensor& index_reduce_inplace(at::Tensor& x,int64_t d,const at::Tensor& index,const at::Tensor& src,c10::string_view reduce,bool include){at::assert_no_internal_overlap(x);at::assert_no_partial_overlap(x,src);at::assert_no_overlap(x,index);return assign(x,index_reduce_native(x,d,index,src,reduce,include));}
at::Tensor index_copy_native(const at::Tensor& x,int64_t raw,const at::Tensor& index,const at::Tensor& src){auto d=axis_for(x,raw);return indexed(x,d,expanded_axis_index(x,d,index,src),!x.dim()?src.reshape({}):src,2);}
at::Tensor& index_copy_out(const at::Tensor& x,int64_t d,const at::Tensor& index,const at::Tensor& src,at::Tensor& out){at::assert_no_partial_overlap(out,x);at::assert_no_partial_overlap(out,src);at::assert_no_overlap(out,index);return assign(out,index_copy_native(x,d,index,src));}
at::Tensor& index_copy_inplace(at::Tensor& x,int64_t d,const at::Tensor& index,const at::Tensor& src){at::assert_no_internal_overlap(x);at::assert_no_partial_overlap(x,src);at::assert_no_overlap(x,index);return assign(x,index_copy_native(x,d,index,src));}

at::Tensor legacy_scatter(const at::Tensor& x,int64_t d,const at::Tensor& i,const at::Tensor& src,c10::string_view reduce){TORCH_CHECK(reduce=="add"||reduce=="multiply","reduce argument must be either add or multiply");return scatter_reduce_native(x,d,i,src,reduce=="add"?"sum":"prod",true);}
at::Tensor& legacy_scatter_out(const at::Tensor& x,int64_t d,const at::Tensor& i,const at::Tensor& src,c10::string_view reduce,at::Tensor& out){return assign(out,legacy_scatter(x,d,i,src,reduce));}
at::Tensor& legacy_scatter_inplace(at::Tensor& x,int64_t d,const at::Tensor& i,const at::Tensor& src,c10::string_view reduce){at::assert_no_internal_overlap(x);at::assert_no_partial_overlap(x,src);return assign(x,legacy_scatter(x,d,i,src,reduce));}
at::Tensor legacy_scatter_value(const at::Tensor& x,int64_t d,const at::Tensor& i,const at::Scalar& value,c10::string_view reduce){return legacy_scatter(x,d,i,at::full(i.sizes(),value,x.options()),reduce);}
at::Tensor& legacy_scatter_value_out(const at::Tensor& x,int64_t d,const at::Tensor& i,const at::Scalar& value,c10::string_view reduce,at::Tensor& out){return assign(out,legacy_scatter_value(x,d,i,value,reduce));}
at::Tensor& legacy_scatter_value_inplace(at::Tensor& x,int64_t d,const at::Tensor& i,const at::Scalar& value,c10::string_view reduce){at::assert_no_internal_overlap(x);return assign(x,legacy_scatter_value(x,d,i,value,reduce));}
at::Tensor index_fill_native(const at::Tensor& x,int64_t raw,const at::Tensor& raw_index,const at::Scalar& value){auto bound=x.dim()?x.size(axis_for(x,raw)):1;auto index=at::where(raw_index.lt(0),raw_index.add(bound),raw_index);auto d=axis_for(x,raw);TORCH_CHECK(index.dim()<=1,"Index has to be a vector/scalar");auto shape=x.sizes().vec();if(shape.empty())return scatter_value_native(x,d,index.reshape({}),value);shape[d]=index.numel();std::vector<int64_t> view(x.dim(),1);view[d]=index.numel();return scatter_value_native(x,d,index.reshape(view).expand(shape),value);}
at::Tensor& index_fill_inplace(at::Tensor& x,int64_t d,const at::Tensor& index,const at::Scalar& value){at::assert_no_internal_overlap(x);return assign(x,index_fill_native(x,d,index,value));}
at::Tensor& index_fill_out(const at::Tensor& x,int64_t d,const at::Tensor& index,const at::Scalar& value,at::Tensor& out){return assign(out,index_fill_native(x,d,index,value));}
at::Tensor index_fill_tensor(const at::Tensor& x,int64_t raw,const at::Tensor& raw_index,const at::Tensor& value){auto bound=x.dim()?x.size(axis_for(x,raw)):1;auto index=at::where(raw_index.lt(0),raw_index.add(bound),raw_index);TORCH_CHECK(value.dim()==0,"index_fill_ only supports a 0-dimensional value tensor");auto d=axis_for(x,raw);TORCH_CHECK(index.dim()<=1,"Index has to be a vector/scalar");auto shape=x.sizes().vec();if(shape.empty())return indexed(x,d,index.reshape({}),value.to(x.device(),x.scalar_type()),2);shape[d]=index.numel();std::vector<int64_t> view(x.dim(),1);view[d]=index.numel();return indexed(x,d,index.reshape(view).expand(shape),value.to(x.device(),x.scalar_type()).expand(shape),2);}
at::Tensor& index_fill_tensor_inplace(at::Tensor& x,int64_t d,const at::Tensor& index,const at::Tensor& value){at::assert_no_internal_overlap(x);return assign(x,index_fill_tensor(x,d,index,value));}
at::Tensor& index_fill_tensor_out(const at::Tensor& x,int64_t d,const at::Tensor& index,const at::Tensor& value,at::Tensor& out){return assign(out,index_fill_tensor(x,d,index,value));}
}
TORCH_LIBRARY_IMPL(aten,PrivateUse1,m){
 m.impl("scatter.reduce",&legacy_scatter);m.impl("scatter.reduce_out",&legacy_scatter_out);m.impl("scatter_.reduce",&legacy_scatter_inplace);
 m.impl("scatter.value_reduce",&legacy_scatter_value);m.impl("scatter.value_reduce_out",&legacy_scatter_value_out);m.impl("scatter_.value_reduce",&legacy_scatter_value_inplace);
 m.impl("index_fill.int_Tensor",&index_fill_tensor);m.impl("index_fill_.int_Tensor",&index_fill_tensor_inplace);m.impl("index_fill.int_Tensor_out",&index_fill_tensor_out);
 m.impl("index_fill.int_Scalar",&index_fill_native);m.impl("index_fill_.int_Scalar",&index_fill_inplace);m.impl("index_fill.int_Scalar_out",&index_fill_out);
 m.impl("scatter_reduce.two",&scatter_reduce_native);m.impl("scatter_reduce.two_out",&scatter_reduce_out);m.impl("scatter_reduce_.two",&scatter_reduce_inplace);
 m.impl("index_reduce",&index_reduce_native);m.impl("index_reduce.out",&index_reduce_out);m.impl("index_reduce_",&index_reduce_inplace);
 m.impl("index_copy",&index_copy_native);m.impl("index_copy.out",&index_copy_out);m.impl("index_copy_",&index_copy_inplace);
 m.impl("gather",&gather_native);m.impl("gather.out",&gather_out_native);
 m.impl("scatter_add",&scatter_add_native);m.impl("scatter_add_",&scatter_add__native);m.impl("scatter_add.out",&scatter_add_out_native);
 m.impl("scatter.src",&scatter_src_native);m.impl("scatter_.src",&scatter_src__native);m.impl("scatter.src_out",&scatter_src_out_native);
 m.impl("scatter.value",&scatter_value_native);m.impl("scatter_.value",&scatter_value__native);m.impl("scatter.value_out",&scatter_value_out_native);
 m.impl("index_select",&index_select_native);m.impl("index_select.out",&index_select_out_native);
 m.impl("index_add",&index_add_native);m.impl("index_add_",&index_add__native);m.impl("index_add.out",&index_add_out_native);
 m.impl("embedding_dense_backward",&embedding_backward);m.impl("embedding_renorm_",&embedding_renorm_native);
}
}
