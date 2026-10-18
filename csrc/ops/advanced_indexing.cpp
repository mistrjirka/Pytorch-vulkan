#include "native_utils.h"
#include <ATen/native/TensorAdvancedIndexingUtils.h>
#include <ATen/native/Resize.h>
#include <ATen/MemoryOverlap.h>
#include <torch/library.h>
namespace torch_vulkan::ops {
namespace {
using namespace native;
uint32_t dtype_code(at::ScalarType t){TORCH_CHECK(t==at::kBool||t==at::kFloat||t==at::kInt||t==at::kLong||t==at::kComplexFloat,"Vulkan indexing supports FP32/complex64/int32/int64/bool");return t==at::kBool?0:t==at::kFloat?1:t==at::kInt?2:t==at::kLong?3:4;}
at::Tensor nonzero_native(const at::Tensor& x){
 TORCH_CHECK(can_pack_view(x)&&x.numel()<=INT32_MAX,"nonzero exceeds Vulkan indexing limits");
 if(!x.numel())return at::empty({0,x.dim()},x.options().dtype(at::kLong));
 auto data=packed(x,x.scalar_type()),prefix=at::empty({x.numel()},x.options().dtype(at::kInt)),count=at::empty({1},prefix.options());
 run("nonzero_scan",{base(data,x.scalar_type())},{write_base(prefix,at::kInt),write_base(count,at::kInt)},{u32(x.numel()),dtype_code(x.scalar_type())},1);
 int64_t n=count.cpu().item<int>();auto out=at::empty({n,x.dim()},x.options().dtype(at::kLong));if(!out.numel())return out;
 std::vector<uint32_t> p={u32(x.numel()),u32(x.dim())};p.resize(10,1);for(int d=0;d<x.dim();d++)p[2+d]=u32(x.size(d));
 run("nonzero_write",{base(prefix,at::kInt)},{write_base(out,at::kLong)},p,(x.numel()+127)/128);return out;
}
at::Tensor& nonzero_out(const at::Tensor& x,at::Tensor& out){TORCH_CHECK(out.device()==x.device()&&out.scalar_type()==at::kLong,"nonzero output must be int64 on the input device");at::assert_no_internal_overlap(out);at::assert_no_partial_overlap(out,x);auto result=nonzero_native(x);at::native::resize_output(out,result.sizes());out.copy_(result);return out;}
// Use upstream's public header helpers for mask expansion and broadcasting.
// AdvancedIndex's constructor itself is not exported by libtorch, so construct
// the resulting view metadata here without relying on a private binary symbol.
struct NormalizedIndex {
 at::Tensor src;std::vector<at::Tensor> indices;std::vector<int64_t> indexed_sizes,indexed_strides;
 NormalizedIndex(const at::Tensor& x,const c10::List<std::optional<at::Tensor>>& original){
  TORCH_CHECK_INDEX(original.size()<=size_t(x.dim()),"too many indices for tensor");
  at::native::checkIndexTensorTypes(original,true);
  auto expanded=at::native::expandTensors(x,original,true);
  try{expanded=at::expand_outplace(expanded);}catch(const std::exception&){TORCH_CHECK_INDEX(false,"shape mismatch: indexing tensors could not be broadcast together");}
  TORCH_CHECK_INDEX(expanded.size()<=size_t(x.dim()),"too many indices for tensor");expanded.resize(x.dim());
  auto working=x;if(!at::native::hasContiguousSubspace(expanded))std::tie(working,expanded)=at::native::transposeToFront(working,expanded);
  int before=0,after=0;bool seen=false;std::vector<int64_t> replacement;
  for(int d=0;d<working.dim();d++){
   if(expanded[d].defined()){seen=true;replacement=expanded[d].sizes().vec();indexed_sizes.push_back(working.size(d));indexed_strides.push_back(working.stride(d)*working.element_size());}
   else if(seen)after++;else before++;
  }
  std::vector<int64_t> shape,strides;
  for(int d=0;d<before;d++){shape.push_back(working.size(d));strides.push_back(working.stride(d));}
  shape.insert(shape.end(),replacement.begin(),replacement.end());strides.insert(strides.end(),replacement.size(),0);
  for(int d=working.dim()-after;d<working.dim();d++){shape.push_back(working.size(d));strides.push_back(working.stride(d));}
  src=working.as_strided(shape,strides,working.storage_offset());
  if(src.numel())for(auto size:indexed_sizes)TORCH_CHECK_INDEX(size!=0,"index is out of bounds for dimension with size 0");
  for(auto& index:expanded)if(index.defined()){
   auto reshaped=std::vector<int64_t>(before,1);reshaped.insert(reshaped.end(),replacement.begin(),replacement.end());reshaped.insert(reshaped.end(),after,1);
   indices.push_back(index.to(at::kLong).reshape(reshaped));
  }
 }
};
struct IndexPlan{NormalizedIndex info;at::Tensor indices,metadata;IndexPlan(const at::Tensor& x,const c10::List<std::optional<at::Tensor>>& original):info(x,original){
 TORCH_CHECK(original.size()<=size_t(x.dim()),"too many indices for tensor");TORCH_CHECK(can_pack_view(info.src),"advanced indexing exceeds Vulkan indexing limits");
 int64_t count=info.src.numel();indices=at::empty({int64_t(info.indices.size()),count},x.options().dtype(at::kLong));
 for(size_t d=0;d<info.indices.size();d++)if(count)indices.select(0,d).copy_(info.indices[d].expand(info.src.sizes()).reshape({count}));
 std::vector<uint32_t> meta;for(auto size:info.src.sizes())meta.push_back(u32(size));for(auto stride:info.src.strides())meta.push_back(u32(stride));
 for(auto size:info.indexed_sizes){TORCH_CHECK(size<=INT32_MAX,"advanced index dimension exceeds int32");meta.push_back(u32(size));}for(auto stride:info.indexed_strides)meta.push_back(u32(stride/x.element_size()));
 if(meta.empty())meta.push_back(0);auto host=at::empty({int64_t(meta.size())},at::TensorOptions().dtype(at::kInt));std::memcpy(host.data_ptr(),meta.data(),meta.size()*4);metadata=host.to(x.device());
 }};
at::Tensor advanced_read(const at::Tensor& x,const c10::List<std::optional<at::Tensor>>& indices){
 auto type=dtype_code(x.scalar_type());IndexPlan plan(x,indices);auto out=at::empty(plan.info.src.sizes(),x.options());if(!out.numel())return out;
 auto status=at::zeros({1},x.options().dtype(at::kInt));auto dummy=VulkanAllocator::instance().constant_float_buffer(1,0);
 run("advanced_index",{base(x,x.scalar_type()),plan.indices.numel()?base(plan.indices,at::kLong):dummy,base(plan.metadata,at::kInt),dummy},{write_base(out,x.scalar_type()),write_base(status,at::kInt)},
 {u32(out.numel()),u32(plan.info.src.dim()),u32(plan.info.indices.size()),u32(plan.info.src.storage_offset()),type,0},(out.numel()+(type==0?511:127))/(type==0?512:128));
 TORCH_CHECK_INDEX(status.cpu().item<int>()==0,"index out of bounds");return out;
}
at::Tensor& advanced_read_out(const at::Tensor& x,const c10::List<std::optional<at::Tensor>>& indices,at::Tensor& out){TORCH_CHECK(out.device()==x.device()&&out.scalar_type()==x.scalar_type(),"index output mismatch");at::assert_no_internal_overlap(out);at::assert_no_partial_overlap(out,x);auto result=advanced_read(x,indices);at::native::resize_output(out,result.sizes());out.copy_(result);return out;}
at::Tensor& advanced_write(at::Tensor& x,const c10::List<std::optional<at::Tensor>>& indices,const at::Tensor& values,bool accumulate,bool){
 at::assert_no_internal_overlap(x);at::assert_no_overlap(x,values);auto type=dtype_code(x.scalar_type());TORCH_CHECK(values.scalar_type()==x.scalar_type(),"Index put requires the source and destination dtypes match");
 IndexPlan plan(x,indices);auto vals=values.to(x.device()).expand(plan.info.src.sizes());if(!vals.numel())return x;vals=packed(vals,vals.scalar_type());
 auto dest=write_base(x,x.scalar_type());auto status=at::zeros({1},x.options().dtype(at::kInt));auto dummy=VulkanAllocator::instance().constant_float_buffer(1,0);
 run("advanced_index",{dest,plan.indices.numel()?base(plan.indices,at::kLong):dummy,base(plan.metadata,at::kInt),base(vals,vals.scalar_type())},{dest,write_base(status,at::kInt)},
 {u32(vals.numel()),u32(plan.info.src.dim()),u32(plan.info.indices.size()),u32(plan.info.src.storage_offset()),type,accumulate?2u:1u},1);
 TORCH_CHECK_INDEX(status.cpu().item<int>()==0,"index out of bounds");return x;
}
}
TORCH_LIBRARY_IMPL(aten,PrivateUse1,m){m.impl("nonzero",&nonzero_native);m.impl("nonzero.out",&nonzero_out);m.impl("index.Tensor",&advanced_read);m.impl("index.Tensor_out",&advanced_read_out);m.impl("_index_put_impl_",&advanced_write);}
}
