#include "native_utils.h"
#include "device_copy.h"
#include "sparse_compressed.h"
#include "../vulkan_device.h"
#include <ATen/SparseTensorImpl.h>
#include <ATen/ops/empty_native.h>
#include <ATen/ops/indices_native.h>
#include <ATen/ops/values_native.h>
#include <ATen/ops/_indices_native.h>
#include <ATen/ops/_values_native.h>
#include <ATen/ops/_nnz_native.h>
#include <ATen/ops/sparse_dim_native.h>
#include <ATen/ops/dense_dim_native.h>
#include <ATen/ops/is_coalesced_native.h>
#include <ATen/ops/_coalesced_native.h>
#include <ATen/ops/clone_native.h>
#include <ATen/ops/copy_native.h>
#include <torch/library.h>
namespace torch_vulkan::ops {
namespace {
using namespace native;
bool sparse_dtype(at::ScalarType type){return device_copy_supported(type)||type==at::kComplexFloat;}
uint32_t sparse_kind(at::ScalarType type){return type==at::kBool?1u:type==at::kComplexFloat?4u:type==at::kLong?3u:2u;}
void typed_params(std::vector<uint32_t>& p,const at::Tensor& values){p.push_back(sparse_kind(values.scalar_type()));p.push_back(u32(values.element_size()));}

at::SparseTensorImpl* sparse_impl(const at::Tensor& x){TORCH_CHECK(x.layout()==at::kSparse&&x.device().type()==c10::DeviceType::PrivateUse1,"expected Vulkan COO tensor");return static_cast<at::SparseTensorImpl*>(x.unsafeGetTensorImpl());}
at::Tensor make_sparse(int64_t sd,int64_t dd,at::IntArrayRef shape,const at::Tensor& indices,const at::Tensor& values,bool coalesced){
 TORCH_CHECK(sd>=0&&dd>=0&&sd+dd==shape.size()&&shape.size()<=8&&indices.scalar_type()==at::kLong&&sparse_dtype(values.scalar_type())&&indices.dim()==2&&indices.size(0)==sd&&values.dim()==dd+1&&values.size(0)==indices.size(1)&&indices.device()==values.device(),"invalid Vulkan COO structure");validate_vulkan_device(values.device());for(int a=0;a<shape.size();a++){TORCH_CHECK(shape[a]>=0,"negative sparse size");if(a>=sd)TORCH_CHECK(shape[a]==values.size(a-sd+1),"sparse dense-block shape mismatch");}
 auto impl=c10::make_intrusive<at::SparseTensorImpl>(at::DispatchKeySet(at::DispatchKey::SparsePrivateUse1),at::scalarTypeToTypeMeta(values.scalar_type()));impl->raw_resize_(sd,dd,shape);impl->set_indices_and_values_unsafe(indices,values);impl->set_coalesced(coalesced);return at::Tensor(std::move(impl));
}
at::Tensor with_tensors(int64_t sd,int64_t dd,c10::SymIntArrayRef symbolic,const at::Tensor& indices,const at::Tensor& values,std::optional<at::ScalarType> dtype,std::optional<at::Layout> layout,std::optional<at::Device> device,std::optional<bool> pin,std::optional<bool> coalesced){
 TORCH_CHECK(!pin.value_or(false)&&layout.value_or(at::kSparse)==at::kSparse&&sparse_dtype(dtype.value_or(values.scalar_type())),"Vulkan COO requires supported unpinned storage");auto dev=device.value_or(values.device());validate_vulkan_device(dev);std::vector<int64_t> shape;for(auto d:symbolic)shape.push_back(d.expect_int());auto i=indices.to(dev,at::kLong),v=values.to(dev,dtype.value_or(values.scalar_type()));return make_sparse(sd,dd,shape,i,v,coalesced.value_or(i.size(1)<2));
}
at::Tensor with_dims(int64_t sd,int64_t dd,at::IntArrayRef shape,std::optional<at::ScalarType> dtype,std::optional<at::Layout> layout,std::optional<at::Device> device,std::optional<bool> pin){
 TORCH_CHECK(!pin.value_or(false)&&layout.value_or(at::kSparse)==at::kSparse&&sparse_dtype(dtype.value_or(at::kFloat))&&sd>=0&&dd>=0&&sd+dd==shape.size(),"invalid Vulkan empty COO options");auto dev=device.value_or(at::Device(c10::DeviceType::PrivateUse1,0));validate_vulkan_device(dev);auto options=at::TensorOptions().device(dev).dtype(dtype.value_or(at::kFloat));std::vector<int64_t> value_shape={0};value_shape.insert(value_shape.end(),shape.begin()+sd,shape.end());return make_sparse(sd,dd,shape,at::empty({sd,0},options.dtype(at::kLong)),at::empty(value_shape,options),true);
}
at::Tensor to_sparse_dim(const at::Tensor& x,int64_t sd){
 TORCH_CHECK(sd>0||x.dim()==0,"dense_to_sparse: sparse_dim argument must be in >0 when self.dim()>0");
 TORCH_CHECK(sparse_dtype(x.scalar_type())&&can_pack_view(x)&&sd>=0&&sd<=x.dim(),"Vulkan to_sparse requires supported storage and valid sparse_dim");if(!x.numel()){auto shape=x.sizes().slice(sd).vec();shape.insert(shape.begin(),0);return make_sparse(sd,x.dim()-sd,x.sizes(),at::empty({sd,0},x.options().dtype(at::kLong)),at::empty(shape,x.options()),true);}
 int64_t rows=1,features=1;for(int d=0;d<x.dim();d++)(d<sd?rows:features)*=x.size(d);
 auto ranks=at::empty({rows},x.options().dtype(at::kInt)),count=at::empty({},ranks.options());auto dummy=VulkanAllocator::instance().constant_float_buffer(1,0);auto input=x.numel()?packed(x,x.scalar_type()):at::Tensor();auto count_input=input.defined()&&x.scalar_type()!=at::kFloat?at::ne(input,0).to(at::kFloat):input;
 run("sparse_count",{count_input.defined()?base(count_input):dummy},{rows?write_base(ranks,at::kInt):dummy,write_base(count,at::kInt)},{u32(rows),u32(features)},1);int64_t nnz=count.cpu().item<int>();
 auto indices=at::empty({sd,nnz},x.options().dtype(at::kLong));auto value_shape=x.sizes().slice(sd).vec();value_shape.insert(value_shape.begin(),nnz);auto values=at::empty(value_shape,x.options());
 if(nnz&&features){std::vector<uint32_t> p={u32(rows),u32(features),u32(nnz),u32(sd)};p.resize(12,1);for(int d=0;d<sd;d++)p[4+d]=u32(x.size(d));if(x.scalar_type()!=at::kFloat)typed_params(p,values);run(x.scalar_type()==at::kFloat?"sparse_pack":"sparse_pack_typed",{base(input,input.scalar_type()),base(ranks,at::kInt)},{sd?write_base(indices,at::kLong):dummy,write_base(values,values.scalar_type())},p,(x.numel()+127)/128);}
 return make_sparse(sd,x.dim()-sd,x.sizes(),indices,values,true);
}
at::Tensor to_sparse(const at::Tensor& x,std::optional<at::Layout> layout,at::OptionalIntArrayRef block,std::optional<int64_t> dense){if(layout.value_or(at::kSparse)!=at::kSparse)return to_compressed_native(x,*layout,block,dense);TORCH_CHECK(!block,"blocksize is only supported for compressed block layouts");return to_sparse_dim(x,x.dim()-dense.value_or(0));}
std::vector<uint32_t> dense_params(const at::Tensor& sparse,int64_t count,bool mask){auto* s=sparse_impl(sparse);uint64_t features=1;for(int d=s->sparse_dim();d<sparse.dim();d++)features*=sparse.size(d);std::vector<uint32_t> p={u32(count),u32(features),u32(s->nnz()),u32(s->sparse_dim()),uint32_t(mask)};p.resize(13,1);for(int d=0;d<s->sparse_dim();d++)p[5+d]=u32(sparse.size(d));return p;}
at::Tensor to_dense(const at::Tensor& x,std::optional<at::ScalarType> dtype,std::optional<bool>){TORCH_CHECK(dtype.value_or(x.scalar_type())==x.scalar_type(),"Vulkan COO to_dense dtype conversion requires an explicit values cast");auto* s=sparse_impl(x);auto out=at::empty(x.sizes(),x.options().layout(at::kStrided));if(!out.numel())return out;if(!s->nnz()){out.zero_();return out;}auto i=s->sparse_dim()?packed(s->indices(),at::kLong):at::Tensor(),v=packed(s->values(),s->values().scalar_type());auto dummy=VulkanAllocator::instance().constant_float_buffer(1,0);auto p=dense_params(x,out.numel(),false);if(x.scalar_type()!=at::kFloat)typed_params(p,v);run(x.scalar_type()==at::kFloat?"sparse_dense":"sparse_dense_typed",{i.defined()?base(i,at::kLong):dummy,base(v,v.scalar_type())},{write_base(out,out.scalar_type())},p,(out.numel()+127)/128);return out;}
at::Tensor coalesce_native(const at::Tensor& x){
 auto* s=sparse_impl(x);if(s->coalesced())return x;int64_t n=s->nnz(),sd=s->sparse_dim();if(n<2)return make_sparse(sd,s->dense_dim(),x.sizes(),s->indices().clone(),s->values().clone(),true);
 auto options=s->values().options();auto i=sd?packed(s->indices(),at::kLong):at::Tensor(),v=s->values().numel()?packed(s->values(),s->values().scalar_type()):at::Tensor();auto dummy=VulkanAllocator::instance().constant_float_buffer(1,0);
 auto order=at::empty({n},options.dtype(at::kInt)),heads=at::empty_like(order),ranks=at::empty_like(order),count=at::empty({},order.options());
 run("sparse_coalesce_order",{i.defined()?base(i,at::kLong):dummy},{write_base(order,at::kInt),write_base(heads,at::kInt)},{u32(n),u32(sd)},(n+127)/128);
 run("sparse_coalesce_scan",{base(heads,at::kInt)},{write_base(ranks,at::kInt),write_base(count,at::kInt)},{u32(n)},1);int64_t nnz=count.cpu().item<int>();
 auto outi=at::empty({sd,nnz},s->indices().options());auto shape=s->values().sizes().vec();shape[0]=nnz;auto outv=at::empty(shape,options);int64_t features=s->values().numel()/n;
 std::vector<uint32_t> parameters={u32(n),u32(sd),u32(features),u32(nnz)};if(x.scalar_type()!=at::kFloat)typed_params(parameters,s->values());run(x.scalar_type()==at::kFloat?"sparse_coalesce_values":"sparse_coalesce_values_typed",{i.defined()?base(i,at::kLong):dummy,v.defined()?base(v,v.scalar_type()):dummy,base(order,at::kInt),base(ranks,at::kInt)},{sd?write_base(outi,at::kLong):dummy,outv.numel()?write_base(outv,outv.scalar_type()):dummy},parameters,(n*std::max<int64_t>(1,features)+127)/128);
 return make_sparse(sd,s->dense_dim(),x.sizes(),outi,outv,true);
}
void validate_sparse(const at::Tensor& indices,const at::Tensor& values,at::IntArrayRef shape,std::optional<bool> coalesced,std::optional<bool>){
 TORCH_CHECK(indices.dim()==2&&indices.scalar_type()==at::kLong&&values.dim()>=1&&indices.size(1)==values.size(0)&&indices.device()==values.device(),"invalid sparse indices/values");int64_t sd=indices.size(0);TORCH_CHECK(sd<=8&&sd+values.dim()-1==shape.size(),"invalid sparse rank");
 for(int d=0;d<shape.size();d++){TORCH_CHECK(shape[d]>=0,"negative sparse size");if(d>=sd)TORCH_CHECK(shape[d]==values.size(d-sd+1),"sparse dense shape mismatch");}int64_t n=indices.size(1);if(!n)return;
 auto i=indices.numel()?packed(indices,at::kLong):at::Tensor();auto dummy=VulkanAllocator::instance().constant_float_buffer(1,0);auto flags=at::empty({std::min<int64_t>(256,(n+4095)/4096)},values.options().dtype(at::kFloat));std::vector<uint32_t> p={u32(n),u32(sd),uint32_t(coalesced.value_or(false))};p.resize(11,1);for(int d=0;d<sd;d++)p[3+d]=u32(shape[d]);
 run("sparse_validate",{i.defined()?base(i,at::kLong):dummy},{write_base(flags)},p,flags.numel(),4);TORCH_CHECK((flags.numel()==1?flags:flags.sum()).cpu().item<float>()==0,"invalid sparse coordinates or coalesced ordering");
}
at::Tensor sparse_factory(const at::Tensor& indices,const at::Tensor& raw_values,at::IntArrayRef shape,std::optional<at::ScalarType> dtype,std::optional<at::Layout> layout,std::optional<at::Device> device,std::optional<bool> pin,std::optional<bool> coalesced){
 TORCH_CHECK(layout.value_or(at::kSparse)==at::kSparse&&!pin.value_or(false),"Vulkan COO requires sparse layout and unpinned storage");auto dev=device.value_or(raw_values.device());validate_vulkan_device(dev);auto i=indices.to(dev,at::kLong),v=raw_values.to(dev,dtype.value_or(raw_values.scalar_type()));if(v.dim()==0)v=v.unsqueeze(0);
 TORCH_CHECK(i.dim()==2&&v.dim()>=1,"invalid sparse constructor dimensions");
 if(at::globalContext().checkSparseTensorInvariants().value_or(false))validate_sparse(i,v,shape,coalesced,std::nullopt);
 return at::_sparse_coo_tensor_with_dims_and_tensors(i.size(0),v.dim()-1,shape,i,v,v.options().layout(at::kSparse),coalesced);
}
at::Tensor sparse_factory_infer(const at::Tensor& indices,const at::Tensor& values,std::optional<at::ScalarType> dtype,std::optional<at::Layout> layout,std::optional<at::Device> device,std::optional<bool> pin,std::optional<bool> coalesced){
 TORCH_CHECK(indices.dim()==2&&indices.size(0)<=8,"invalid sparse indices rank");auto dev=device.value_or(values.device());auto i=indices.to(dev,at::kLong);std::vector<int64_t> shape(i.size(0),0);
 if(i.numel()){auto input=packed(i,at::kLong);auto sizes=at::empty({i.size(0)},i.options().dtype(at::kInt));run("sparse_shape",{base(input,at::kLong)},{write_base(sizes,at::kInt)},{u32(i.size(1)),u32(i.size(0))},1);auto host=sizes.cpu();auto ptr=host.data_ptr<int32_t>();for(int d=0;d<i.size(0);d++){TORCH_CHECK(ptr[d]>0,"sparse coordinates must be nonnegative and inferred sizes <= INT32_MAX");shape[d]=ptr[d];}}
 for(int d=1;d<values.dim();d++)shape.push_back(values.size(d));return sparse_factory(i,values,shape,dtype,layout,dev,pin,coalesced);
}
at::Tensor sparse_mask_native(const at::Tensor& raw,const at::Tensor& mask){auto x=raw.layout()==at::kSparse?to_dense(raw,std::nullopt,std::nullopt):raw;auto* s=sparse_impl(mask);TORCH_CHECK(x.sizes()==mask.sizes()&&x.device()==mask.device()&&sparse_dtype(x.scalar_type()),"sparse_mask mismatch");auto values=at::empty(s->values().sizes(),x.options());if(values.numel()){auto i=s->sparse_dim()?packed(s->indices(),at::kLong):at::Tensor(),v=packed(x,x.scalar_type());auto dummy=VulkanAllocator::instance().constant_float_buffer(1,0);auto p=dense_params(mask,values.numel(),true);if(x.scalar_type()!=at::kFloat)typed_params(p,v);run(x.scalar_type()==at::kFloat?"sparse_dense":"sparse_dense_typed",{i.defined()?base(i,at::kLong):dummy,base(v,v.scalar_type())},{write_base(values,values.scalar_type())},p,(values.numel()+127)/128);}return make_sparse(s->sparse_dim(),s->dense_dim(),mask.sizes(),s->indices(),values,s->coalesced());}
}
TORCH_LIBRARY_IMPL(aten,PrivateUse1,m){m.impl("sparse_coo_tensor.indices_size",&sparse_factory);m.impl("sparse_coo_tensor.indices",&sparse_factory_infer);m.impl("_validate_sparse_coo_tensor_args",&validate_sparse);m.impl("_to_sparse",&to_sparse);m.impl("_to_sparse.sparse_dim",&to_sparse_dim);m.impl("sparse_mask",&sparse_mask_native);}
TORCH_LIBRARY_IMPL(aten,AutogradPrivateUse1,m){m.impl("sparse_coo_tensor.indices_size",&sparse_factory);m.impl("sparse_coo_tensor.indices",&sparse_factory_infer);}
TORCH_LIBRARY_IMPL(aten,SparsePrivateUse1,m){
 m.impl("_coalesce",&coalesce_native);m.impl("sparse_mask",&sparse_mask_native);
 m.impl("empty.memory_format",&at::native::empty_sparse_symint);m.impl("_sparse_coo_tensor_with_dims_and_tensors",&with_tensors);m.impl("_sparse_coo_tensor_with_dims",&with_dims);
 m.impl("_to_dense",&to_dense);m.impl("_indices",&at::native::_indices_sparse);m.impl("_values",&at::native::_values_sparse);m.impl("indices",&at::native::indices_sparse);m.impl("values",&at::native::values_sparse);m.impl("_nnz",&at::native::_nnz_sparse);m.impl("sparse_dim",&at::native::sparse_dim_sparse);m.impl("dense_dim",&at::native::dense_dim_sparse);m.impl("is_coalesced",&at::native::is_coalesced_sparse);m.impl("_coalesced_",&at::native::_coalesced_sparse_);m.impl("clone",&at::native::clone_sparse);m.impl("copy_",&at::native::copy_sparse_wrapper_);
}
}
