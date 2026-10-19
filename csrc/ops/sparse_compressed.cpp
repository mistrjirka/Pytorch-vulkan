#include "native_utils.h"
#include "device_copy.h"
#include "sparse_compressed.h"
#include "../vulkan_device.h"
#include <ATen/SparseCsrTensorImpl.h>
#include <ATen/ops/empty_native.h>
#include <ATen/ops/clone_native.h>
#include <ATen/ops/copy_native.h>
#include <torch/library.h>

namespace torch_vulkan::ops {
namespace {
using namespace native;
bool compressed_layout(at::Layout l){return l==at::kSparseCsr||l==at::kSparseCsc||l==at::kSparseBsr||l==at::kSparseBsc;}
bool column_layout(at::Layout l){return l==at::kSparseCsc||l==at::kSparseBsc;}
bool blocked_layout(at::Layout l){return l==at::kSparseBsr||l==at::kSparseBsc;}
bool value_dtype(at::ScalarType t){return device_copy_supported(t)||t==at::kComplexFloat;}
at::SparseCsrTensorImpl* impl(const at::Tensor& x){TORCH_CHECK(compressed_layout(x.layout()),"expected a sparse compressed tensor");return static_cast<at::SparseCsrTensorImpl*>(x.unsafeGetTensorImpl());}
int64_t product(at::IntArrayRef sizes){int64_t n=1;for(auto size:sizes){TORCH_CHECK(size>=0&&(!size||n<=UINT32_MAX/uint64_t(size)),"Vulkan sparse indexing exceeds uint32");n*=size;}return n;}
void validate_members(const at::Tensor& c,const at::Tensor& p,int64_t rows,int64_t cols,int64_t nnz,bool strict=false){
 if(!c.numel())return;auto pointers=packed(c,c.scalar_type()),plain=p.numel()?packed(p,p.scalar_type()):at::Tensor();auto status=at::zeros({1},c.options().dtype(at::kInt));auto dummy=VulkanAllocator::instance().constant_float_buffer(2,0);
 auto count=std::max(c.numel(),p.numel());run("sparse_compressed_validate",{base(pointers,pointers.scalar_type()),plain.defined()?base(plain,plain.scalar_type()):dummy},{write_base(status,at::kInt)},{u32(c.numel()/(rows+1)),u32(rows),u32(cols),u32(nnz),u32(c.element_size()/4),uint32_t(strict)},std::min<int64_t>(256,(count+127)/128),512);
 TORCH_CHECK(status.cpu().item<int32_t>()==0,"sparse compressed index invariants are violated");
}

void validate_compressed_indices(bool,const at::Tensor& c,const at::Tensor& p,int64_t compressed,int64_t plain,int64_t nnz){validate_members(c,p,compressed,plain,nnz,true);}
at::Tensor make_compressed(const at::Tensor& c,const at::Tensor& p,const at::Tensor& values,at::IntArrayRef shape,at::Layout layout,bool validate=false){
 TORCH_CHECK(compressed_layout(layout)&&c.dim()>=1&&p.dim()==c.dim()&&shape.size()>=2,"invalid sparse compressed structure");
 TORCH_CHECK((c.scalar_type()==at::kInt||c.scalar_type()==at::kLong)&&p.scalar_type()==c.scalar_type(),"sparse compressed indices must have matching int32 or int64 dtype");
 TORCH_CHECK(c.device()==p.device()&&p.device()==values.device()&&value_dtype(values.scalar_type()),"sparse compressed device or value dtype mismatch");validate_vulkan_device(values.device());
 int64_t batch=c.dim()-1,block_dims=blocked_layout(layout)?2:0;
 TORCH_CHECK(values.dim()>=batch+1+block_dims&&shape.size()==size_t(values.dim()+1-block_dims),"invalid sparse compressed value rank");
 for(int64_t d=0;d<batch;d++)TORCH_CHECK(c.size(d)==shape[d]&&p.size(d)==shape[d]&&values.size(d)==shape[d],"sparse compressed batch dimensions mismatch");
 auto nnz=p.size(-1);TORCH_CHECK(values.size(batch)==nnz,"sparse compressed nnz/value size mismatch");
 int64_t br=block_dims?values.size(batch+1):1,bc=block_dims?values.size(batch+2):1;
 TORCH_CHECK(br>0&&bc>0&&shape[batch]%br==0&&shape[batch+1]%bc==0,"sparse compressed matrix sizes must be divisible by positive block sizes");
 auto rows=shape[batch]/br,cols=shape[batch+1]/bc;
 TORCH_CHECK(c.size(-1)==(column_layout(layout)?cols:rows)+1,"sparse compressed pointer size mismatch");
 for(int64_t d=batch+2;d<int64_t(shape.size());d++)TORCH_CHECK(shape[d]==values.size(d-1+block_dims),"sparse compressed dense block dimensions mismatch");
 product(shape);if(validate)validate_members(c,p,column_layout(layout)?cols:rows,column_layout(layout)?rows:cols,nnz,true);
 auto result=c10::make_intrusive<at::SparseCsrTensorImpl>(at::DispatchKeySet(at::DispatchKey::SparseCsrPrivateUse1),values.device(),layout,at::scalarTypeToTypeMeta(values.scalar_type()));
 result->set_member_tensors(c,p,values,shape);return at::Tensor(std::move(result));
}
at::Tensor compressed_factory(const at::Tensor& c,const at::Tensor& p,const at::Tensor& values,at::IntArrayRef shape,std::optional<at::ScalarType> dtype,std::optional<at::Layout> layout,std::optional<at::Device> device,std::optional<bool> pin){
 TORCH_CHECK(!pin.value_or(false),"Vulkan sparse compressed storage cannot be pinned");auto dev=device.value_or(values.device());auto type=dtype.value_or(values.scalar_type());
 return make_compressed(c.to(dev),p.to(dev),values.to(dev,type),shape,layout.value_or(at::kSparseCsr),at::globalContext().checkSparseTensorInvariants().value_or(false));
}
at::Tensor compressed_unsafe(const at::Tensor& c,const at::Tensor& p,const at::Tensor& values,c10::SymIntArrayRef symbolic,std::optional<at::ScalarType> dtype,std::optional<at::Layout> layout,std::optional<at::Device> device,std::optional<bool> pin){
 std::vector<int64_t> shape;for(auto size:symbolic)shape.push_back(size.expect_int());return compressed_factory(c,p,values,shape,dtype,layout,device,pin);
}
at::Tensor factory_infer(const at::Tensor& c,const at::Tensor& p,const at::Tensor& values,std::optional<at::ScalarType> dtype,std::optional<at::Layout> layout,std::optional<at::Device> device,std::optional<bool> pin){
 TORCH_CHECK(c.dim()>=1&&values.dim()>=c.dim(),"invalid sparse compressed structure");auto l=layout.value_or(at::kSparseCsr);int64_t batch=c.dim()-1,br=blocked_layout(l)?values.size(batch+1):1,bc=blocked_layout(l)?values.size(batch+2):1;
 auto shape=c.sizes().slice(0,batch).vec();int64_t compressed=c.size(-1)-1,plain=p.numel()?p.max().cpu().item<int64_t>()+1:0;
 shape.push_back((column_layout(l)?plain:compressed)*br);shape.push_back((column_layout(l)?compressed:plain)*bc);
 for(int64_t d=batch+1+(blocked_layout(l)?2:0);d<values.dim();d++)shape.push_back(values.size(d));
 return compressed_factory(c,p,values,shape,dtype,l,device,pin);
}
at::Tensor coordinate_rows(const at::Tensor& x){
 auto* s=impl(x);auto batch=product(x.sizes().slice(0,s->batch_dim()));auto nnz=s->nnz();auto c=s->compressed_indices().reshape({batch,s->compressed_indices().size(-1)}).contiguous();
 auto positions=at::arange(nnz,c.options().dtype(at::kLong)).unsqueeze(0).expand({batch,nnz});return at::searchsorted(c,positions,false,true).sub(1);
}
at::Tensor as_coo(const at::Tensor& x,bool coalesced=true){
 auto* s=impl(x);auto batch_dims=s->batch_dim(),batch=product(x.sizes().slice(0,batch_dims)),nnz=s->nnz(),dense=s->dense_dim();bool block=blocked_layout(x.layout());
 int64_t br=block?s->values().size(batch_dims+1):1,bc=block?s->values().size(batch_dims+2):1,entries=batch*nnz*br*bc;
 auto value_shape=x.sizes().slice(batch_dims+2).vec();value_shape.insert(value_shape.begin(),entries);auto values=s->values().reshape(value_shape);
 auto indices=at::empty({batch_dims+2,entries},s->plain_indices().options().dtype(at::kLong));
 if(entries){
  auto compressed=coordinate_rows(x),plain=s->plain_indices().reshape({batch,nnz}).to(at::kLong);
  auto rows=(column_layout(x.layout())?plain:compressed).reshape({batch,nnz,1,1}).mul(br).add(at::arange(br,plain.options()).reshape({1,1,br,1})).expand({batch,nnz,br,bc}).reshape({entries});
  auto cols=(column_layout(x.layout())?compressed:plain).reshape({batch,nnz,1,1}).mul(bc).add(at::arange(bc,plain.options()).reshape({1,1,1,bc})).expand({batch,nnz,br,bc}).reshape({entries});
  std::vector<at::Tensor> coordinates;int64_t stride=batch;auto batch_indices=at::arange(batch,plain.options());
  for(int64_t d=0;d<batch_dims;d++){stride/=x.size(d);coordinates.push_back(at::floor_divide(batch_indices,stride).remainder(x.size(d)).reshape({batch,1,1,1}).expand({batch,nnz,br,bc}).reshape({entries}));}
  coordinates.push_back(rows);coordinates.push_back(cols);indices=at::stack(coordinates);
 }
 auto result=at::_sparse_coo_tensor_with_dims_and_tensors(batch_dims+2,dense,x.sizes(),indices,values,values.options().layout(at::kSparse),false);return coalesced?result.coalesce():result;
}
at::Tensor compressed_to_dense(const at::Tensor& x,std::optional<at::ScalarType> dtype,std::optional<bool> masked){auto y=as_coo(x);if(dtype&&*dtype!=x.scalar_type())y=y.to(*dtype);return y.to_dense(std::nullopt,masked);}
at::Tensor compressed_to_sparse(const at::Tensor& x,std::optional<at::Layout> layout,at::OptionalIntArrayRef block,std::optional<int64_t> dense){if(layout.value_or(at::kSparse)==at::kSparse)return x.layout()==at::kSparse?x:as_coo(x);return to_compressed_native(x,*layout,block,dense);}
at::Tensor compressed_to_sparse_dim(const at::Tensor& x,int64_t dim){auto y=as_coo(x);if(dim==y.sparse_dim())return y;return y.to_dense().to_sparse(dim);}
at::Tensor to_csr(const at::Tensor& x,std::optional<int64_t> dense){return to_compressed_native(x,at::kSparseCsr,std::nullopt,dense);}
at::Tensor to_csc(const at::Tensor& x,std::optional<int64_t> dense){return to_compressed_native(x,at::kSparseCsc,std::nullopt,dense);}
at::Tensor to_bsr(const at::Tensor& x,at::IntArrayRef block,std::optional<int64_t> dense){return to_compressed_native(x,at::kSparseBsr,block,dense);}
at::Tensor to_bsc(const at::Tensor& x,at::IntArrayRef block,std::optional<int64_t> dense){return to_compressed_native(x,at::kSparseBsc,block,dense);}
at::Tensor compressed_values(const at::Tensor& x){return impl(x)->values();}
at::Tensor compressed_indices(const at::Tensor& x){return impl(x)->compressed_indices();}
at::Tensor plain_indices(const at::Tensor& x){return impl(x)->plain_indices();}
int64_t nnz_native(const at::Tensor& x){return impl(x)->nnz();}
int64_t sparse_dims(const at::Tensor&){return 2;}
int64_t dense_dims(const at::Tensor& x){return impl(x)->dense_dim();}
at::Tensor transpose_compressed(const at::Tensor& x,int64_t a,int64_t b){
 a=at::maybe_wrap_dim(a,x.dim());b=at::maybe_wrap_dim(b,x.dim());if(a==b)return x;auto* s=impl(x);auto batch=s->batch_dim();
 TORCH_CHECK((a==batch&&b==batch+1)||(b==batch&&a==batch+1),"Vulkan sparse compressed transpose requires the two sparse dimensions");
 auto shape=x.sizes().vec();std::swap(shape[a],shape[b]);auto layout=x.layout()==at::kSparseCsr?at::kSparseCsc:x.layout()==at::kSparseCsc?at::kSparseCsr:x.layout()==at::kSparseBsr?at::kSparseBsc:at::kSparseBsr;
 auto values=blocked_layout(layout)?s->values().transpose(batch+1,batch+2):s->values();return make_compressed(s->compressed_indices(),s->plain_indices(),values,shape,layout);
}
at::Tensor t_compressed(const at::Tensor& x){TORCH_CHECK(x.dim()==2,"t() expects a tensor with <= 2 sparse and 0 dense dimensions");return transpose_compressed(x,0,1);}
at::Tensor copy_compressed(const at::Tensor& x,std::optional<at::ScalarType> dtype,std::optional<at::Layout> layout,std::optional<at::Device> device,std::optional<bool> pin,bool nonblocking,std::optional<at::MemoryFormat> memory){
 TORCH_CHECK(!layout||*layout==x.layout(),"to(options) does not support converting layouts");TORCH_CHECK(!memory||*memory==at::MemoryFormat::Preserve,"sparse compressed copy requires preserve memory format");auto* s=impl(x);auto dev=device.value_or(x.device());auto c=s->compressed_indices().to(dev,s->compressed_indices().scalar_type(),nonblocking,true),p=s->plain_indices().to(dev,s->plain_indices().scalar_type(),nonblocking,true),v=s->values().to(dev,dtype.value_or(x.scalar_type()),nonblocking,true);
 return at::_sparse_compressed_tensor_unsafe(c,p,v,x.sizes(),v.options().layout(x.layout()));
}
uint32_t reduce_mode(c10::string_view name){
 if(name=="sum"||name=="add")return 0;if(name=="mean")return 1;if(name=="amax"||name=="max")return 2;if(name=="amin"||name=="min")return 3;
 TORCH_CHECK(false,"reduce argument must be either sum, prod, mean, amax or amin, got ",name);
}
void matrix_inputs(const at::Tensor& a,const at::Tensor& b){
 TORCH_CHECK(a.layout()==at::kSparseCsr&&a.dim()==2&&impl(a)->dense_dim()==0&&b.layout()==at::kStrided&&b.dim()==2,"sparse matrix reduction requires a CSR matrix and a dense matrix");
 TORCH_CHECK(a.scalar_type()==at::kFloat&&b.scalar_type()==at::kFloat&&a.device()==b.device(),"Vulkan sparse matrix reduction requires FP32 tensors on the same device");
 TORCH_CHECK(a.size(1)==b.size(0),"mat1 and mat2 shapes cannot be multiplied");auto* s=impl(a);validate_members(s->compressed_indices(),s->plain_indices(),a.size(0),a.size(1),s->nnz());
}
void sparse_matrix_run(const at::Tensor& a,const at::Tensor& b,const at::Tensor& grad,const at::Tensor& saved,const at::Tensor& out,const at::Tensor& arg,uint32_t reduce,uint32_t phase,float alpha=1,float beta=1){
 if(!out.numel())return;auto* s=impl(a);auto c=packed(s->compressed_indices(),s->compressed_indices().scalar_type()),p=packed(s->plain_indices(),s->plain_indices().scalar_type()),v=packed(s->values()),other=packed(b);
 auto g=grad.defined()?packed(grad):at::Tensor(),args=saved.defined()&&saved.numel()?packed(saved,saved.scalar_type()):at::Tensor();auto entry_rows=s->nnz()?coordinate_rows(a).reshape({s->nnz()}):at::Tensor();
 auto dummy=VulkanAllocator::instance().constant_float_buffer(2,0);auto pb=p.numel()?base(p,p.scalar_type()):dummy,vb=v.numel()?base(v):dummy;
 run("sparse_matrix_fp32",{base(c,c.scalar_type()),pb,vb,other.numel()?base(other):dummy,g.defined()&&g.numel()?base(g):dummy,args.defined()?base(args,args.scalar_type()):dummy,entry_rows.defined()?base(entry_rows,at::kLong):dummy},{write_base(out),arg.defined()&&arg.numel()?write_base(arg,arg.scalar_type()):dummy},{u32(a.size(0)),u32(a.size(1)),u32(phase==3?b.size(1):b.size(-1)),u32(s->nnz()),u32(c.element_size()/4),reduce,phase,uint32_t(arg.defined()&&arg.numel()),u32(c.element_size()/4),bits(alpha),bits(beta)},(out.numel()+127)/128);
}
std::tuple<at::Tensor,at::Tensor> sparse_mm_reduce(const at::Tensor& a,const at::Tensor& b,c10::string_view name){
 matrix_inputs(a,b);auto mode=reduce_mode(name);auto out=at::empty({a.size(0),b.size(1)},b.options());bool save=mode>=2&&(a.requires_grad()||b.requires_grad());
 auto arg=at::empty(save?out.sizes():at::IntArrayRef{0},b.options().dtype(impl(a)->compressed_indices().scalar_type()));sparse_matrix_run(a,b,{},{},out,arg,mode,0);return {out,arg};
}
std::tuple<at::Tensor,at::Tensor> sparse_mm_backward(const at::Tensor& a,const at::Tensor& grad,const at::Tensor& b,c10::string_view name,const at::Tensor& arg,std::array<bool,2> mask){
 matrix_inputs(a,b);auto mode=reduce_mode(name);TORCH_CHECK(grad.sizes()==at::IntArrayRef({a.size(0),b.size(1)})&&grad.scalar_type()==at::kFloat&&grad.device()==a.device(),"sparse matrix gradient shape/device/dtype mismatch");
 if(mode>=2)TORCH_CHECK(arg.scalar_type()==impl(a)->compressed_indices().scalar_type()&&arg.sizes()==grad.sizes(),"sparse extrema backward requires saved argument indices");
 auto* s=impl(a);at::Tensor ga,gb;
 if(mask[0]){auto values=at::empty(s->values().sizes(),b.options());sparse_matrix_run(a,b,grad,arg,values,{},mode,1);ga=make_compressed(s->compressed_indices(),s->plain_indices(),values,a.sizes(),a.layout());}
 if(mask[1]){gb=at::empty(b.sizes(),b.options());sparse_matrix_run(a,b,grad,arg,gb,{},mode,2);}return {ga,gb};
}
at::Tensor compressed_mask(const at::Tensor& raw,const at::Tensor& mask){
 TORCH_CHECK(raw.sizes()==mask.sizes()&&raw.device()==mask.device(),"sparse_mask expects tensors with the same shape and device");auto dense=raw.layout()==at::kStrided?raw:raw.to_dense();auto coo=as_coo(mask,false);auto i=coo._indices();
 c10::List<std::optional<at::Tensor>> coordinates;for(int64_t d=0;d<i.size(0);d++)coordinates.push_back(i.select(0,d));
 auto gathered=dense.is_complex()?at::view_as_complex(at::index(at::view_as_real(at::resolve_conj(dense)),coordinates)):at::index(dense,coordinates);auto values=gathered.reshape(impl(mask)->values().sizes());auto* s=impl(mask);return make_compressed(s->compressed_indices(),s->plain_indices(),values,mask.sizes(),mask.layout());
}
at::Tensor sampled_addmm(const at::Tensor& a,const at::Tensor& b,const at::Tensor& c,const at::Scalar& beta,const at::Scalar& alpha){
 TORCH_CHECK(a.layout()==at::kSparseCsr&&impl(a)->dense_dim()==0,"sparse_sampled_addmm expects a CSR tensor with no dense dimensions");
 TORCH_CHECK(a.scalar_type()==at::kFloat&&b.scalar_type()==at::kFloat&&c.scalar_type()==at::kFloat&&a.device()==b.device()&&b.device()==c.device(),"sparse_sampled_addmm requires FP32 matrices on the same device");
 TORCH_CHECK(b.dim()>=2&&c.dim()>=2&&b.size(-1)==c.size(-2)&&a.size(-2)==b.size(-2)&&a.size(-1)==c.size(-1),"sparse_sampled_addmm matrix shape mismatch");
 if(a.dim()!=2||b.dim()!=2||c.dim()!=2){auto dense=at::matmul(b,c).mul(alpha);if(!beta.equal(0))dense=dense.add(a.to_dense().mul(beta));return compressed_mask(dense,a);}
 matrix_inputs(a,at::empty({a.size(1),0},b.options()));auto* s=impl(a);auto values=at::empty(s->values().sizes(),b.options());sparse_matrix_run(a,b,c,{},values,{},0,3,alpha.toFloat(),beta.toFloat());return make_compressed(s->compressed_indices(),s->plain_indices(),values,a.sizes(),a.layout());
}
at::Tensor& sampled_addmm_out(const at::Tensor& a,const at::Tensor& b,const at::Tensor& c,const at::Scalar& beta,const at::Scalar& alpha,at::Tensor& out){
 TORCH_CHECK(out.layout()==at::kSparseCsr&&out.scalar_type()==a.scalar_type()&&out.device()==a.device(),"sparse_sampled_addmm out layout/dtype/device mismatch");auto value=sampled_addmm(a,b,c,beta,alpha);impl(out)->set_member_tensors(impl(value)->compressed_indices(),impl(value)->plain_indices(),impl(value)->values(),value.sizes());return out;
}
at::Tensor compressed_mm(const at::Tensor& a,const at::Tensor& b){
 if(a.layout()==at::kSparseCsr&&a.dim()==2&&b.layout()==at::kStrided&&b.dim()==2&&a.scalar_type()==at::kFloat&&b.scalar_type()==at::kFloat)return std::get<0>(sparse_mm_reduce(a,b,"sum"));
 auto left=a.layout()==at::kStrided?a:a.to_dense(),right=b.layout()==at::kStrided?b:b.to_dense();auto result=at::mm(left,right);return compressed_layout(a.layout())&&compressed_layout(b.layout())?to_compressed_native(result,at::kSparseCsr,std::nullopt,std::nullopt):result;
}
at::Tensor compressed_addmm(const at::Tensor& input,const at::Tensor& a,const at::Tensor& b,const at::Scalar& beta,const at::Scalar& alpha){
 auto product=compressed_mm(a,b);auto dense=product.layout()==at::kStrided?product:product.to_dense();auto result=dense.mul(alpha);if(!beta.equal(0))result=result.add((input.layout()==at::kStrided?input:input.to_dense()).mul(beta));return input.layout()==at::kStrided?result:compressed_mask(result,input);
}
at::Tensor compressed_mul_scalar(const at::Tensor& a,const at::Scalar& scalar){auto* s=impl(a);return make_compressed(s->compressed_indices(),s->plain_indices(),s->values().mul(scalar),a.sizes(),a.layout());}
at::Tensor compressed_mul(const at::Tensor& a,const at::Tensor& b){auto x=compressed_layout(a.layout())?a:b,other=compressed_layout(a.layout())?b:a;auto* s=impl(x);auto values=other.dim()==0?s->values().mul(other):s->values().mul(impl(compressed_mask(other,x))->values());return make_compressed(s->compressed_indices(),s->plain_indices(),values,x.sizes(),x.layout());}

} // namespace

at::Tensor to_compressed_native(const at::Tensor& raw,at::Layout layout,at::OptionalIntArrayRef block,std::optional<int64_t> requested_dense){
 using namespace native;TORCH_CHECK(compressed_layout(layout),"expected a sparse compressed layout");
 if(raw.layout()==layout&&!blocked_layout(layout))return raw;
 auto x=raw.layout()==at::kStrided?raw:raw.to_dense();auto dense=requested_dense.value_or(0);
 TORCH_CHECK(x.dim()>=2&&dense>=0&&dense<=x.dim()-2,"dense_to_sparse_compressed: invalid dense_dim or matrix rank");TORCH_CHECK(value_dtype(x.scalar_type()),"Vulkan sparse compressed values require a supported dtype; got ",x.scalar_type());
 auto batch_dims=x.dim()-2-dense,batch=product(x.sizes().slice(0,batch_dims));int64_t br=1,bc=1;
 if(blocked_layout(layout)){TORCH_CHECK(block&&block->size()==2,"blocksize needs to be a tuple of size 2");br=(*block)[0];bc=(*block)[1];}
 TORCH_CHECK(br>0&&bc>0&&x.size(batch_dims)%br==0&&x.size(batch_dims+1)%bc==0,"matrix sizes must be divisible by positive block sizes");
 auto rows=x.size(batch_dims)/br,cols=x.size(batch_dims+1)/bc,features=product(x.sizes().slice(batch_dims+2));product(x.sizes());
 auto blocks=x.reshape({batch,rows,br,cols,bc,features}).permute(column_layout(layout)?at::IntArrayRef{0,3,1,2,4,5}:at::IntArrayRef{0,1,3,2,4,5});
 auto compressed=column_layout(layout)?cols:rows,plain=column_layout(layout)?rows:cols;
 auto occupied=blocks.ne(0).reshape({batch,compressed,plain,br*bc*features}).any(-1);auto counts=occupied.to(at::kLong).sum({1,2});int64_t nnz=0;
 if(counts.numel()){nnz=counts.select(0,0).cpu().item<int64_t>();TORCH_CHECK(at::all(counts.eq(nnz)).cpu().item<bool>(),"Expect the same number of specified elements per batch.");}
 auto positions=at::nonzero(occupied.reshape({batch*compressed*plain})).reshape({batch*nnz});
 auto pointer=at::cat({at::zeros({batch,1},positions.options()),occupied.to(at::kLong).sum(-1).cumsum(-1)},-1);
 auto columns=positions.remainder(std::max<int64_t>(1,plain));auto flat=blocks.reshape({batch*compressed*plain,br*bc*features});auto values=x.is_complex()?at::view_as_complex(at::index_select(at::view_as_real(flat),0,positions)):at::index_select(flat,0,positions);
 auto ps=x.sizes().slice(0,batch_dims).vec(),cs=ps,vs=ps;ps.push_back(compressed+1);cs.push_back(nnz);vs.push_back(nnz);if(blocked_layout(layout)){vs.push_back(br);vs.push_back(bc);}for(auto size:x.sizes().slice(batch_dims+2))vs.push_back(size);
 return make_compressed(pointer.reshape(ps),columns.reshape(cs),values.reshape(vs),x.sizes(),layout);
}

TORCH_LIBRARY_IMPL(aten,PrivateUse1,m){
 m.impl("_validate_compressed_sparse_indices",&validate_compressed_indices);
 m.impl("_sparse_compressed_tensor_unsafe",&compressed_unsafe);m.impl("_sparse_csr_tensor_unsafe",&compressed_factory);
 m.impl("sparse_csr_tensor.crow_col_value_size",&compressed_factory);m.impl("sparse_csr_tensor.crow_col_value",&factory_infer);
 m.impl("_to_sparse_csr",&to_csr);m.impl("_to_sparse_csc",&to_csc);m.impl("_to_sparse_bsr",&to_bsr);m.impl("_to_sparse_bsc",&to_bsc);
}
TORCH_LIBRARY_IMPL(aten,SparsePrivateUse1,m){
 m.impl("_to_sparse_csr",&to_csr);m.impl("_to_sparse_csc",&to_csc);m.impl("_to_sparse_bsr",&to_bsr);m.impl("_to_sparse_bsc",&to_bsc);m.impl("_to_sparse",&compressed_to_sparse);
}
TORCH_LIBRARY_IMPL(aten,SparseCsrPrivateUse1,m){
 m.impl("values",&compressed_values);m.impl("crow_indices",&compressed_indices);m.impl("ccol_indices",&compressed_indices);m.impl("col_indices",&plain_indices);m.impl("row_indices",&plain_indices);
 m.impl("_sparse_mm_reduce_impl",&sparse_mm_reduce);m.impl("_sparse_mm_reduce_impl_backward",&sparse_mm_backward);m.impl("sparse_sampled_addmm",&sampled_addmm);m.impl("sparse_sampled_addmm.out",&sampled_addmm_out);
 m.impl("sparse_mask",&compressed_mask);m.impl("mm",&compressed_mm);m.impl("addmm",&compressed_addmm);m.impl("mul.Scalar",&compressed_mul_scalar);m.impl("mul.Tensor",&compressed_mul);
 m.impl("_nnz",&nnz_native);m.impl("sparse_dim",&sparse_dims);m.impl("dense_dim",&dense_dims);m.impl("_to_dense",&compressed_to_dense);
 m.impl("_to_copy",&copy_compressed);m.impl("clone",&at::native::clone_sparse_compressed);m.impl("copy_",&at::native::copy_sparse_compressed_);m.impl("empty.memory_format",&at::native::empty_sparse_compressed_symint);
 m.impl("transpose.int",&transpose_compressed);m.impl("t",&t_compressed);m.impl("_to_sparse",&compressed_to_sparse);m.impl("_to_sparse.sparse_dim",&compressed_to_sparse_dim);
 m.impl("_to_sparse_csr",&to_csr);m.impl("_to_sparse_csc",&to_csc);m.impl("_to_sparse_bsr",&to_bsr);m.impl("_to_sparse_bsc",&to_bsc);
}
} // namespace torch_vulkan::ops
