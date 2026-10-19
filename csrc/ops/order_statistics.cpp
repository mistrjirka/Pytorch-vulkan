#include "native_utils.h"
#include "device_copy.h"
#include <torch/library.h>
#include <ATen/native/Resize.h>
#include <ATen/MemoryOverlap.h>
#include <ATen/WrapDimUtils.h>
namespace torch_vulkan::ops {
namespace {
using namespace native;
int64_t axis(const at::Tensor& x,int64_t d){return at::maybe_wrap_dim(d,x.dim());}
std::tuple<at::Tensor,at::Tensor> sort_native(const at::Tensor& x,int64_t raw,bool descending){
 int64_t d=axis(x,raw);TORCH_CHECK(device_copy_supported(x.scalar_type())&&can_pack_view(x),"Vulkan sort requires packable real data");
 if(x.element_size()<4){auto [v,i]=sort_native(x.to(at::kInt),raw,descending);return {v.to(x.scalar_type()),i};}
 auto values=at::empty(x.sizes(),x.options()),indices=at::empty(x.sizes(),x.options().dtype(at::kLong));
 if(x.numel()){
  uint64_t inner=1;for(int a=d+1;a<x.dim();a++)inner*=x.size(a);
  auto dtype=x.scalar_type();std::vector<uint32_t> p={u32(x.numel()),u32(x.dim()?x.size(d):1),u32(inner),uint32_t(descending)};if(dtype!=at::kFloat)p.push_back(u32(x.element_size()/4));
  run(dtype==at::kFloat?"sort_axis":"sort_integer",{base(packed(x,dtype),dtype)},{write_base(values,dtype),write_base(indices,at::kLong)},p,(x.numel()+127)/128);
 }return {values,indices};
}
std::tuple<at::Tensor,at::Tensor> sort_stable_native(const at::Tensor& x,std::optional<bool>,int64_t d,bool desc){return sort_native(x,d,desc);}
std::tuple<at::Tensor&,at::Tensor&> sort_out_native(const at::Tensor& x,int64_t d,bool desc,at::Tensor& values,at::Tensor& indices){
 TORCH_CHECK(values.device()==x.device()&&indices.device()==x.device()&&values.scalar_type()==x.scalar_type()&&indices.scalar_type()==at::kLong,"sort output mismatch");
 at::assert_no_internal_overlap(values);at::assert_no_internal_overlap(indices);auto [v,i]=sort_native(x,d,desc);at::native::resize_output(values,v.sizes());at::native::resize_output(indices,i.sizes());values.copy_(v);indices.copy_(i);return {values,indices};
}
std::tuple<at::Tensor&,at::Tensor&> sort_stable_out_native(const at::Tensor& x,std::optional<bool>,int64_t d,bool desc,at::Tensor& v,at::Tensor& i){return sort_out_native(x,d,desc,v,i);}
std::tuple<at::Tensor,at::Tensor> topk_native(const at::Tensor& x,c10::SymInt k_,int64_t raw,bool largest,bool){
 int64_t d=axis(x,raw),k=k_.expect_int();TORCH_CHECK(k>=0&&k<=(x.dim()?x.size(d):1),"topk k out of range");auto [v,i]=sort_native(x,d,largest);
 if(x.dim()){v=v.narrow(d,0,k);i=i.narrow(d,0,k);if(k){v=packed(v,v.scalar_type());i=packed(i,at::kLong);}}
 return {v,i};
}
std::tuple<at::Tensor,at::Tensor> general_scan(const at::Tensor& x,int64_t raw,std::optional<at::ScalarType> requested,uint32_t mode){
 auto d=axis(x,raw);TORCH_CHECK(device_copy_supported(x.scalar_type()),"scan supports real input");auto dtype=requested.value_or((mode==2||mode==3)||x.is_floating_point()?x.scalar_type():at::kLong);TORCH_CHECK(device_copy_supported(dtype)&&(dtype!=at::kBool||(mode==2||mode==3)),"unsupported scan output dtype");
 auto compute=dtype==at::kBool||dtype==at::kByte||dtype==at::kChar||dtype==at::kShort?at::kInt:dtype;auto values=at::empty(x.sizes(),x.options().dtype(compute)),indices=at::empty((mode==2||mode==3)?x.sizes():at::IntArrayRef({0}),x.options().dtype(at::kLong));
 if(x.numel()){uint64_t inner=1;for(int a=d+1;a<x.dim();a++)inner*=x.size(a);auto n=x.dim()?x.size(d):1,rows=x.numel()/n;auto input=packed(x.to(compute),compute);auto ib=(mode==2||mode==3)?write_base(indices,at::kLong):VulkanAllocator::instance().constant_float_buffer(1,0);
 run("scan_general",{base(input,compute)},{write_base(values,compute),ib},{u32(rows),u32(n),u32(inner),compute==at::kFloat?0u:compute==at::kInt?1u:2u,mode},(rows+127)/128);
 }return {values.to(dtype),indices};
}
at::Tensor cumprod_native(const at::Tensor& x,int64_t dim,std::optional<at::ScalarType> dtype){return std::get<0>(general_scan(x,dim,dtype,1));}
std::tuple<at::Tensor,at::Tensor> cummax_native(const at::Tensor& x,int64_t dim){return general_scan(x,dim,std::nullopt,2);}
std::tuple<at::Tensor,at::Tensor> cummin_native(const at::Tensor& x,int64_t dim){return general_scan(x,dim,std::nullopt,3);}

at::Tensor scan_native(const at::Tensor& x,int64_t raw,std::optional<at::ScalarType> dtype){
 if(x.scalar_type()!=at::kFloat||dtype.value_or(at::kFloat)!=at::kFloat)return std::get<0>(general_scan(x,raw,dtype,0));
 int64_t d=axis(x,raw);TORCH_CHECK(can_pack_view(x),"cumsum exceeds Vulkan indexing limits");
 auto out=at::empty(x.sizes(),x.options());if(!x.numel())return out;uint64_t inner=1;for(int a=d+1;a<x.dim();a++)inner*=x.size(a);
 int64_t n=x.dim()?x.size(d):1,rows=x.numel()/n;run("scan_axis",{base(packed(x))},{write_base(out)},{u32(rows),u32(n),u32(inner)},rows,516);return out;
}
at::Tensor flip_native(const at::Tensor& x,at::IntArrayRef dims){
 TORCH_CHECK((device_copy_supported(x.scalar_type())||x.scalar_type()==at::kComplexFloat)&&can_pack_view(x),"Vulkan flip requires supported dtype");
 uint32_t mask=0;for(auto raw:dims){auto d=axis(x,raw);TORCH_CHECK(!(mask&(1u<<d)),"flip duplicate dim");mask|=1u<<d;}
 auto out=at::empty(x.sizes(),x.options());if(out.numel()){
  std::vector<uint32_t> p={u32(x.numel()),mask,uint32_t(x.element_size())};p.resize(11,1);for(int d=0;d<x.dim();d++)p[3+d]=u32(x.size(d));
  run("flip",{base(packed(x,x.scalar_type()),x.scalar_type())},{write_base(out,out.scalar_type())},p,(uint64_t(out.numel())*out.element_size()+511)/512);
 }return out;
}
at::Tensor triangle(const at::Tensor& x,int64_t diagonal,bool upper){
 TORCH_CHECK(x.dim()>=2&&can_pack_view(x),"triangular requires packable rank>=2 tensor");auto out=at::empty(x.sizes(),x.options());if(!out.numel())return out;
 diagonal=std::clamp<int64_t>(diagonal,INT32_MIN,INT32_MAX);
 run("triangular",{base(packed(x,x.scalar_type()),x.scalar_type())},{write_base(out,out.scalar_type())},
  {u32(x.numel()),u32(x.size(-2)),u32(x.size(-1)),uint32_t(x.element_size()/4),uint32_t(x.scalar_type()==at::kBool),uint32_t(int32_t(diagonal)),uint32_t(upper)},(x.numel()+(x.scalar_type()==at::kBool?511:127))/(x.scalar_type()==at::kBool?512:128));return out;
}
at::Tensor tril_native(const at::Tensor& x,int64_t d){return triangle(x,d,false);}
at::Tensor triu_native(const at::Tensor& x,int64_t d){return triangle(x,d,true);}
at::Tensor& triangle_out(const at::Tensor& x,int64_t d,at::Tensor& out,bool upper){TORCH_CHECK(x.device()==out.device()&&x.scalar_type()==out.scalar_type(),"triangular output mismatch");at::assert_no_internal_overlap(out);auto result=triangle(x,d,upper);at::native::resize_output(out,result.sizes());out.copy_(result);return out;}
at::Tensor& tril_out(const at::Tensor& x,int64_t d,at::Tensor& out){return triangle_out(x,d,out,false);}
at::Tensor& triu_out(const at::Tensor& x,int64_t d,at::Tensor& out){return triangle_out(x,d,out,true);}
at::Tensor& tril_inplace(at::Tensor& x,int64_t d){return triangle_out(x,d,x,false);}
at::Tensor& triu_inplace(at::Tensor& x,int64_t d){return triangle_out(x,d,x,true);}
std::tuple<at::Tensor,at::Tensor> var_mean_native(const at::Tensor& x,at::OptionalIntArrayRef dims,const std::optional<at::Scalar>& correction,bool keepdim){
 TORCH_CHECK(x.scalar_type()==at::kFloat&&can_pack_view(x),"Vulkan variance requires packable FP32 data");
 auto mean=at::mean(x,dims,true);auto diff=x.sub(mean);auto squares=diff.mul(diff);
 int64_t n=1;if(!dims||dims->empty())n=x.numel();else for(auto d:*dims)n*=x.dim()?x.size(axis(x,d)):1;
 auto variance=at::sum(squares,dims,keepdim).div(std::max(0.0,double(n)-correction.value_or(at::Scalar(1)).toDouble()));
 return {variance,mean.reshape(variance.sizes())};
}
at::Tensor var_native(const at::Tensor& x,at::OptionalIntArrayRef dims,const std::optional<at::Scalar>& correction,bool keepdim){return std::get<0>(var_mean_native(x,dims,correction,keepdim));}
at::Tensor var_dim_native(const at::Tensor& x,at::OptionalIntArrayRef dims,bool unbiased,bool keepdim){return var_native(x,dims,at::Scalar(unbiased?1:0),keepdim);}
at::Tensor var_all_native(const at::Tensor& x,bool unbiased){return var_dim_native(x,std::nullopt,unbiased,false);}
std::tuple<at::Tensor,at::Tensor,at::Tensor> unique_native(const at::Tensor& x,bool sorted,bool inverse,bool counts){
 auto flat=x.reshape({-1});auto empty=at::empty({0},x.options().dtype(at::kLong));if(!x.numel())return {flat.clone(),inverse?empty.reshape(x.sizes()):empty,empty.clone()};auto [values,order]=sort_native(flat,0,false);
 auto change=at::cat({at::ones({1},x.options().dtype(at::kBool)),at::ne(values.narrow(0,1,values.numel()-1),values.narrow(0,0,values.numel()-1))});auto positions=at::nonzero(change).select(1,0);auto output=at::index_select(values,0,positions);auto inv=empty,frequency=empty.clone();
 if(inverse){auto groups=at::cumsum(change,0,at::kLong).sub(1);inv=at::empty_like(order);inv.scatter_(0,order,groups);inv=inv.reshape(x.sizes());}
 if(counts){auto ends=at::cat({positions.narrow(0,1,positions.numel()-1),at::full({1},x.numel(),positions.options())});frequency=ends.sub(positions);}
 return {output,inv,frequency};
}
at::Tensor prod_dim_native(const at::Tensor& x,int64_t raw,bool keep,std::optional<at::ScalarType> dtype){
 auto d=axis(x,raw);auto type=dtype.value_or(x.is_floating_point()?x.scalar_type():at::kLong);TORCH_CHECK(device_copy_supported(type),"prod requires real input/output");if(!x.dim())return x.to(type).clone();if(type==at::kBool)return at::all(x.ne(0),d,keep);
 auto shape=x.sizes().vec();if(keep)shape[d]=1;else shape.erase(shape.begin()+d);auto compute=type==at::kByte||type==at::kChar||type==at::kShort?at::kInt:type;auto out=at::empty(shape,x.options().dtype(compute));if(!out.numel())return out.to(type);if(!x.size(d))return at::ones(shape,x.options().dtype(type));
 uint64_t inner=1;for(int a=d+1;a<x.dim();a++)inner*=x.size(a);auto input=packed(x.to(compute),compute);auto dummy=VulkanAllocator::instance().constant_float_buffer(1,0);run("scan_general",{base(input,compute)},{write_base(out,compute),dummy},{u32(out.numel()),u32(x.size(d)),u32(inner),compute==at::kFloat?0u:compute==at::kInt?1u:2u,5},(out.numel()+127)/128);return out.to(type);
}
at::Tensor prod_all_native(const at::Tensor& x,std::optional<at::ScalarType> dtype){return prod_dim_native(x.reshape({-1}),0,false,dtype);}
std::tuple<at::Tensor,at::Tensor> unique_pair(const at::Tensor& x,bool sorted,bool inverse){auto [values,inv,counts]=unique_native(x,sorted,inverse,false);return {values,inv};}
at::Tensor logcumsumexp_native(const at::Tensor& x,int64_t raw){TORCH_CHECK(x.scalar_type()==at::kFloat,"logcumsumexp requires FP32 input");return std::get<0>(general_scan(x,raw,std::nullopt,4));}
std::tuple<at::Tensor,at::Tensor,at::Tensor> unique_rows(const at::Tensor& x,std::optional<int64_t> raw,bool consecutive,bool inverse,bool counts){
 auto empty=at::empty({0},x.options().dtype(at::kLong));int64_t d=raw?axis(x,*raw):0;auto data=raw?x.movedim(d,0):x.reshape({-1});TORCH_CHECK(!raw||x.dim()>0,"unique_dim requires at least one dimension");auto rows=data.size(0);int64_t cols=1;for(int a=1;a<data.dim();a++)cols*=data.size(a);
 if(raw){int zeroes=0;for(auto size:x.sizes())zeroes+=size==0;TORCH_CHECK(zeroes<=1,"Number of zero sized dimensions is more than one, so unique cannot be applied");TORCH_CHECK(cols>0,"There are 0 sized dimensions, and they aren't selected, so unique cannot be applied");}
 if(!rows)return {raw?x.clone():data.clone(),inverse&&!raw?empty.reshape(x.sizes()):empty,empty.clone()};
 auto flat=data.reshape({rows,cols});auto order=at::arange(rows,empty.options());
 if(!consecutive){auto dtype=x.element_size()<4?at::kInt:x.scalar_type();TORCH_CHECK(dtype==at::kFloat||dtype==at::kInt||dtype==at::kLong,"unique requires real input");auto input=packed(flat.to(dtype),dtype);run("unique_rows",{base(input,dtype)},{write_base(order,at::kLong)},{u32(rows),u32(cols),dtype==at::kFloat?0u:dtype==at::kInt?1u:2u},(rows+127)/128);data=at::index_select(data,0,order);flat=data.reshape({rows,cols});}
 auto changes=at::cat({at::ones({1},x.options().dtype(at::kBool)),at::any(at::ne(flat.narrow(0,1,rows-1),flat.narrow(0,0,rows-1)),1)});auto positions=at::nonzero(changes).select(1,0),output=at::index_select(data,0,positions);if(raw)output=output.movedim(0,d);else output=output.reshape({-1});auto inv=empty,frequency=empty.clone();
 if(inverse){auto groups=at::cumsum(changes,0,at::kLong).sub(1);if(consecutive)inv=groups;else {inv=at::empty_like(groups);inv.scatter_(0,order,groups);}if(!raw)inv=inv.reshape(x.sizes());}
 if(counts){auto ends=at::cat({positions.narrow(0,1,positions.numel()-1),at::full({1},rows,empty.options())});frequency=ends.sub(positions);}return {output,inv,frequency};
}
std::tuple<at::Tensor,at::Tensor,at::Tensor> unique_dim_native(const at::Tensor& x,int64_t dim,bool sorted,bool inverse,bool counts){return unique_rows(x,dim,false,inverse,counts);}
std::tuple<at::Tensor,at::Tensor,at::Tensor> unique_consecutive_native(const at::Tensor& x,bool inverse,bool counts,std::optional<int64_t> dim){return unique_rows(x,dim,true,inverse,counts);}
std::tuple<at::Tensor,at::Tensor,at::Tensor> unique_dim_consecutive_native(const at::Tensor& x,int64_t dim,bool inverse,bool counts){return unique_rows(x,dim,true,inverse,counts);}

at::Tensor searchsorted_general(const at::Tensor& boundaries,const at::Tensor& value,bool out_int32,bool right,std::optional<c10::string_view> side,const std::optional<at::Tensor>& sorter){
 TORCH_CHECK(boundaries.dim()>=1,"boundaries tensor must be at least 1 dimension");TORCH_CHECK(boundaries.device()==value.device(),"boundaries and input value tensors must be on the same device");if(side){TORCH_CHECK(*side=="left"||*side=="right","side can only be 'left' or 'right'");TORCH_CHECK(!right||*side=="right","side and right arguments contradict each other");right=*side=="right";}
 if(boundaries.dim()>1){TORCH_CHECK(value.dim()==boundaries.dim(),"boundaries tensor must be 1 dimension or match the input tensor dimensions");TORCH_CHECK(boundaries.sizes().slice(0,boundaries.dim()-1)==value.sizes().slice(0,value.dim()-1),"boundaries tensor first dimensions must match input tensor dimensions");}
 auto n=boundaries.size(-1);TORCH_CHECK(!out_int32||n<INT32_MAX,"boundaries dimension exceeds int32");auto type=at::result_type(boundaries,value);TORCH_CHECK(device_copy_supported(type),"searchsorted requires real inputs");if(type==at::kBool||type==at::kByte||type==at::kChar||type==at::kShort)type=at::kInt;
 if(sorter){TORCH_CHECK(sorter->scalar_type()==at::kLong&&sorter->device()==boundaries.device()&&sorter->sizes()==boundaries.sizes(),"sorter must have the same size/device as boundaries and int64 dtype");if(sorter->numel())TORCH_CHECK(at::all(at::logical_and(sorter->ge(0),sorter->lt(n))).cpu().item<bool>(),"sorter index out of range");}
 auto result=at::empty(value.sizes(),value.options().dtype(out_int32?at::kInt:at::kLong));if(!result.numel())return result;if(!n)return result.zero_();auto b=packed(boundaries.to(type),type),v=packed(value.to(type),type);auto packed_sorter=sorter?packed(*sorter,at::kLong):at::Tensor();auto order=sorter?base(packed_sorter,at::kLong):VulkanAllocator::instance().constant_float_buffer(2,0);run("searchsorted",{base(b,type),base(v,type),order},{write_base(result,result.scalar_type())},{u32(result.numel()),u32(n),boundaries.dim()==1?0u:u32(value.size(-1)),type==at::kFloat?0u:type==at::kInt?1u:2u,uint32_t(right),uint32_t(sorter.has_value()),out_int32?1u:2u},(result.numel()+127)/128);return result;
}
at::Tensor searchsorted_scalar(const at::Tensor& b,const at::Scalar& value,bool out_int32,bool right,std::optional<c10::string_view> side,const std::optional<at::Tensor>& sorter){return searchsorted_general(b,at::scalar_tensor(value,b.options().dtype(at::result_type(b,value))),out_int32,right,side,sorter);}
at::Tensor bucketize_general(const at::Tensor& value,const at::Tensor& boundaries,bool out_int32,bool right){TORCH_CHECK(boundaries.dim()==1,"boundaries tensor must be 1 dimension");return searchsorted_general(boundaries,value,out_int32,right,std::nullopt,std::nullopt);}
at::Tensor bucketize_scalar(const at::Scalar& value,const at::Tensor& boundaries,bool out_int32,bool right){return bucketize_general(at::scalar_tensor(value,boundaries.options().dtype(at::result_type(boundaries,value))),boundaries,out_int32,right);}

}
TORCH_LIBRARY_IMPL(aten,PrivateUse1,m){
 m.impl("searchsorted.Tensor",&searchsorted_general);m.impl("searchsorted.Scalar",&searchsorted_scalar);m.impl("bucketize.Tensor",&bucketize_general);m.impl("bucketize.Scalar",&bucketize_scalar);
 m.impl("unique_dim",&unique_dim_native);m.impl("unique_consecutive",&unique_consecutive_native);m.impl("unique_dim_consecutive",&unique_dim_consecutive_native);
 m.impl("prod",&prod_all_native);m.impl("prod.dim_int",&prod_dim_native);m.impl("_unique",&unique_pair);m.impl("_logcumsumexp",&logcumsumexp_native);
 m.impl("_unique2",&unique_native);
 m.impl("sort",&sort_native);m.impl("sort.stable",&sort_stable_native);m.impl("sort.values",&sort_out_native);m.impl("sort.values_stable",&sort_stable_out_native);
 m.impl("tril",&tril_native);m.impl("triu",&triu_native);m.impl("tril.out",&tril_out);m.impl("triu.out",&triu_out);m.impl("tril_",&tril_inplace);m.impl("triu_",&triu_inplace);
 m.impl("cumprod",&cumprod_native);m.impl("cummax",&cummax_native);m.impl("cummin",&cummin_native);
 m.impl("topk",&topk_native);m.impl("cumsum",&scan_native);m.impl("flip",&flip_native);
 m.impl("var.correction",&var_native);m.impl("var_mean.correction",&var_mean_native);
}
}
#include "native_utils.h"
#include <ATen/ExpandUtils.h>
#include <ATen/WrapDimUtils.h>
#include <torch/library.h>
#include <ATen/core/dispatch/Dispatcher.h>
namespace torch_vulkan::ops {namespace {
using namespace native;
std::tuple<at::Tensor,at::Tensor,at::Tensor> quantile_ranks_meta(const at::Tensor& q,const at::Tensor& counts,int64_t mode){auto shape=at::infer_size_dimvector(q.sizes(),counts.sizes());return {at::empty(shape,q.options().dtype(at::kLong)),at::empty(shape,q.options().dtype(at::kLong)),at::empty(shape,q.options())};}
std::tuple<at::Tensor,at::Tensor,at::Tensor> quantile_ranks_native(const at::Tensor& q,const at::Tensor& counts,int64_t mode){TORCH_CHECK(q.scalar_type()==at::kFloat&&counts.scalar_type()==at::kLong&&q.device()==counts.device()&&mode>=0&&mode<5,"invalid quantile rank inputs");TORCH_CHECK(at::all(at::logical_and(q.ge(0),q.le(1))).cpu().item<bool>(),"quantile() q values must be in the range [0, 1]");auto [lower,upper,weight]=quantile_ranks_meta(q,counts,mode);if(lower.numel()){auto qs=packed(q.expand(lower.sizes())),ns=packed(counts.expand(lower.sizes()),at::kLong);run("quantile_ranks",{base(qs),base(ns,at::kLong)},{write_base(lower,at::kLong),write_base(upper,at::kLong),write_base(weight)},{u32(lower.numel()),uint32_t(mode)},(lower.numel()+127)/128);}return {lower,upper,weight};}
template<bool Ignore>at::Tensor quantile_tensor_native(const at::Tensor& x,const at::Tensor& q,std::optional<int64_t> dim,bool keep,c10::string_view interpolation){
 TORCH_CHECK(x.numel()>0,"quantile() input tensor must be non-empty");TORCH_CHECK(q.dim()<=1,"quantile() q must be a scalar or 1D tensor");TORCH_CHECK(x.scalar_type()==at::kFloat,"quantile() input tensor must be either float or double dtype");TORCH_CHECK(q.scalar_type()==x.scalar_type(),"quantile() q tensor must be same dtype as the input tensor");TORCH_CHECK(q.device()==x.device(),"quantile() q tensor must be on the same device as the input tensor");
 uint32_t mode=interpolation=="linear"?0:interpolation=="lower"?1:interpolation=="higher"?2:interpolation=="midpoint"?3:interpolation=="nearest"?4:5;TORCH_CHECK(mode!=5,"quantile() interpolation must be one of linear, lower, higher, midpoint or nearest, but got ",interpolation);
 int64_t axis=dim?at::maybe_wrap_dim(*dim,x.dim()):0;auto reduced=!dim?x.flatten():x.dim()==0?x.reshape({1}):x.movedim(axis,-1);auto sorted=std::get<0>(at::sort(reduced,-1,false));auto n=reduced.size(-1);u32(n);auto countshape=reduced.sizes().vec();countshape.back()=1;auto counts=Ignore?at::sum(at::logical_not(at::isnan(reduced)),{-1},true,at::kLong):at::full(countshape,n,x.options().dtype(at::kLong));
 auto qvector=q.reshape({q.numel()});auto shape=at::infer_size_dimvector(counts.sizes(),qvector.sizes());auto quantiles=qvector.expand(shape),lengths=counts.expand(shape);
 static auto rank_op=c10::Dispatcher::singleton().findSchemaOrThrow("torch_vulkan::_quantile_ranks","").typed<decltype(quantile_ranks_native)>();auto [lower,upper,weight]=rank_op.call(quantiles.detach(),lengths,mode);
 if(mode==0){auto factor=at::clamp_min(lengths.sub(1),0).to(at::kFloat);weight=weight.add(quantiles.sub(quantiles.detach()).mul(factor));}
 if constexpr(!Ignore){auto nan=at::any(at::isnan(reduced),-1,true).expand(shape);auto last=at::full({},n-1,lower.options());lower=at::where(nan,last,lower);upper=at::where(nan,last,upper);weight=at::where(nan,at::zeros({},x.options()),weight);}
 auto below=at::gather(sorted,-1,lower,false);auto result=mode==0||mode==3?at::lerp(below,at::gather(sorted,-1,upper,false),weight):below;
 result=q.dim()==0?result.squeeze(-1):result.movedim(-1,0);auto output=x.sizes().vec();if(dim){if(x.dim()){if(keep)output[axis]=1;else output.erase(output.begin()+axis);}}else output=keep?std::vector<int64_t>(x.dim(),1):std::vector<int64_t>{};if(q.dim())output.insert(output.begin(),q.numel());return result.reshape(output);
}
template<bool Ignore>at::Tensor quantile_scalar_native(const at::Tensor& x,double q,std::optional<int64_t> dim,bool keep,c10::string_view interpolation){TORCH_CHECK(q>=0&&q<=1,"quantile() q must be in the range [0, 1] but got ",q);return quantile_tensor_native<Ignore>(x,at::full({},q,x.options()),dim,keep,interpolation);}
}
TORCH_LIBRARY_IMPL(aten,PrivateUse1,m){m.impl("quantile",&quantile_tensor_native<false>);m.impl("quantile.scalar",&quantile_scalar_native<false>);m.impl("nanquantile",&quantile_tensor_native<true>);m.impl("nanquantile.scalar",&quantile_scalar_native<true>);}
TORCH_LIBRARY_IMPL(aten,AutogradPrivateUse1,m){m.impl("quantile",&quantile_tensor_native<false>);m.impl("quantile.scalar",&quantile_scalar_native<false>);m.impl("nanquantile",&quantile_tensor_native<true>);m.impl("nanquantile.scalar",&quantile_scalar_native<true>);}
}

namespace torch_vulkan::ops {
TORCH_LIBRARY_FRAGMENT(torch_vulkan,m){m.def("_quantile_ranks(Tensor q, Tensor lengths, int mode) -> (Tensor lower, Tensor upper, Tensor weight)");}
TORCH_LIBRARY_IMPL(torch_vulkan,PrivateUse1,m){m.impl("_quantile_ranks",&quantile_ranks_native);}
TORCH_LIBRARY_IMPL(torch_vulkan,AutogradPrivateUse1,m){m.impl("_quantile_ranks",torch::CppFunction::makeFallthrough());}
TORCH_LIBRARY_IMPL(torch_vulkan,Meta,m){m.impl("_quantile_ranks",&quantile_ranks_meta);}
}
#include <ATen/native/ReduceOpsUtils.h>
namespace torch_vulkan::ops {namespace {
at::Tensor hash_tensor_native(const at::Tensor& raw,at::IntArrayRef dims,bool keep,int64_t mode){TORCH_CHECK(raw.dim()<=64,"only tensors with up to 64 dims are supported");if(!raw.numel())at::native::zero_numel_check_dims(raw,dims,"hash_tensor");TORCH_CHECK(mode==0,"Unknown hash_tensor mode: ",mode);TORCH_CHECK(device_copy_supported(raw.scalar_type()),"hash_tensor requires real FP32 or integer inputs");
 std::vector<bool> reduce(std::max<int64_t>(1,raw.dim()),dims.empty());for(auto d:dims){d=at::maybe_wrap_dim(d,raw.dim());{TORCH_CHECK(!reduce[d],"dim ",d," appears multiple times in the list of dims");reduce[d]=true;}}
 std::vector<int64_t> order,shape;int64_t rows=1,cols=1;for(int64_t d=0;d<raw.dim();d++){if(reduce[d]){cols*=raw.size(d);if(keep)shape.push_back(1);}else{rows*=raw.size(d);order.push_back(d);shape.push_back(raw.size(d));}}for(int64_t d=0;d<raw.dim();d++)if(reduce[d])order.push_back(d);
 auto x=raw.scalar_type()==at::kFloat?raw:raw.to(at::kLong);x=x.permute(order).reshape({rows,cols});x=packed(x,x.scalar_type());bool floating=x.scalar_type()==at::kFloat;auto options=raw.options().dtype(at::kLong);if(!rows)return at::empty(shape,options).view(at::kUInt64);
 do{auto blocks=(cols+1023)/1024;auto output=at::empty({rows,blocks},options);run("hash_tensor_general",{base(x,x.scalar_type())},{write_base(output,at::kLong)},{u32(rows),u32(cols),u32(blocks),uint32_t(floating)},cols==1?(rows+127)/128:rows*blocks);x=output;cols=blocks;floating=false;}while(cols>1);return x.reshape(shape).view(at::kUInt64);
}
}
TORCH_LIBRARY_IMPL(aten,PrivateUse1,m){m.impl("hash_tensor",&hash_tensor_native);}
}
namespace torch_vulkan::ops {namespace {
at::Tensor bincount_native(const at::Tensor& raw,const std::optional<at::Tensor>& weights,c10::SymInt symbolic_min){TORCH_CHECK(device_copy_supported(raw.scalar_type())&&raw.scalar_type()!=at::kFloat&&raw.scalar_type()!=at::kBool,"bincount is not implemented for ",raw.scalar_type());auto minlength=symbolic_min.expect_int();TORCH_CHECK(minlength>=0,"minlength should be >= 0");if(raw.dim()==1&&!raw.numel())return at::zeros({minlength},raw.options().dtype(at::kLong));TORCH_CHECK(raw.dim()==1,"bincount only supports 1-d non-negative integral inputs.");auto x=raw.to(at::kLong);TORCH_CHECK(x.min().cpu().item<int64_t>()>=0,"bincount only supports 1-d non-negative integral inputs.");auto maximum=x.max().cpu().item<int64_t>();TORCH_CHECK(maximum<INT64_MAX,"maximum value of input overflowed, it should be < ",INT64_MAX," but got ",maximum);bool weighted=weights&&weights->defined();if(weighted){TORCH_CHECK(weights->dim()==1&&weights->size(0)==raw.size(0),"weights should be 1-d and have the same length as input");TORCH_CHECK(weights->device()==raw.device(),"bincount weights device mismatch");TORCH_CHECK(weights->scalar_type()==at::kFloat,"Vulkan weighted bincount requires FP32 weights");}auto bins=std::max(maximum+1,minlength);u32(bins);auto out=at::zeros({bins},raw.options().dtype(weighted?at::kFloat:at::kLong));auto indices=packed(x,at::kLong),values=weighted?packed(*weights):at::zeros({1},raw.options().dtype(at::kFloat));run("bincount_general",{base(indices,at::kLong),base(values),base(out,out.scalar_type())},{write_base(out,out.scalar_type())},{u32(raw.numel()),uint32_t(weighted)},(raw.numel()+127)/128);return out;}
}
TORCH_LIBRARY_IMPL(aten,PrivateUse1,m){m.impl("bincount",&bincount_native);}
}
