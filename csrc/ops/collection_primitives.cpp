#include "native_utils.h"
#include "device_copy.h"
#include <ATen/WrapDimUtils.h>
#include <torch/library.h>
#include <cmath>
namespace torch_vulkan::ops {
namespace {
using namespace native;
std::tuple<at::Tensor,at::Tensor> select_ordered(const at::Tensor& raw,int64_t dim,bool keep,int64_t k,bool nanmedian,bool median){
 TORCH_CHECK(device_copy_supported(raw.scalar_type())&&raw.scalar_type()!=at::kBool,"selection requires real input");
 auto d=at::maybe_wrap_dim(dim,raw.dim());auto x=raw.dim()?raw:raw.reshape({1});if(!raw.dim())d=0;auto n=x.size(d);
 TORCH_CHECK(n>0,"Expected reduction dim ",d," to have non-zero size");TORCH_CHECK(k>=1&&k<=n,"selected number k out of range for dimension ",d);
 auto [values,order]=at::sort(x,d,false);auto shape=x.sizes().vec();shape[d]=1;auto index=at::full(shape,k-1,x.options().dtype(at::kLong));
 if(nanmedian&&x.scalar_type()==at::kFloat){auto count=at::sum(at::logical_not(at::isnan(x)),{d},true,at::kLong);index=at::where(count.eq(0),at::full({},n/2,index.options()),at::floor_divide(count.sub(1),2).clamp_min(0));}
 auto out=at::gather(values,d,index,false),indices=at::gather(order,d,index,false);
 if(median&&x.scalar_type()==at::kFloat){auto invalid=at::isnan(x);auto any=at::any(invalid,d,true);auto first=at::argmax(invalid.to(at::kLong),d,true);out=at::where(any,at::gather(x,d,first,false),out);indices=at::where(any,first,indices);}
 if(!keep){out=out.squeeze(d);indices=indices.squeeze(d);}if(!raw.dim()){out=out.reshape({});indices=indices.reshape({});}return {out,indices};
}
std::tuple<at::Tensor,at::Tensor> kth(const at::Tensor& x,c10::SymInt k,int64_t dim,bool keep){return select_ordered(x,dim,keep,k.expect_int(),false,false);}
std::tuple<at::Tensor,at::Tensor> median_dim(const at::Tensor& x,int64_t dim,bool keep){auto d=at::maybe_wrap_dim(dim,x.dim());return select_ordered(x,dim,keep,((x.dim()?x.size(d):1)+1)/2,false,true);}
std::tuple<at::Tensor,at::Tensor> nanmedian_dim(const at::Tensor& x,int64_t dim,bool keep){auto d=at::maybe_wrap_dim(dim,x.dim());return select_ordered(x,dim,keep,((x.dim()?x.size(d):1)+1)/2,true,false);}
at::Tensor median_all(const at::Tensor& x){if(!x.numel())return at::full({},NAN,x.options());return std::get<0>(median_dim(x.reshape({-1}),0,false));}
at::Tensor nanmedian_all(const at::Tensor& x){if(!x.numel())return at::full({},NAN,x.options());return std::get<0>(nanmedian_dim(x.reshape({-1}),0,false));}
std::tuple<at::Tensor,at::Tensor> mode_general(const at::Tensor& raw,int64_t dim,bool keep){
 TORCH_CHECK(device_copy_supported(raw.scalar_type()),"mode requires real data");auto d=at::maybe_wrap_dim(dim,raw.dim());auto x=raw.dim()?raw:raw.reshape({1});if(!raw.dim())d=0;auto n=x.size(d);TORCH_CHECK(n>0,"Expected reduction dim to have non-zero size");
 auto dtype=x.element_size()<4?at::kInt:x.scalar_type();auto [values,order]=at::sort(x.to(dtype),d,false);auto shape=x.sizes().vec();if(keep)shape[d]=1;else shape.erase(shape.begin()+d);if(!raw.dim())shape={};
 auto output=at::empty(shape,x.options().dtype(dtype)),indices=at::empty(shape,x.options().dtype(at::kLong));if(!output.numel())return {output.to(raw.scalar_type()),indices};uint64_t inner=1;for(int a=d+1;a<x.dim();a++)inner*=x.size(a);
 auto v=packed(values,dtype),i=packed(order,at::kLong);run("mode_sorted",{base(v,dtype),base(i,at::kLong)},{write_base(output,dtype),write_base(indices,at::kLong)},{u32(output.numel()),u32(n),u32(inner),dtype==at::kFloat?0u:dtype==at::kInt?1u:2u},(output.numel()+127)/128);return {output.to(raw.scalar_type()),indices};
}
at::Tensor isin_general(const at::Tensor& elements,const at::Tensor& test,bool,bool invert){
 TORCH_CHECK(elements.device()==test.device(),"isin inputs must be on the same device");auto dtype=at::result_type(elements,test);TORCH_CHECK(device_copy_supported(dtype),"isin requires real data");auto out=at::empty(elements.sizes(),elements.options().dtype(at::kBool));if(!elements.numel())return out;if(!test.numel())return at::full_like(out,invert);
 auto values=elements.to(dtype),sorted=std::get<0>(at::sort(test.to(dtype).reshape({-1}),0,false));auto index=at::searchsorted(sorted,values,false,false,std::nullopt,std::nullopt);auto present=at::index_select(sorted,0,index.reshape({-1}).clamp_max(sorted.numel()-1)).reshape(elements.sizes()).eq(values);return invert?at::logical_not(present):present;
}
at::Tensor isin_scalar(const at::Tensor& x,const at::Scalar& test,bool unique,bool invert){return isin_general(x,at::scalar_tensor(test,x.options().dtype(at::result_type(x,test))),unique,invert);}
at::Tensor scalar_isin(const at::Scalar& value,const at::Tensor& test,bool unique,bool invert){return isin_general(at::scalar_tensor(value,test.options().dtype(at::result_type(test,value))),test,unique,invert);}
at::Tensor repeated_indices(const at::Tensor& repeats,std::optional<c10::SymInt> size){
 TORCH_CHECK(repeats.dim()==1&&(repeats.scalar_type()==at::kInt||repeats.scalar_type()==at::kLong),"repeats must be a 1-D int or long tensor");TORCH_CHECK(at::all(repeats.ge(0)).cpu().item<bool>(),"repeats can not be negative");auto offsets=at::cumsum(repeats,0,at::kLong);auto total=repeats.numel()?offsets.select(0,repeats.numel()-1).cpu().item<int64_t>():0;TORCH_CHECK(total>=0,"repeat_interleave total size overflow");u32(total);if(size)TORCH_CHECK(size->expect_int()==total,"allocated size does not match required size");auto values=at::arange(total,offsets.options());return at::searchsorted(offsets,values,false,true,std::nullopt,std::nullopt).to(repeats.scalar_type());
}
at::Tensor renorm_general(const at::Tensor& x,const at::Scalar& p,int64_t dim,const at::Scalar& maxnorm){
 TORCH_CHECK(x.dim()>=2,"renorm: input needs at least 2 dimensions, got ",x.dim()," dimensions");TORCH_CHECK(p.toDouble()>0,"renorm: non-positive-norm not supported");TORCH_CHECK(maxnorm.toDouble()>=0,"renorm: expected maxnorm to be >= 0");auto d=at::maybe_wrap_dim(dim,x.dim());std::vector<int64_t> dims;for(int i=0;i<x.dim();i++)if(i!=d)dims.push_back(i);auto norms=at::linalg_vector_norm(x,p,dims,true,std::nullopt);auto factor=at::where(norms.gt(maxnorm),at::full({},maxnorm,norms.options()).div(norms.add(1e-7)),at::ones({},norms.options()));return x.mul(factor);
}
at::Tensor addr_general(const at::Tensor& x,const at::Tensor& a,const at::Tensor& b,const at::Scalar& beta,const at::Scalar& alpha){TORCH_CHECK(a.dim()==1&&b.dim()==1,"addr: expected 1-D vectors");TORCH_CHECK(x.device()==a.device()&&x.device()==b.device(),"addr: expected tensors on the same device");auto product=a.unsqueeze(1).mul(b.unsqueeze(0));auto expanded=x.expand(product.sizes());return beta.toDouble()==0?product.mul(alpha):expanded.mul(beta).add(product,alpha);}
at::Tensor& fill_tensor(at::Tensor& x,const at::Tensor& value){TORCH_CHECK(value.dim()==0,"fill_ only supports 0-dimension value tensor");TORCH_CHECK(x.device()==value.device()||value.is_cpu(),"fill_ value must be on the same device");if(value.is_cpu())return x.fill_(value.item());x.copy_(value.expand(x.sizes()));return x;}
}
TORCH_LIBRARY_IMPL(aten,AutogradPrivateUse1,m){m.impl("renorm",&renorm_general);m.impl("renorm_",[](at::Tensor& x,const at::Scalar& p,int64_t dim,const at::Scalar& maxnorm)->at::Tensor&{auto original=x.clone();x.copy_(renorm_general(original,p,dim,maxnorm));return x;});}
TORCH_LIBRARY_IMPL(aten,PrivateUse1,m){
 m.impl("kthvalue",&kth);m.impl("median",&median_all);m.impl("median.dim",&median_dim);m.impl("nanmedian",&nanmedian_all);m.impl("nanmedian.dim",&nanmedian_dim);m.impl("mode",&mode_general);
 m.impl("isin.Tensor_Tensor",&isin_general);m.impl("isin.Tensor_Scalar",&isin_scalar);m.impl("isin.Scalar_Tensor",&scalar_isin);m.impl("repeat_interleave.Tensor",&repeated_indices);m.impl("renorm",&renorm_general);m.impl("addr",&addr_general);m.impl("fill_.Tensor",&fill_tensor);
}
}
