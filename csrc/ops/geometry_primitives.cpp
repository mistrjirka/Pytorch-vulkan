#include "native_utils.h"
#include <ATen/ExpandUtils.h>
#include <torch/library.h>
#include <ATen/native/im2col_shape_check.h>
namespace torch_vulkan::ops {namespace {
using namespace native;
std::array<int64_t,2> image_pair(at::IntArrayRef values,const char* name,bool zero=false){TORCH_CHECK(values.size()==2,name," must have two entries");return {values[0],values[1]};}
at::Tensor image_columns(const at::Tensor& raw,at::IntArrayRef image_size,at::IntArrayRef kernel,at::IntArrayRef dilation,at::IntArrayRef padding,at::IntArrayRef stride,bool reverse){
 if(raw.scalar_type()==at::kComplexFloat){
  auto real=image_columns(at::real(raw),image_size,kernel,dilation,padding,stride,reverse);
  auto imag=image_columns(at::imag(raw),image_size,kernel,dilation,padding,stride,reverse);
  return at::complex(real,imag);
 }

 auto k=image_pair(kernel,"kernel_size"),d=image_pair(dilation,"dilation"),p=image_pair(padding,"padding",true),s=image_pair(stride,"stride");TORCH_CHECK(raw.scalar_type()==at::kFloat||(!reverse&&raw.scalar_type()==at::kBool),"Vulkan image columns requires FP32 data (or bool im2col)");
 TORCH_CHECK(reverse?(raw.dim()==2||raw.dim()==3):(raw.dim()==3||raw.dim()==4),"image columns input has invalid rank");if(reverse){auto size=image_pair(image_size,"output_size");at::native::col2im_shape_check(raw,{},size[0],size[1],k[0],k[1],d[0],d[1],p[0],p[1],s[0],s[1]);}else at::native::im2col_shape_check(raw,{},k[0],k[1],d[0],d[1],p[0],p[1],s[0],s[1]);bool batched=raw.dim()==(reverse?3:4);auto batch=batched?raw.size(0):1;int64_t h,w,c;
 if(reverse){auto size=image_pair(image_size,"output_size");h=size[0];w=size[1];auto columns=raw.size(-2);TORCH_CHECK(columns%(k[0]*k[1])==0,"col2im channels must be divisible by kernel area");c=columns/(k[0]*k[1]);}else{c=raw.size(-3);h=raw.size(-2);w=raw.size(-1);TORCH_CHECK(c&&h&&w,"im2col input spatial/channel dimensions must be nonzero");}
 auto oh=(h+2*p[0]-d[0]*(k[0]-1)-1)/s[0]+1,ow=(w+2*p[1]-d[1]*(k[1]-1)-1)/s[1]+1;TORCH_CHECK(h+2*p[0]>=d[0]*(k[0]-1)+1&&w+2*p[1]>=d[1]*(k[1]-1)+1&&oh>0&&ow>0,"image columns calculated sliding block shape is nonpositive");if(reverse)TORCH_CHECK(raw.size(-1)==oh*ow,"col2im input column length does not match calculated sliding blocks");
 auto x=raw.to(at::kFloat);std::vector<int64_t> shape=reverse?std::vector<int64_t>{batch,c,h,w}:std::vector<int64_t>{batch,c*k[0]*k[1],oh*ow};if(!batched)shape.erase(shape.begin());auto out=at::empty(shape,x.options());
 if(out.numel()){auto input=packed(x);run("image_columns",{base(input)},{write_base(out)},{u32(out.numel()),u32(batch),u32(c),u32(h),u32(w),u32(k[0]),u32(k[1]),u32(d[0]),u32(d[1]),u32(p[0]),u32(p[1]),u32(s[0]),u32(s[1]),u32(oh),u32(ow),uint32_t(reverse)},(out.numel()+127)/128);}return out.to(raw.scalar_type());
}
at::Tensor im2col_native(const at::Tensor& x,at::IntArrayRef kernel,at::IntArrayRef dilation,at::IntArrayRef padding,at::IntArrayRef stride){return image_columns(x,{},kernel,dilation,padding,stride,false);}
at::Tensor col2im_native(const at::Tensor& x,c10::SymIntArrayRef size,at::IntArrayRef kernel,at::IntArrayRef dilation,at::IntArrayRef padding,at::IntArrayRef stride){auto actual=c10::asIntArrayRefSlow(size,__FILE__,__LINE__);return image_columns(x,actual,kernel,dilation,padding,stride,true);}
at::Tensor heaviside_native(const at::Tensor& x,const at::Tensor& values){TORCH_CHECK(x.scalar_type()==values.scalar_type(),"heaviside is not yet implemented for tensors with different dtypes.");TORCH_CHECK(!x.is_complex(),"heaviside is not yet implemented for complex tensors.");TORCH_CHECK(x.device()==values.device(),"heaviside input device mismatch");return at::where(x.eq(0),values,at::where(x.gt(0),at::ones({},x.options()),at::zeros({},x.options())));}
}
TORCH_LIBRARY_IMPL(aten,PrivateUse1,m){m.impl("im2col",&im2col_native);m.impl("col2im",&col2im_native);m.impl("heaviside",&heaviside_native);}
}
namespace torch_vulkan::ops {namespace {
at::Tensor cdist_forward_native(const at::Tensor& x,const at::Tensor& y,double p,std::optional<int64_t> mode){TORCH_CHECK(x.dim()>=2&&y.dim()>=2,"cdist only supports at least 2D tensors");TORCH_CHECK(x.size(-1)==y.size(-1),"X1 and X2 must have the same number of columns");TORCH_CHECK(x.scalar_type()==at::kFloat&&y.scalar_type()==at::kFloat&&x.device()==y.device(),"cdist requires FP32 tensors on the same device");TORCH_CHECK(p>=0,"cdist only supports non-negative p values");TORCH_CHECK(mode.value_or(0)>=0&&mode.value_or(0)<=2,"invalid cdist compute mode");
 auto difference=x.unsqueeze(-2).sub(y.unsqueeze(-3));auto absolute=at::abs(difference);if(p==0)return at::sum(absolute.ne(0),{-1},false,at::kFloat);if(p==1)return at::sum(absolute,{-1},false);if(std::isinf(p)){if(!x.size(-1)){auto shape=at::infer_size_dimvector(x.sizes().slice(0,x.dim()-2),y.sizes().slice(0,y.dim()-2));shape.push_back(x.size(-2));shape.push_back(y.size(-2));return at::zeros(shape,x.options());}return at::amax(absolute,{-1},false);}return at::pow(at::sum(at::pow(absolute,p),{-1},false),1./p);
}
at::Tensor cdist_backward_native(const at::Tensor& grad,const at::Tensor& x,const at::Tensor& y,double p,const at::Tensor& distance){auto delta=x.unsqueeze(-2).sub(y.unsqueeze(-3));auto abs=at::abs(delta);at::Tensor derivative;if(p==0)return at::zeros_like(x);if(p==1)derivative=at::sign(delta);else if(std::isinf(p))derivative=at::sign(delta).mul(abs.eq(distance.unsqueeze(-1)));else{auto factor=at::pow(abs,p-1).mul(at::sign(delta));derivative=at::where(at::logical_or(abs.eq(0),distance.unsqueeze(-1).eq(0)),at::zeros({},x.options()),factor.div(at::pow(distance,p-1).unsqueeze(-1)));}return at::sum(derivative.mul(grad.unsqueeze(-1)),{-2},false).sum_to_size(x.sizes());}
}
TORCH_LIBRARY_IMPL(aten,PrivateUse1,m){m.impl("_cdist_forward",&cdist_forward_native);m.impl("_cdist_backward",&cdist_backward_native);}
}
namespace torch_vulkan::ops {namespace {
at::Tensor unfold_backward_native(const at::Tensor& grad,c10::SymIntArrayRef symbolic,int64_t dim,int64_t size,int64_t step){
 auto original=c10::asIntArrayRefSlow(symbolic,__FILE__,__LINE__).vec();
 if(original.empty()){
  at::maybe_wrap_dim(dim,0);TORCH_CHECK(size>=0&&size<=1&&step>0,"invalid unfold window/step");if(size==0)return at::zeros({},grad.options());return grad.sum().reshape({});
 }
 TORCH_CHECK(original.size()<=8,"Vulkan unfold backward supports at most 8 input dimensions");
 dim=at::maybe_wrap_dim(dim,original.size());
 TORCH_CHECK(size>=0&&size<=original[dim]&&step>0,"invalid unfold window/step");
 TORCH_CHECK(grad.scalar_type()==at::kFloat||grad.scalar_type()==at::kComplexFloat,"unfold backward requires FP32 or complex64 gradients");
 TORCH_CHECK(step>=size||grad.dim()==int64_t(original.size()+1),"unfold backward gradient rank mismatch");
 TORCH_CHECK(step>=size||grad.size(-1)==size,"unfold backward gradient window size mismatch");
 // Overlapping CPU folds use the supplied frame count; non-overlapping folds
 // copy into an unfolded output view, broadcasting a singleton frame axis.
 int64_t windows=step>=size?(original[dim]-size)/step+1:grad.size(dim);
 auto shape=original;shape[dim]=windows;shape.push_back(size);
 auto expanded=grad.expand(shape);
 auto out=at::empty(original,grad.options());if(!out.numel())return out;
 if(size==0||!grad.numel()){out.zero_();return out;}
 auto contiguous=packed(expanded.contiguous().view({-1}),grad.scalar_type());
 auto words=grad.scalar_type()==at::kComplexFloat?2u:1u;
 std::vector<uint32_t> params={u32(out.numel()*words),u32(original.size()),u32(dim),u32(size),u32(step),u32(windows),words};params.resize(15,1);
 for(size_t d=0;d<original.size();d++)params[7+d]=u32(original[d]);
 run("unfold_backward_general",{base(contiguous,grad.scalar_type())},{write_base(out,grad.scalar_type())},params,(out.numel()*words+127)/128);return out;
}
}
TORCH_LIBRARY_IMPL(aten,PrivateUse1,m){m.impl("unfold_backward",&unfold_backward_native);}
}
#include "native_utils.h"
#include <ATen/WrapDimUtils.h>
#include <torch/library.h>
namespace torch_vulkan::ops {namespace {
using namespace native;
using OptionalSegments=std::optional<at::Tensor>;
uint32_t segment_mode(c10::string_view reduce){if(reduce=="sum")return 0;if(reduce=="mean")return 1;if(reduce=="max"||reduce=="amax")return 2;if(reduce=="min"||reduce=="amin")return 3;if(reduce=="prod")return 4;TORCH_CHECK(false,"reduce argument must be either sum, prod, mean, amax or amin, got ",reduce);}
at::Tensor segment_boundaries(const at::Tensor& data,const OptionalSegments& lengths,const OptionalSegments& offsets,int64_t axis,bool unsafe){bool offset=offsets&&offsets->defined(),length=lengths&&lengths->defined();TORCH_CHECK(offset||length,"segment_reduce(): Either lengths or offsets must be defined.");auto source=offset?*offsets:*lengths;TORCH_CHECK(source.device()==data.device()&&data.dim()>=source.dim(),"segment_reduce(): segment tensor device/rank mismatch");TORCH_CHECK(axis==source.dim()-1,"segment_reduce(): Expected axis to be the last dimension of ",offset?"offsets":"lengths"," but got ",axis,".");TORCH_CHECK(source.scalar_type()==at::kInt||source.scalar_type()==at::kLong,"segment_reduce requires int32 or int64 segments");TORCH_CHECK(source.sizes().slice(0,axis)==data.sizes().slice(0,axis),"segment_reduce segment batch shape mismatch");auto indices=source.to(at::kLong);
 if(!offset){if(!unsafe){TORCH_CHECK(indices.min().cpu().item<int64_t>()>=0,"lengths contains negative value!");TORCH_CHECK(at::all(indices.sum({-1}).eq(data.size(axis))).cpu().item<bool>(),"segment_reduce(): Expected all rows of lengths along axis ",axis,"to sum to data.size(lengths.dim()-1) when !unsafe.");}auto shape=indices.sizes().vec();shape.back()=1;indices=at::cat({at::zeros(shape,indices.options()),at::cumsum(indices,-1,at::kLong)},-1);}
 TORCH_CHECK(indices.size(-1)>=1,"segment_reduce offsets must have at least one entry");if(indices.numel()){auto valid=at::logical_and(indices.ge(0),indices.le(data.size(axis)));if(indices.size(-1)>1){auto left=indices.narrow(-1,0,indices.size(-1)-1),right=indices.narrow(-1,1,indices.size(-1)-1);TORCH_CHECK(at::all(right.ge(left)).cpu().item<bool>(),"segment_reduce offsets must be nondecreasing");}TORCH_CHECK(at::all(valid).cpu().item<bool>(),"segment_reduce offsets exceed data bounds");}return indices;
}
at::Tensor segment_run(const at::Tensor& data,const at::Tensor& bounds,int64_t axis,uint32_t mode,const std::optional<at::Scalar>& initial,const at::Tensor& grad={},const at::Tensor& values={}){TORCH_CHECK(data.scalar_type()==at::kFloat,"Vulkan segment_reduce requires FP32 data");int64_t segments=bounds.size(-1)-1,length=data.size(axis),inner=1;for(int64_t d=axis+1;d<data.dim();d++)inner*=data.size(d);auto shape=data.sizes().vec();shape[axis]=segments;auto out=at::empty(grad.defined()?data.sizes():at::IntArrayRef(shape),data.options());if(!out.numel())return out;auto input=packed(data),offsets=packed(bounds,at::kLong);auto gradient=grad.defined()?packed(grad):at::zeros({1},data.options()),output=values.defined()?packed(values):at::zeros({1},data.options());run("segment_reduce_general",{base(input),base(offsets,at::kLong),base(gradient),base(output)},{write_base(out)},{u32(out.numel()),u32(length),u32(segments),u32(inner),mode,uint32_t(grad.defined()),uint32_t(initial.has_value()),bits(initial?initial->toFloat():0)},(out.numel()+127)/128);return out;}
at::Tensor segment_reduce_native(const at::Tensor& data,c10::string_view reduce,const OptionalSegments& lengths,const OptionalSegments& indices,const OptionalSegments& offsets,int64_t axis,bool unsafe,const std::optional<at::Scalar>& initial){axis=at::maybe_wrap_dim(axis,data.dim());TORCH_CHECK(!indices||!indices->defined(),"segment_reduce(): indices based reduction is not supported yet.");auto mode=segment_mode(reduce);auto bounds=segment_boundaries(data,lengths,offsets,axis,unsafe);return segment_run(data,bounds,axis,mode,initial);}
at::Tensor segment_backward_native(const at::Tensor& grad,const at::Tensor& output,const at::Tensor& data,c10::string_view reduce,const OptionalSegments& lengths,const OptionalSegments& offsets,int64_t axis,const std::optional<at::Scalar>& initial){axis=at::maybe_wrap_dim(axis,data.dim());auto bounds=segment_boundaries(data,lengths,offsets,axis,true);return segment_run(data,bounds,axis,segment_mode(reduce),initial,grad,output);}
}
TORCH_LIBRARY_IMPL(aten,PrivateUse1,m){m.impl("segment_reduce",&segment_reduce_native);m.impl("_segment_reduce_backward",&segment_backward_native);}
}
namespace torch_vulkan::ops {namespace {
at::Tensor pdist_forward_native(const at::Tensor& x,double p){TORCH_CHECK(x.dim()==2,"pdist only supports 2D tensors, got: ",x.dim(),"D");TORCH_CHECK(p>=0,"pdist only supports non-negative p values");TORCH_CHECK(x.scalar_type()==at::kFloat,"pdist requires FP32 input");auto n=x.size(0);auto pair=at::triu_indices(n,n,1,x.options().dtype(at::kLong));auto distance=cdist_forward_native(x,x,p,2);return distance.reshape({-1}).index_select(0,pair.select(0,0).mul(n).add(pair.select(0,1)));}
at::Tensor pdist_backward_native(const at::Tensor& grad,const at::Tensor& x,double p,const at::Tensor& distance){TORCH_CHECK(x.dim()==2&&x.scalar_type()==at::kFloat,"pdist backward requires FP32 2D input");auto n=x.size(0),k=x.size(1);auto pairs=at::triu_indices(n,n,1,x.options().dtype(at::kLong));auto row=pairs.select(0,0),col=pairs.select(0,1);auto delta=x.index_select(0,row).sub(x.index_select(0,col));auto absolute=at::abs(delta);at::Tensor derivative;if(p==0)return at::zeros_like(x);if(p==1)derivative=at::sign(delta);else if(std::isinf(p))derivative=at::sign(delta).mul(absolute.eq(distance.unsqueeze(-1)));else derivative=at::where(at::logical_or(absolute.eq(0),distance.unsqueeze(-1).eq(0)),at::zeros({},x.options()),at::pow(absolute,p-1).mul(at::sign(delta)).div(at::pow(distance,p-1).unsqueeze(-1)));auto values=derivative.mul(grad.unsqueeze(-1));auto out=at::zeros_like(x);out.index_add_(0,row,values);out.index_add_(0,col,values.neg());return out;}
}
TORCH_LIBRARY_IMPL(aten,PrivateUse1,m){m.impl("_pdist_forward",&pdist_forward_native);m.impl("_pdist_backward",&pdist_backward_native);}
}
