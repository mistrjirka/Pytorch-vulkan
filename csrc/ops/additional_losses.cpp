#include "native_utils.h"
#include <torch/library.h>
namespace torch_vulkan::ops {
namespace {
using namespace native;
void loss_inputs(const at::Tensor& x,const at::Tensor& target){TORCH_CHECK(x.scalar_type()==at::kFloat&&target.scalar_type()==at::kFloat&&x.device()==target.device(),"loss expects float32 input and target on the same device");}
at::Tensor reduce_loss(const at::Tensor& value,int64_t reduction){TORCH_CHECK(reduction>=0&&reduction<=2,"invalid loss reduction");return reduction==0?value:reduction==1?value.mean():value.sum();}
at::Tensor smooth_loss(const at::Tensor& x,const at::Tensor& target,int64_t reduction,double beta){loss_inputs(x,target);TORCH_CHECK(beta>=0,"smooth_l1_loss does not support negative values for beta");auto diff=at::abs(x.sub(target));auto value=beta==0?diff:at::where(diff.lt(beta),diff.mul(diff).mul(.5/beta),diff.sub(.5*beta));return reduce_loss(value,reduction);}
at::Tensor smooth_grad(const at::Tensor& grad,const at::Tensor& x,const at::Tensor& target,int64_t reduction,double beta){loss_inputs(x,target);TORCH_CHECK(beta>=0,"smooth_l1_loss does not support negative values for beta");auto diff=x.sub(target),value=beta==0?at::sign(diff):at::where(at::abs(diff).lt(beta),diff.div(beta),at::sign(diff));if(reduction==1)value=value.div(value.numel());return value.mul(grad);}
at::Tensor huber_loss(const at::Tensor& x,const at::Tensor& target,int64_t reduction,double delta){loss_inputs(x,target);TORCH_CHECK(delta>0,"huber_loss does not support non-positive values for delta.");auto diff=at::abs(x.sub(target));auto value=at::where(diff.lt(delta),diff.mul(diff).mul(.5),diff.sub(.5*delta).mul(delta));return reduce_loss(value,reduction);}
at::Tensor huber_grad(const at::Tensor& grad,const at::Tensor& x,const at::Tensor& target,int64_t reduction,double delta){loss_inputs(x,target);TORCH_CHECK(delta>0,"huber_loss does not support non-positive values for delta.");auto diff=x.sub(target),value=at::where(at::abs(diff).lt(delta),diff,at::sign(diff).mul(delta));if(reduction==1)value=value.div(value.numel());return value.mul(grad);}
at::Tensor bce_loss(const at::Tensor& x,const at::Tensor& target,const std::optional<at::Tensor>& weight,int64_t reduction){loss_inputs(x,target);TORCH_CHECK(x.sizes()==target.sizes(),"binary_cross_entropy input and target must have the same shape");auto valid=at::logical_and(at::logical_and(x.ge(0),x.le(1)),at::logical_and(target.ge(0),target.le(1)));TORCH_CHECK(at::all(valid).cpu().item<bool>(),"all elements of input and target should be between 0 and 1");auto value=target.mul(at::log(x).clamp_min(-100)).add(at::rsub(target,1).mul(at::log1p(x.neg()).clamp_min(-100))).neg();if(weight&&weight->defined())value=value.mul(*weight);return reduce_loss(value,reduction);}
at::Tensor bce_grad(const at::Tensor& grad,const at::Tensor& x,const at::Tensor& target,const std::optional<at::Tensor>& weight,int64_t reduction){loss_inputs(x,target);auto value=x.sub(target).div(x.mul(at::rsub(x,1)).clamp_min(1e-12));if(weight&&weight->defined())value=value.mul(*weight);if(reduction==1)value=value.div(x.numel());return value.mul(grad);}
}
TORCH_LIBRARY_IMPL(aten,PrivateUse1,m){
 m.impl("smooth_l1_loss",&smooth_loss);m.impl("smooth_l1_loss_backward",&smooth_grad);m.impl("huber_loss",&huber_loss);m.impl("huber_loss_backward",&huber_grad);m.impl("binary_cross_entropy",&bce_loss);m.impl("binary_cross_entropy_backward",&bce_grad);
}
}
