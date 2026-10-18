#include "native_utils.h"
#include <torch/library.h>
namespace torch_vulkan::ops {
namespace {
using namespace native;
void check(const at::Tensor& x,const at::Tensor& ref){TORCH_CHECK(x.device()==ref.device()&&x.scalar_type()==at::kFloat&&can_pack_view(x),"Vulkan LSTM requires packable FP32 tensors on one device");}
Buffer optional_buffer(const at::Tensor& x){return x.defined()&&x.numel()?base(x):VulkanAllocator::instance().constant_float_buffer(1,0);}
std::tuple<at::Tensor,at::Tensor,at::Tensor> lstm(const at::Tensor& input,const at::Tensor& hidden,const at::Tensor& cx,
 const std::optional<at::Tensor>& ib,const std::optional<at::Tensor>& hb){
 check(cx,cx);check(input,cx);check(hidden,cx);
 TORCH_CHECK(cx.dim()==2&&input.dim()==2&&input.sizes()==hidden.sizes()&&input.size(0)==cx.size(0)&&input.size(1)==4*cx.size(1),"LSTM gate shape mismatch");
 auto h=at::empty(cx.sizes(),cx.options()),c=at::empty_like(h),ws=at::empty(input.sizes(),input.options());
 at::Tensor bi,bh;
 if(ib&&ib->defined()){check(*ib,cx);TORCH_CHECK(ib->dim()==1&&ib->numel()==input.size(1),"LSTM input bias shape mismatch");bi=packed(*ib);}
 if(hb&&hb->defined()){check(*hb,cx);TORCH_CHECK(hb->dim()==1&&hb->numel()==input.size(1),"LSTM hidden bias shape mismatch");bh=packed(*hb);}
 if(h.numel())run("lstm_cell",{base(packed(input)),base(packed(hidden)),base(packed(cx)),optional_buffer(bi),optional_buffer(bh)},
  {base(h),base(c),base(ws)},{u32(cx.numel()),u32(cx.size(1)),uint32_t(bi.defined()),uint32_t(bh.defined())},(h.numel()+127)/128);
 return {h,c,ws};
}
std::tuple<at::Tensor,at::Tensor,at::Tensor> lstm_backward(const std::optional<at::Tensor>& gh,const std::optional<at::Tensor>& gc,
 const at::Tensor& cx,const at::Tensor& cy,const at::Tensor& ws,bool has_bias){
 check(cx,cx);check(cy,cx);check(ws,cx);TORCH_CHECK(cx.dim()==2&&cx.sizes()==cy.sizes()&&ws.dim()==2&&ws.size(0)==cx.size(0)&&ws.size(1)==4*cx.size(1),"LSTM backward shape mismatch");
 at::Tensor h,c;
 if(gh&&gh->defined()){check(*gh,cx);TORCH_CHECK(gh->sizes()==cx.sizes(),"LSTM hidden gradient shape mismatch");h=packed(*gh);}
 if(gc&&gc->defined()){check(*gc,cx);TORCH_CHECK(gc->sizes()==cx.sizes(),"LSTM cell gradient shape mismatch");c=packed(*gc);}
 auto gates=at::empty(ws.sizes(),ws.options()),dx=at::empty_like(cx);
 if(cx.numel())run("lstm_cell_backward",{optional_buffer(h),optional_buffer(c),base(packed(cx)),base(packed(cy)),base(packed(ws))},
  {base(gates),base(dx)},{u32(cx.numel()),u32(cx.size(1)),uint32_t(h.defined()),uint32_t(c.defined())},(cx.numel()+127)/128);
 at::Tensor db;if(has_bias)db=gates.sum(0);
 return {gates,dx,db};
}
}
TORCH_LIBRARY_IMPL(aten,PrivateUse1,m){
 m.impl("_thnn_fused_lstm_cell",&lstm);
 m.impl("_thnn_fused_lstm_cell_backward_impl",&lstm_backward);
}
}
