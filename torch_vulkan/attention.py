"""General FP32 fused attention with a device-resident training backward."""
import math
import torch

class _Attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx,q,k,v,mask,dropout,causal,scale):
        out,lse,state=torch.ops.torch_vulkan.attention_forward(q,k,v,mask,dropout,causal,scale)
        ctx.save_for_backward(q,k,v,mask,out,lse,state)
        ctx.config=(dropout,causal,scale)
        return out

    @staticmethod
    def backward(ctx,grad):
        q,k,v,mask,out,lse,state=ctx.saved_tensors
        dq,dk,dv,dm=torch.ops.torch_vulkan.attention_backward(grad,q,k,v,mask,out,lse,*ctx.config,state)
        return dq,dk,dv,dm if mask.requires_grad else None,None,None,None

def scaled_dot_product_attention(query,key,value,attn_mask=None,dropout_p=0.,is_causal=False,*,scale=None,enable_gqa=False):
    if any(x.dtype!=torch.float32 or x.ndim<2 for x in (query,key,value)):
        raise RuntimeError('Vulkan attention currently requires FP32 tensors with rank >=2')
    if query.shape[-1]!=key.shape[-1] or key.shape[-2]!=value.shape[-2] or query.shape[-1]==0:
        raise RuntimeError('Attention feature or sequence dimensions mismatch')
    if enable_gqa:
        if min(x.ndim for x in (query,key,value))<3 or key.shape[-3]!=value.shape[-3] or query.shape[-3]%key.shape[-3]:
            raise RuntimeError('Invalid grouped-query attention head counts')
        factor=query.shape[-3]//key.shape[-3]
        def repeat_heads(x):
            return x.unsqueeze(-3).expand(*x.shape[:-2],factor,*x.shape[-2:]).reshape(*x.shape[:-3],query.shape[-3],*x.shape[-2:])
        key,value=repeat_heads(key),repeat_heads(value)
    leading=torch.broadcast_shapes(query.shape[:-2],key.shape[:-2],value.shape[:-2])
    q,k,v=[x.expand(*leading,*x.shape[-2:]).reshape(math.prod(leading),*x.shape[-2:]) for x in (query,key,value)]
    if attn_mask is None:
        mask=query.new_empty((0,))
    else:
        if attn_mask.dtype not in (torch.bool,torch.float32):
            raise RuntimeError('Attention mask must be bool or FP32')
        mask=attn_mask.expand(*leading,q.shape[-2],k.shape[-2]).reshape(q.shape[0],q.shape[-2],k.shape[-2])
    out=_Attention.apply(q,k,v,mask,dropout_p,is_causal,scale if scale is not None else 1/math.sqrt(query.shape[-1]))
    return out.reshape(*leading,q.shape[-2],v.shape[-1])

# Override the composite entry at its autograd key so saved tensors and gradients
# belong to the fused operator rather than the math decomposition.
_autograd=torch.library.Library('aten','IMPL','AutogradPrivateUse1')
_autograd.impl('scaled_dot_product_attention',scaled_dot_product_attention)
_backend=torch.library.Library('aten','IMPL','PrivateUse1')
_backend.impl('scaled_dot_product_attention',scaled_dot_product_attention)


@torch.library.register_fake('torch_vulkan::attention_forward')
def _fake_forward(q,k,v,mask,dropout,causal,scale):
    return q.new_empty((q.shape[0],q.shape[1],v.shape[2])),q.new_empty(q.shape[:2]),torch.empty(4,dtype=torch.int64,device='cpu')


@torch.library.register_fake('torch_vulkan::attention_backward')
def _fake_backward(grad,q,k,v,mask,out,lse,dropout,causal,scale,state):
    return q.new_empty(q.shape),k.new_empty(k.shape),v.new_empty(v.shape),q.new_empty((q.shape[0],q.shape[1],k.shape[1]))
