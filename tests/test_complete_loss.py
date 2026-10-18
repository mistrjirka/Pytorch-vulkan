import pytest
import torch
import torch.nn.functional as F
import torch_vulkan

@pytest.mark.parametrize('reduction',['none','mean','sum'])
@pytest.mark.parametrize('shape',[(3,5),(2,3,4), (0,5)])
def test_mse_loss_and_both_gradients(shape,reduction):
    x=torch.randn(shape).transpose(0,1).requires_grad_();y=torch.randn_like(x).requires_grad_()
    a,b=[t.detach().to('vulkan').requires_grad_() for t in (x,y)]
    expected=F.mse_loss(x,y,reduction=reduction);g=torch.randn_like(expected).to('vulkan')
    torch_vulkan.reset_execution_stats();actual=F.mse_loss(a,b,reduction=reduction);actual.backward(g);stats=torch_vulkan.execution_stats()
    expected.backward(g.cpu());assert stats['cpu_fallback_calls']==0 and stats['host_read_max_bytes']==0,stats
    torch.testing.assert_close(actual.cpu(),expected,equal_nan=True)
    for p,q in ((a,x),(b,y)):torch.testing.assert_close(p.grad.cpu(),q.grad)


def test_mse_out_views_and_broadcast():
    x=torch.randn(3,5);target=torch.randn(1,5);a,b=x.to('vulkan'),target.to('vulkan')
    parent=torch.full((7,11),17.).to('vulkan');out=parent[1:7:2,1:11:2]
    torch.ops.aten.mse_loss.out(a,b,0,out=out)
    expected=torch.full((7,11),17.);expected[1:7:2,1:11:2]=(x-target).square()
    torch.testing.assert_close(parent.cpu(),expected)
