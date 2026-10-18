"""No generic CPU fallback or large host materialization in requested workloads."""
import pytest
import torch
import torch.nn.functional as F
import torch_vulkan

@pytest.mark.parametrize('kind',['conv3d','transpose','lstm','dropout','embedding','index','sort','fft','solve','attention','compile'])
def test_device_residency(kind):
    torch.manual_seed(73)
    def tensor(*shape):return torch.randn(shape,device='vulkan').requires_grad_()
    allowed_status_bytes=0
    if kind=='conv3d':
        x=tensor(2,4,6,5,4);w=tensor(6,2,3,2,2);fn=lambda:F.conv3d(x,w,padding=1,groups=2)
    elif kind=='transpose':
        x=tensor(2,4,3,4);w=tensor(4,3,3,2);fn=lambda:F.conv_transpose2d(x,w,stride=2,padding=1)
    elif kind=='lstm':
        model=torch.nn.LSTM(5,7,2,bidirectional=True,dropout=.2).to('vulkan');x=tensor(4,2,5);fn=lambda:model(x)[0]
    elif kind=='dropout':
        x=tensor(10,11);fn=lambda:F.dropout(x,.3,True)
    elif kind=='embedding':
        x=tensor(19,7);idx=torch.tensor([1,2,2,7,0,7,12],device='vulkan');fn=lambda:F.embedding(idx,x,padding_idx=0,scale_grad_by_freq=True);allowed_status_bytes=4
    elif kind=='index':
        x=tensor(7,9);idx=torch.tensor([[1,2,3],[3,2,1],[0,1,0],[5,3,4],[1,2,3],[3,2,1],[2,3,4]],device='vulkan');fn=lambda:torch.gather(x,1,idx);allowed_status_bytes=4
    elif kind=='sort':
        x=tensor(7,9);fn=lambda:torch.sort(x,dim=1).values;allowed_status_bytes=4
    elif kind=='fft':
        x=tensor(3,16);fn=lambda:torch.fft.irfft(torch.fft.rfft(x),n=16)
    elif kind=='solve':
        x=(torch.eye(5)*3+torch.randn(5,5)*.1).to('vulkan').requires_grad_();b=tensor(5,3);fn=lambda:torch.linalg.solve(x,b);allowed_status_bytes=4
    elif kind=='attention':
        q,k,v=[tensor(2,3,7,5) for _ in range(3)];fn=lambda:F.scaled_dot_product_attention(q,k,v,is_causal=True,dropout_p=.2)
    else:
        x=tensor(7,9);w=tensor(9,5)
        compiled=torch.compile(lambda a,b:(a.sin()@b).relu(),fullgraph=True)
        fn=lambda:compiled(x,w)
        # Compile before measuring execution. Compiler constant evaluation is
        # separate from execution of user tensors and must not hide host reads.
        fn().sum().backward();x.grad=None;w.grad=None
    torch_vulkan.reset_execution_stats()
    out=fn();out.sum().backward()
    stats=torch_vulkan.execution_stats()
    assert stats['cpu_fallback_calls']==0,stats
    assert stats['host_read_max_bytes']<=allowed_status_bytes,stats
