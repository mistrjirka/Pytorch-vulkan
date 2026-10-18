"""Host/device coherence and aliasing under eager or deferred Vulkan execution.

Run with optional TORCH_VULKAN_GPU_RESIDENT=1 and
TORCH_VULKAN_DEFER_SUBMISSIONS=1 to exercise the additional code paths.
"""
import torch
import torch_vulkan


def test_gpu_chain_reads_once_at_boundary():
    torch.manual_seed(153)
    x_cpu=torch.randn(96,96)
    w_cpu=torch.randn(96,96)
    x=x_cpu.to('vulkan')
    w=w_cpu.to('vulkan')
    cpu=x_cpu
    scale_cpu=torch.full((96,96),0.05)
    scale_vk=scale_cpu.to('vulkan')
    for _ in range(12):
        cpu=torch.relu(cpu @ w_cpu)*scale_cpu
        x=torch.relu(x @ w)*scale_vk
    torch.testing.assert_close(x.cpu(),cpu,rtol=3e-4,atol=2e-4)


def test_interleaved_host_write_and_gpu_readback():
    torch.manual_seed(451)
    cpu=torch.randn(11,23)
    gpu=cpu.to('vulkan')
    offset_cpu=torch.full((11,23),0.7)
    offset_vk=offset_cpu.to('vulkan')
    for i in range(5):
        cpu=torch.relu(cpu+offset_cpu)
        gpu=torch.relu(gpu+offset_vk)
        torch.testing.assert_close(gpu.cpu(),cpu,rtol=1e-5,atol=1e-5)
        # Overwrite part of the Vulkan tensor from CPU; subsequent Vulkan ops
        # must upload the new host data rather than reuse stale GPU contents.
        cpu[2:4]=torch.randn(2,23)
        copied=gpu.cpu()
        copied[2:4]=cpu[2:4]
        gpu.copy_(copied.to('vulkan'))
        torch.testing.assert_close(gpu.cpu(),cpu,rtol=1e-6,atol=1e-6)


def test_vulkan_output_view_after_gpu_write():
    torch.manual_seed(601)
    a=torch.randn(32,16)
    b=torch.randn(32,16)
    gpu=(a.to('vulkan')+b.to('vulkan')).view(4,8,16)
    torch.testing.assert_close(gpu.transpose(0,1).cpu(),
                               (a+b).view(4,8,16).transpose(0,1),
                               rtol=1e-6,atol=1e-6)


def test_deferred_graph_reduces_submissions():
    import os
    if os.environ.get('TORCH_VULKAN_DEFER_SUBMISSIONS')!='1':
        return
    if os.environ.get('TORCH_VULKAN_GPU_RESIDENT')!='1':
        return
    x=torch.zeros(4096,device='vulkan')
    add=torch.full((4096,),0.01,device='vulkan')
    torch_vulkan.synchronize()
    prev=torch_vulkan.queue_stats()['submissions']
    for _ in range(60):
        x=torch.relu(x+add)
    torch_vulkan.synchronize()
    cur=torch_vulkan.queue_stats()['submissions']
    assert cur-prev==1,(prev,cur)
    torch.testing.assert_close(x.cpu(),torch.full((4096,),0.6),
                               rtol=1e-4,atol=1e-4)
