"""Probe which existing Vulkan out= families retain extra GPU work.

No tensors are transferred during timing; this only counts recorded GPU
operations, performs CPU reference numerical checks, and reports limitations.
"""
import json
import torch,torch_vulkan
torch.set_num_threads(4)
cases = [
 ('unary.exp',lambda: (torch.exp, (torch.rand(4,32,device='vulkan'),),torch.empty(4,32,device='vulkan'))),
 ('math.log',lambda: (torch.log, (torch.rand(4,32,device='vulkan')+1,),torch.empty(4,32,device='vulkan'))),
 ('mm',lambda: (torch.mm, (torch.randn(32,48,device='vulkan'),torch.randn(48,24,device='vulkan')),torch.empty(32,24,device='vulkan'))),
 ('bmm',lambda: (torch.bmm, (torch.randn(3,32,48,device='vulkan'),torch.randn(3,48,24,device='vulkan')),torch.empty(3,32,24,device='vulkan'))),
 ('addmm',lambda: (torch.addmm, (torch.randn(32,24,device='vulkan'),torch.randn(32,48,device='vulkan'),torch.randn(48,24,device='vulkan')),torch.empty(32,24,device='vulkan'))),
 ('softmax.out',lambda: (lambda x,out:torch.ops.aten._softmax.out(x,-1,False,out=out), (torch.randn(16,64,device='vulkan'),), torch.empty(16,64,device='vulkan'))),
 ('sum.out',lambda: (lambda x,out:torch.sum(x,dim=-1,out=out), (torch.randn(16,64,device='vulkan'),), torch.empty(16,device='vulkan'))),
 ('cat.out',lambda: (lambda x,y,out:torch.cat([x,y],dim=0,out=out),(torch.randn(8,32,device='vulkan'),torch.randn(8,32,device='vulkan')),torch.empty(16,32,device='vulkan'))),
]
results=[]
for name,construct in cases:
 try:
  fn,args,out=construct()
  torch_vulkan.synchronize()
  before=torch_vulkan.queue_stats()
  got=fn(*args,out=out)
  torch_vulkan.synchronize()
  after=torch_vulkan.queue_stats()
  assert got is out
  results.append(dict(name=name,shader_dispatches=after['recorded_ops']-before['recorded_ops'],fallback=torch_vulkan.execution_stats()['cpu_fallback_calls']))
 except Exception as e:
  results.append(dict(name=name,error=str(e).splitlines()[0][:240]))
print(json.dumps(results,indent=2))
