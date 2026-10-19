import sys,torch
sys.path.append('/workspace/Pytorch-vulkan/.venv/lib/python3.12/site-packages')
from torch.testing._internal import composite_compliance as cc
from torch.testing._internal.common_utils import TestCase
from torch.utils._pytree import tree_map
blob=torch.load('/tmp/pv-masked-prod-repro.pt',weights_only=False)
for device in ['cpu','cuda']:
 args=tree_map(lambda t:t.detach().to(device).requires_grad_(t.requires_grad) if isinstance(t,torch.Tensor) else t,blob['args']);kwargs=tree_map(lambda t:t.to(device) if isinstance(t,torch.Tensor) else t,blob['kwargs'])
 values=iter(blob['tangents']);original=torch.randn_like
 torch.randn_like=lambda t:next(values).to(device)
 try:
  cc.check_forward_ad_formula(torch.masked.prod,args,kwargs,assert_equal_fn=TestCase().assertEqual);print(device,'PASS')
 except Exception as e:print(device,type(e).__name__,str(e))
 finally:torch.randn_like=original
