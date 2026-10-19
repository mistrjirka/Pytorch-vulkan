import importlib.util,sys,torch,unittest
import torch_vulkan
from torch.testing._internal.common_device_type import PrivateUse1TestBase
from torch.testing._internal import composite_compliance as cc
from torch.utils._pytree import tree_map
PrivateUse1TestBase.bypass_device_restrictions=True
original_check=cc.check_forward_ad_formula
original_randn=torch.randn_like
captured=0

def cpu(t):
 if not isinstance(t,torch.Tensor):return t
 return t.detach().cpu().requires_grad_(t.requires_grad)

def check(op,args,kwargs,*other):
 global captured
 tangents=[]
 def randn(*a,**kw):
  t=original_randn(*a,**kw);tangents.append(t.cpu());return t
 torch.randn_like=randn
 try:return original_check(op,args,kwargs,*other)
 except Exception:
  torch.save({'args':tree_map(cpu,args),'kwargs':tree_map(cpu,kwargs),'tangents':tangents},'/tmp/pv-masked-prod-repro.pt');captured+=1;raise
 finally:torch.randn_like=original_randn
cc.check_forward_ad_formula=check
spec=importlib.util.spec_from_file_location('upstream_test_ops','/workspace/pytorch-opinfo-5c488690/test/test_ops.py');module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
case=module.TestCompositeComplianceVULKAN('test_forward_ad_masked_prod_vulkan_float32');result=unittest.TestResult();case.run(result);print('captured',captured,'errors',len(result.errors),'failures',len(result.failures));print(result.errors[-1][1][-1800:] if result.errors else '')
