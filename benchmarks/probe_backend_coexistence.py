"""Compare CUDA coexistence with/without a registered PrivateUse1 backend.

Set PV_IMPORT_VULKAN=0 to test CUDA alone and PV_RESULT to a JSON output path.
Select a baseline or current backend using PYTHONPATH in a fresh process.
"""
import os,json,traceback
from pathlib import Path
import torch
result={'torch':torch.__version__}
try:
 if os.environ.get('PV_IMPORT_VULKAN','1')=='1':
  import torch_vulkan
  result['vulkan_module']=torch_vulkan.__file__
 result['selected_accelerator']=str(torch.accelerator.current_accelerator())
 x=torch.randn(7,9,device='cuda').requires_grad_()
 (x.sin().square().mean()).backward();result['eager_cuda']='ok';x.grad=None
 fn=torch.compile(lambda x:x.sin().square().mean(),fullgraph=True)
 fn(x).backward();result['compiled_cuda']='ok'
except Exception as e:result.update(error_type=type(e).__name__,error=str(e),status='error')
else:result['status']='ok'
Path(os.environ['PV_RESULT']).write_text(json.dumps(result,indent=2)+'\n');print(result)
