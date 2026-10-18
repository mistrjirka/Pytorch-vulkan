import torch,json,sys
b=sys.argv[1]
if b=='vulkan':import torch_vulkan
r={'backend':b}
try:
 e=torch.Event(device=b);e.record();e.synchronize();r.update(status='ok',record_and_synchronize=True)
except Exception as ex:r.update(status='error',error=str(ex),error_type=type(ex).__name__)
print(json.dumps(r))
