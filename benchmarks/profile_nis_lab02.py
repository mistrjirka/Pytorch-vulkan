"""Stage-resolved FP32 MyNet profile: same model, checkpoint, seeded shapes.

No frame/video decoding inside timed measurements. No full stylization script.
CUDA/ROCm: torch.profiler GPU kernel trace + CUDA event stage timing.
Vulkan: shader timestamp accumulation + CPU operator trace.
"""
import argparse, json, os, sys, statistics, time
from collections import defaultdict, Counter
from pathlib import Path
import torch

parser=argparse.ArgumentParser()
parser.add_argument('--backend',required=True, choices=['cuda','rocm','vulkan'])
parser.add_argument('--mode',required=True, choices=['train','frame'])
parser.add_argument('--n',type=int,default=100)
parser.add_argument('--optimizer',choices=['adam','adamw','native_adamw'],default='adam')
parser.add_argument('--foreach',action='store_true')
parser.add_argument('--fused',action='store_true',help='Request PyTorch-native fused=True optimizer')
parser.add_argument('--warmup',type=int,default=12)
parser.add_argument('--profile-steps',type=int,default=3)
parser.add_argument('--output',required=True)
parser.add_argument('--checkpoint',default='none')
a=parser.parse_args()
torch.set_num_threads(4)
torch.manual_seed(20261019)
# Retain IEEE FP32 convolution arithmetic on Ampere/Volta/Turing.
torch.backends.cudnn.allow_tf32=False
torch.backends.cuda.matmul.allow_tf32=False
orig=Path(os.environ.get('NIS_LAB02_SOURCE',
    str(Path.home()/'programovani/magistr/nis/lab02_cnn')))
if not (orig/'models.py').is_file():
 orig=Path(__file__).resolve().parent/'nis-lab02-reference'
if not (orig/'models.py').is_file():
 raise FileNotFoundError('Set NIS_LAB02_SOURCE to a local directory containing Lab02 models.py')
sys.path.insert(0,str(orig))
from models import MyNet

if a.backend=='vulkan':
 import torch_vulkan
 dev=torch.device('vulkan')
 synchronize=torch_vulkan.synchronize
 device_name=torch_vulkan.device_name()
 assert torch_vulkan.is_hardware_accelerated(),device_name
else:
 dev=torch.device('cuda')
 synchronize=torch.cuda.synchronize
 device_name=torch.cuda.get_device_name(0)
 assert (torch.version.hip is not None)==(a.backend=='rocm'),(torch.version.hip,a.backend)

model=MyNet(3,3,5)
if a.checkpoint != "none":
 model.load_state_dict(torch.load(a.checkpoint,map_location="cpu",weights_only=True))
import hashlib
model_hash=hashlib.sha256(b''.join(t.detach().cpu().contiguous().numpy().tobytes()
    for t in model.state_dict().values())).hexdigest()
model=model.to(device=dev,dtype=torch.float32)
model.train(a.mode=='train')
generator=torch.Generator().manual_seed(7102026)
if a.mode=='train':
 # Matches the original Lab02 input tensor geometry for batch 1.
 x_cpu=torch.randn(1,3,128,128,generator=generator,dtype=torch.float32)
 y_cpu=torch.randn(1,3,128,128,generator=generator,dtype=torch.float32)
 x=x_cpu.to(dev)
 y=y_cpu.to(dev)
 if a.optimizer=='adam':
  opt=torch.optim.Adam(model.parameters(),lr=1e-4,weight_decay=1e-5,foreach=(None if a.fused else a.foreach),fused=(True if a.fused else None))
 elif a.optimizer=='adamw':
  opt=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=1e-5,foreach=(None if a.fused else a.foreach),fused=(True if a.fused else None))
 else:
  if a.backend!='vulkan':raise RuntimeError('native_adamw requires Vulkan')
  from torch_vulkan.optim import AdamW as NativeAdamW
  opt=NativeAdamW(model.parameters(),lr=1e-4,weight_decay=1e-5)

else:
 # Synthetic same-shape input: this host may not have the source PNGs.
 x_cpu=torch.randn(1,3,832,720,generator=generator,dtype=torch.float32)
 x=x_cpu.to(dev)
 y=None;opt=None
synchronize()
input_hash=hashlib.sha256(x_cpu.numpy().tobytes()).hexdigest()

def step(stage_timers=None):
 if opt:opt.zero_grad(set_to_none=True)
 def record(name,fn):
  if stage_timers is not None:
   synchronize()
   ts=time.perf_counter()
  with torch.profiler.record_function('NIS_'+name):
   val=fn()
  if stage_timers is not None:
   synchronize()
   stage_timers[name]=(time.perf_counter()-ts)*1000
  return val
 p=record('forward',lambda: (torch.tanh(model(x))+1)*.5)
 if opt:
  loss=record('loss',lambda: (p-y).square().mean())
  record('backward',loss.backward)
  record('adam',opt.step)
 return p

r=dict(status='running',hostname=os.uname().nodename,gpu=device_name,
       backend=a.backend,torch=torch.__version__,hip=torch.version.hip,
       cuda=torch.version.cuda,mode=a.mode,shape=list(x.shape),optimizer=a.optimizer,
       parameter_tensors=sum(1 for t in model.parameters() if t.requires_grad),foreach=a.foreach,fused=a.fused,
       warmup=a.warmup,steps=a.n,profiling_steps=a.profile_steps,
       checkpoint=str(a.checkpoint),loss='MSE (training microbenchmark)',
       model_sha256=model_hash,input_sha256=input_hash,
       data='fixed seeded synthetic FP32 inputs (not actual video frame)',
       cudnn_tf32=torch.backends.cudnn.allow_tf32,
       times_ms=[],phase_wall_ms=[],device_stage_ms=[],cuda_kernels=[],vulkan_shaders=[])
for i in range(a.warmup):
 step()
synchronize()

# Independent timing, no profiler or stage synchronization in the timed loop.
for i in range(a.n):
 t=time.perf_counter()
 p=step()
 synchronize()
 r['times_ms'].append((time.perf_counter()-t)*1000)
 if i%25==0:
  print('SAMPLE',i,round(r['times_ms'][-1],3),flush=True)

if a.backend=='vulkan':
 r['execution']=torch_vulkan.execution_stats()
 r['cache']=torch_vulkan.cache_stats()
# Stage-synchronized diagnostic timing, separate from end-to-end timing.
for i in range(a.profile_steps):
 stage={}
 if a.backend!='vulkan':
  events={}
  # Events on the same CUDA/HIP stream encompass the respective launches.
  names=['forward','loss','backward','adam'] if opt else ['forward']
  stamps=[torch.cuda.Event(enable_timing=True) for _ in range(len(names)+1)]
  stamps[0].record()
  if opt:opt.zero_grad(set_to_none=True)
  p=(torch.tanh(model(x))+1)*.5
  stamps[1].record()
  if opt:
   loss=(p-y).square().mean();stamps[2].record()
   loss.backward();stamps[3].record()
   opt.step();stamps[4].record()
  synchronize()
  r['device_stage_ms'].append({name:stamps[j].elapsed_time(stamps[j+1])
                                for j,name in enumerate(names)})
 step(stage)

 r['phase_wall_ms'].append(stage)

if a.backend=='vulkan':
 import torch_vulkan
 # instrumented Vulkan runs require TORCH_VULKAN_PROFILE_TIMESTAMPS=1
 try:
  before=torch_vulkan.gpu_timing_stats()
  for _ in range(a.profile_steps):step();synchronize()
  after=torch_vulkan.gpu_timing_stats()
  def minus(b,c,key):
   return float(c.get(key,0))-float(b.get(key,0))
  for k,v in after.items():
   b=before.get(k,{})
   count=minus(b,v,'samples')
   if count<=0:continue
   r['vulkan_shaders'].append(dict(name=k,shader_ms_total=minus(b,v,'shader_ms'),
    calls=count,upload_ms=minus(b,v,'upload_ms'),download_ms=minus(b,v,'download_ms'),
    host_eval_ms=minus(b,v,'host_eval_ms')))
 except Exception as exc:r['gpu_timing_error']=str(exc)

# A few profiled steps, not included in timing distributions.
try:
 from torch.profiler import profile,ProfilerActivity,record_function
 activities=[ProfilerActivity.CPU]
 if a.backend!='vulkan':activities.append(ProfilerActivity.CUDA)
 with profile(activities=activities,record_shapes=True) as prof:
  for _ in range(a.profile_steps):step()
  synchronize()
 grouped=prof.key_averages(group_by_input_shape=False)
 r['operators']=sorted([dict(name=e.key,calls=e.count,self_cpu_ms=e.self_cpu_time_total/1000,
  device_ms=getattr(e,'self_device_time_total',0)/1000) for e in grouped],
  key=lambda z:z['self_cpu_ms'],reverse=True)[:70]
 if a.backend!='vulkan':
  kernels=defaultdict(lambda:[0,0])
  for e in prof.events():
   if str(getattr(e,'device_type','')).upper().endswith('CUDA'):
    ms=getattr(e,'self_device_time_total',0)/1000
    if ms>0:
     kernels[e.name][0]+=ms;kernels[e.name][1]+=1
  r['cuda_kernels']=sorted([dict(name=k,gpu_ms=v[0],count=v[1])
         for k,v in kernels.items()],key=lambda x:x['gpu_ms'],reverse=True)
except Exception as exc:
 r['profiler_error']=str(exc)

ordered=sorted(r['times_ms'])
def pct(f):
 pos=(len(ordered)-1)*f;l=int(pos);hi=min(l+1,len(ordered)-1)
 return ordered[l]+(ordered[hi]-ordered[l])*(pos-l)
r.update(status='passed',median_ms=statistics.median(ordered),p10_ms=pct(.1),
 p90_ms=pct(.9),mean_ms=statistics.mean(ordered))
out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
out.write_text(json.dumps(r,indent=2,default=str)+'\n')
print('RESULT',a.backend,a.mode,round(r['median_ms'],3),'ms',out,flush=True)
print('PHASE_WALL',r['phase_wall_ms'],flush=True)
print('DEVICE_STAGES',r['device_stage_ms'],flush=True)
print('KERNELS',r['cuda_kernels'][:8],flush=True)
print('SHADERS',sorted(r['vulkan_shaders'],key=lambda x:x['shader_ms_total'],reverse=True)[:8],flush=True)
