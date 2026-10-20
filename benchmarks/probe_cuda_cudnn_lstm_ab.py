"""Compare cuDNN sequence fusion vs native CUDA recurrent decomposition.

Matched two-layer bidirectional LSTM, identical weights and inputs, no
gradient or optimizer (isolates recurrent eval forward). Runs the RTX 2080 Ti
only and checks the output before computing synchronized timing samples.
"""
import argparse,json,statistics,time,torch
from pathlib import Path
p=argparse.ArgumentParser()
p.add_argument('--out',type=Path,required=True)
p.add_argument('--warmup',type=int,default=15)
p.add_argument('--repeats',type=int,default=80)
a=p.parse_args()
torch.manual_seed(42);torch.set_num_threads(4)
assert torch.cuda.get_device_name(0)=='NVIDIA GeForce RTX 2080 Ti'
model=torch.nn.LSTM(5,7,2,bidirectional=True,batch_first=True).cuda().eval()
x=torch.randn(2,4,5,device='cuda')
reference=None;data=[]
for flag in (True,False,True,False):
 torch.backends.cudnn.enabled=flag
 with torch.no_grad():
  output=model(x)[0]
  torch.cuda.synchronize()
  if reference is None:reference=output.detach().clone()
  torch.testing.assert_close(output,reference,rtol=2e-4,atol=2e-5)
  for _ in range(a.warmup):model(x)
  torch.cuda.synchronize()
  samples=[]
  for _ in range(a.repeats):
   torch.cuda.synchronize();ts=time.perf_counter()
   model(x);torch.cuda.synchronize()
   samples.append((time.perf_counter()-ts)*1000)
  data.append(dict(cudnn_enabled=flag,median_ms=statistics.median(samples),warmup=a.warmup,
   repeats=a.repeats,p10_ms=sorted(samples)[int(.1*(len(samples)-1))],
   p90_ms=sorted(samples)[int(.9*(len(samples)-1))],
   samples_ms=samples,cpu_fallback=False,
   forward_parity_max_abs=float((output-reference).abs().max())))
  print('RESULT',flag,round(data[-1]['median_ms'],4),flush=True)
torch.backends.cudnn.enabled=True
a.out.parent.mkdir(parents=True,exist_ok=True)
a.out.write_text(json.dumps({'model':'torch.nn.LSTM(5,7,2,bidirectional=True,batch_first=True)','input_shape':[2,4,5],
 'gpu':torch.cuda.get_device_name(0),'tests':data},indent=2)+'\n')
