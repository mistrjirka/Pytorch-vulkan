"""Observe actual stock-PyTorch forward/backward operators, not just module names."""
import argparse,collections,gc,json,sys
from pathlib import Path
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from feature_networks import CASES,make_case
from compare_cuda_vulkan import CONFIG,data_for_case

class Record(TorchDispatchMode):
    def __init__(self):self.counts=collections.Counter()
    def __torch_dispatch__(self,func,types,args=(),kwargs=None):
        self.counts[str(func)]+=1
        return func(*args,**(kwargs or {}))

FEATURES={
 'average_pool':['avg_pool'], 'bilinear_resize':['upsample_bilinear'],
 'bicubic_resize':['upsample_bicubic'], 'axis_mean':['mean.dim'],
 'sum':['sum.'], 'dropout_rng':['dropout','bernoulli','normal_'],
 'attention':['scaled_dot_product','bmm.'],
 'mask_math':['where.','isneginf.','all.','clamp.','pow.','gt.','ge.','le.'],
}

def main():
    p=argparse.ArgumentParser();p.add_argument('--zoo',type=Path,required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(4);sys.path.insert(0,str(a.zoo/'benchmarks'));import model_zoo
    report={}
    for name in list(model_zoo.MODELS)+list(CASES):
        try:
            if name in CASES:model,x=make_case(name);loss_fn=lambda y:y.square().mean()
            else:
                torch.manual_seed(42);model,_,loss=model_zoo.MODELS[name]()
                if name in CONFIG:x,label=data_for_case(name,CONFIG[name]['tiny'],torch.Generator().manual_seed(123))
                elif name=='resnet18_224':x,label=data_for_case('resnet18',(2,3,32,32),torch.Generator().manual_seed(123))
                elif name=='lstm':x=torch.randn(2,8,64);label=torch.zeros(2,dtype=torch.long)
                elif name=='conv1d_audio':x=torch.randn(2,1,512);label=torch.zeros(2,dtype=torch.long)
                elif name=='conv_autoencoder':x=torch.randn(1,3,32,32);label=x.clone()
                loss_fn=lambda y:loss(y,label)
            model.train();record=Record()
            with record:loss_fn(model(x)).backward()
            features={feature:[op for op in record.counts if any(token in op for token in tokens)] for feature,tokens in FEATURES.items()}
            transpose=any(isinstance(m,(torch.nn.ConvTranspose1d,torch.nn.ConvTranspose2d,torch.nn.ConvTranspose3d)) for m in model.modules())
            report[name]={'status':'ok','input_shape':list(x.shape),'features':features,'has_transposed_convolution':transpose,'operators':dict(record.counts)}
            print(name,{k:bool(v) for k,v in features.items()},'transpose',transpose,flush=True)
        except Exception as e:report[name]={'status':'error','error':str(e)};print(name,'ERROR',e,flush=True)
        finally:
            if 'model' in locals():del model
            gc.collect()
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2)+'\n')
if __name__=='__main__':main()
