"""Small stock-PyTorch networks that expose gaps missed by the usual zoo subset."""
import torch
from torch import nn
from torch.nn import functional as F

class PoolClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.features=nn.Sequential(nn.Conv2d(3,8,3,padding=1),nn.ReLU(),
            nn.AvgPool2d(3,stride=2,padding=1,ceil_mode=True,count_include_pad=False),
            nn.Conv2d(8,12,3,padding=1),nn.GELU(),nn.AdaptiveAvgPool2d((3,2)))
        self.head=nn.Linear(72,5)
    def forward(self,x):return self.head(self.features(x).flatten(1))

class ResizeAutoencoder(nn.Module):
    def __init__(self,mode):
        super().__init__();self.mode=mode
        self.encoder=nn.Conv2d(3,8,3,stride=2,padding=1)
        self.decoder=nn.Conv2d(8,3,3,padding=1)
    def forward(self,x):
        h=F.silu(self.encoder(x))
        return self.decoder(F.interpolate(h,size=x.shape[-2:],mode=self.mode,align_corners=False))

class ReductionRegressor(nn.Module):
    def __init__(self):
        super().__init__();self.conv=nn.Conv2d(3,8,3,padding=1);self.head=nn.Linear(8,5)
    def forward(self,x):
        h=self.conv(x).clamp(-1.,1.).pow(2)
        return self.head(h.mean((2,3)))+h.sum((1,3)).mean(1,keepdim=True)*.001

class AttentionBlock(nn.Module):
    def __init__(self):
        super().__init__();self.qkv=nn.Linear(16,48);self.out=nn.Linear(16,16)
    def forward(self,x):
        q,k,v=self.qkv(x).chunk(3,dim=-1)
        return self.out(F.scaled_dot_product_attention(q,k,v,dropout_p=0.))

CASES={
 'pool_classifier':(PoolClassifier,(2,3,23,29)),
 'bilinear_autoencoder':(lambda:ResizeAutoencoder('bilinear'),(2,3,23,29)),
 'bicubic_autoencoder':(lambda:ResizeAutoencoder('bicubic'),(2,3,23,29)),
 'reduction_regressor':(ReductionRegressor,(2,3,23,29)),
 'attention_block':(AttentionBlock,(2,37,16)),
}

def make_case(name):
    factory,shape=CASES[name]
    torch.manual_seed(42);model=factory()
    x=torch.randn(shape,generator=torch.Generator().manual_seed(123))
    return model,x
