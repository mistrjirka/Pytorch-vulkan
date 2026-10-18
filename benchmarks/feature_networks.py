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


class VolumeAutoencoder(nn.Module):
    def __init__(self):
        super().__init__();self.encoder=nn.Conv3d(2,4,3,padding=1);self.decoder=nn.ConvTranspose3d(4,2,3,stride=2,padding=1,output_padding=1)
    def forward(self,x):
        h=F.avg_pool3d(F.gelu(self.encoder(x)),2)
        h=F.max_pool3d(self.decoder(h),2)
        return F.interpolate(h,size=x.shape[-3:],mode='trilinear',align_corners=False)

class SpatialTransformer(nn.Module):
    def __init__(self):
        super().__init__();self.conv=nn.Conv2d(2,3,3);self.grid=nn.Parameter(torch.randn(1,5,6,2)*.4)
    def forward(self,x):
        h=self.conv(F.pad(x,(1,1,1,1),mode='reflect'))
        h=F.interpolate(h,size=(7,9),mode='bilinear',antialias=True,align_corners=False)
        return F.grid_sample(h,self.grid.expand(x.shape[0],-1,-1,-1),align_corners=False)

class RecurrentRegressor(nn.Module):
    def __init__(self):
        super().__init__();self.lstm=nn.LSTM(5,7,2,bidirectional=True,batch_first=True);self.head=nn.Linear(14,3)
    def forward(self,x):return self.head(self.lstm(x)[0])

class RetrievalRegressor(nn.Module):
    def __init__(self):
        super().__init__();self.embedding=nn.EmbeddingBag(19,7,mode='mean');self.head=nn.Linear(7,4)
        self.register_buffer('indices',torch.tensor([1,2,2,4,7,1,12,4]));self.register_buffer('offsets',torch.tensor([0,4]))
        self.register_buffer('selection',torch.tensor([3,1,6,0,4,2,5]))
    def forward(self,x):
        h=self.embedding(self.indices,self.offsets)+x
        h=h.index_select(1,self.selection).sort(dim=1).values.cumsum(1)
        return self.head(h)+h.var(1,correction=0,keepdim=True)*.01

class SpectralFilter(nn.Module):
    def __init__(self):
        super().__init__();self.weights=nn.Parameter(torch.randn(9)*.1+1);self.head=nn.Linear(16,5)
    def forward(self,x):return self.head(torch.fft.irfft(torch.fft.rfft(x)*self.weights,n=16))

class SolveRegressor(nn.Module):
    def __init__(self):
        super().__init__();self.factor=nn.Parameter(torch.randn(5,5)*.1);self.register_buffer('identity',torch.eye(5));self.head=nn.Linear(5,3)
    def forward(self,x):
        a=self.factor@self.factor.T+self.identity
        return self.head(torch.linalg.solve(a,x.T).T)

class CausalAttentionBlock(AttentionBlock):
    def forward(self,x):
        q,k,v=self.qkv(x).chunk(3,dim=-1)
        return self.out(F.scaled_dot_product_attention(q,k,v,is_causal=True))

COMPLETION_CASES={
 'volume_autoencoder':(VolumeAutoencoder,(2,2,6,8,10)),
 'spatial_transformer':(SpatialTransformer,(2,2,9,11)),
 'recurrent_regressor':(RecurrentRegressor,(2,4,5)),
 'retrieval_regressor':(RetrievalRegressor,(2,7)),
 'spectral_filter':(SpectralFilter,(3,16)),
 'solve_regressor':(SolveRegressor,(3,5)),
 'causal_attention_block':(CausalAttentionBlock,(2,17,16)),
}
CASES.update(COMPLETION_CASES)
