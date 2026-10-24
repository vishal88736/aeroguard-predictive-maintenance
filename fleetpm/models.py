"""Compact causal repair / CNN-GRU / attention / spline RUL model."""

import math
import torch
from torch import nn
from torch.nn import functional as F

from .features import STAT_DIM


class SplineLinear(nn.Module):
    def __init__(self,inputs,outputs,grid=5,degree=3):
        super().__init__()
        self.degree=degree
        knots=torch.arange(-degree,grid+degree+1,dtype=torch.float32)*(2/grid)-1
        self.register_buffer("knots",knots)
        self.base=nn.Linear(inputs,outputs)
        self.coefficients=nn.Parameter(torch.empty(outputs,inputs,grid+degree))
        nn.init.normal_(self.coefficients,std=.025)

    def forward(self,x):
        # Spline recurrence stays float32 even when CNN/GRU use CUDA AMP.
        with torch.autocast(device_type=x.device.type,enabled=False):
            x=torch.tanh(x.float())
            z=x.unsqueeze(-1); t=self.knots
            basis=((z>=t[:-1])&(z<t[1:])).to(x.dtype)
            for k in range(1,self.degree+1):
                basis=((z-t[:-k-1])/(t[k:-1]-t[:-k-1])*basis[...,:-1]
                       +(t[k+1:]-z)/(t[k+1:]-t[1:-k])*basis[...,1:])
            return self.base(F.silu(x))+torch.einsum("bin,oin->bo",basis,self.coefficients)


class CausalConv(nn.Module):
    def __init__(self,inputs,outputs,kernel):
        super().__init__(); self.kernel=kernel
        self.conv=nn.Conv1d(inputs,outputs,kernel)

    def forward(self,x):
        return self.conv(F.pad(x,(self.kernel-1,0)))


class HybridRUL(nn.Module):
    def __init__(self,window=30,cap=125.0):
        super().__init__(); self.window=window; self.cap=cap
        self.repair=nn.GRU(66,32,batch_first=True)
        self.repair_output=nn.Linear(32,21)
        self.convolutions=nn.Sequential(CausalConv(66,32,3),nn.ReLU(),nn.Dropout(.2),
                                        CausalConv(32,64,5),nn.ReLU(),nn.Dropout(.2))
        self.recurrent=nn.GRU(64,64,batch_first=True)
        encoder=nn.TransformerEncoderLayer(64,4,128,dropout=.2,batch_first=True,norm_first=True)
        self.attention=nn.TransformerEncoder(encoder,1,enable_nested_tensor=False)
        self.gate=nn.Linear(128,64)
        self.statistics=nn.Sequential(nn.Linear(STAT_DIM,32),nn.ReLU(),nn.Linear(32,16),nn.ReLU())
        self.bottleneck=nn.Sequential(nn.Linear(83,32),nn.Tanh())
        self.head=nn.Sequential(SplineLinear(32,8),SplineLinear(8,1))
        position=torch.arange(window).unsqueeze(1)
        div=torch.exp(torch.arange(0,64,2)*(-math.log(10000)/64))
        encoding=torch.zeros(window,64)
        encoding[:,0::2]=torch.sin(position*div); encoding[:,1::2]=torch.cos(position*div)
        self.register_buffer("positions",encoding)
        self.register_buffer("causal_mask",torch.triu(torch.ones(window,window,dtype=torch.bool),diagonal=1))

    def repair_packet(self,x):
        hidden,_=self.repair(x)
        generated=self.repair_output(hidden)
        mask=x[:,:,24:45]
        sensors=mask*x[:,:,:21]+(1-mask)*generated
        return torch.cat([sensors,x[:,:,21:]],dim=-1),generated

    def forward(self,x,stats,quality,lengths,valid):
        packet,generated=self.repair_packet(x)
        local=self.convolutions(packet.transpose(1,2)).transpose(1,2)
        recurrent,_=self.recurrent(local)
        attended=self.attention(local+self.positions[:local.shape[1]],
                                mask=self.causal_mask[:local.shape[1],:local.shape[1]],
                                src_key_padding_mask=~valid)
        index=(lengths-1).long(); batch=torch.arange(len(x),device=x.device)
        hr=recurrent[batch,index]; ha=attended[batch,index]
        gate=torch.sigmoid(self.gate(torch.cat([hr,ha],dim=-1)))
        fused=gate*hr+(1-gate)*ha
        latent=self.bottleneck(torch.cat([fused,self.statistics(stats),quality],dim=-1))
        prediction=self.cap*torch.sigmoid(self.head(latent).squeeze(-1))
        return prediction,generated

    def sparsity(self):
        return sum(layer.coefficients.abs().mean() for layer in self.head)
