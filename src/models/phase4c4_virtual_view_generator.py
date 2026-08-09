from __future__ import annotations
import torch
from torch import nn

class Block(nn.Module):
    def __init__(self,w,d):
        super().__init__(); self.net=nn.Sequential(nn.Linear(w,w),nn.LayerNorm(w),nn.GELU(),nn.Dropout(d),nn.Linear(w,w),nn.LayerNorm(w),nn.GELU(),nn.Dropout(d))
    def forward(self,x): return x+self.net(x)

class GeometryConditionedVirtualViewGenerator(nn.Module):
    def __init__(self,hidden_width=1024,residual_blocks=3,dropout=0.10,translation_scale_m=5.0):
        super().__init__(); self.translation_scale_m=float(translation_scale_m)
        self.input=nn.Sequential(nn.Linear(40,hidden_width),nn.LayerNorm(hidden_width),nn.GELU(),nn.Dropout(dropout))
        self.blocks=nn.Sequential(*[Block(hidden_width,dropout) for _ in range(residual_blocks)])
        self.output=nn.Linear(hidden_width,28)
    def forward(self,pose,r,t):
        b=pose.shape[0]; I=torch.eye(3,dtype=r.dtype,device=r.device).expand(b,-1,-1)
        g=torch.cat([(r-I).flatten(1),t/self.translation_scale_m],1)
        x=torch.cat([pose.flatten(1),g],1)
        return pose+self.output(self.blocks(self.input(x))).reshape(-1,14,2)
