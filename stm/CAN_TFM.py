import torch
import torch.nn as nn
from stm.BaseEncoder.layers import *
class CAN_TFM(nn.Module):
    def __init__(self,configs):
        super(CAN_TFM,self).__init__()
        input_size=configs['C']
#         embed_dim=configs['embed_dim']
        dim=input_size+128
        self.enc=Encoder(dim,ff_shape=1024)
        self.mask=torch.load('/dfs/data/class_labels.pt')
        self.fc=nn.Linear(dim,dim//2)
        self.fc2=nn.Linear(dim//2,1)
        self.prelu=nn.PReLU()
    def forward(self,x,pad_mask=None,signal=None,y=None):
        cur=x[:,:,-1,:]
        self.mask=self.mask.to(x.device)
        rxz=torch.index_select(x,dim=-1,index=torch.nonzero(self.mask==0).squeeze(-1))
        yyt=torch.index_select(x,dim=-1,index=torch.nonzero(self.mask==1).squeeze(-1))
        if rxz.shape[-1]<yyt.shape[-1]:
            rxz,yyt=yyt,rxz
        rxz=rxz.permute(0,1,3,2)
        yyt=yyt+yyt**2+yyt**3
        yyt=torch.matmul(yyt,rxz[...,:yyt.shape[-1],:])
        yyt=torch.sigmoid(yyt)
        rxz=rxz.permute(0,1,3,2)
        yyt=torch.matmul(yyt,rxz[...,yyt.shape[-1]:yyt.shape[-1]+2])
        yyt=torch.sigmoid(yyt)
        yyt=yyt.reshape(*yyt.shape[:-2],-1)
        x=torch.cat((cur,yyt),dim=-1)
        
        x=self.enc(x)
        x=self.fc(x)
        x=self.prelu(x)
        x=self.fc2(x).squeeze(-1)
        if self.training:
            return x,torch.tensor(0)
        else:
            return x
        