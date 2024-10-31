import torch
import torch.nn as nn
import torch.nn.functional as F
from stm.BaseEncoder.layers import MHA, PE, Encoder


class Fieldy(nn.Module):
    """Base encoder"""

    def __init__(
        self,
        configs
    ) -> None:
     
        super(Fieldy, self).__init__()
        self.N=configs['N']
        self.L, self.C =  configs['L'], configs['C']
        
        self.row_enc = Encoder(
            embed_dim=configs['embed_dim'],
            ff_shape=configs['ff_shape'],
            norm_shape=configs['embed_dim'],
            num_heads=configs['num_heads'],
            active_fn=configs['active_fn'],
            eps=configs['eps'],
            dropout=configs['dropout'],
            norm_first=configs['norm_first'],
            num_layers=configs['num_layers'],
        )
        self.col_enc = Encoder(
            embed_dim=configs['embed_dim'],
            ff_shape=configs['ff_shape'],
            norm_shape=configs['embed_dim'],
            num_heads=configs['num_heads'],
            active_fn=configs['active_fn'],
            eps=configs['eps'],
            dropout=configs['dropout'],
            norm_first=configs['norm_first'],
            num_layers=configs['num_layers'],
        )
        self.pe = PE(configs['embed_dim'])
        self.embed_dim=configs['embed_dim']
#         self.embed=nn.Linear(1,configs['embed_dim'])
        self.row_fc=nn.Linear(self.C,configs['embed_dim'])
        self.col_fc=nn.Linear(self.N,configs['embed_dim'])
#         self.row_fc2=nn.Linear()
        self.row_fc2=nn.Linear(configs['embed_dim'],configs['embed_dim'])
        self.col_fc2=nn.Linear(configs['embed_dim'],configs['embed_dim'])
        self.fc2 = nn.Linear(self.C *configs['embed_dim']*2, configs['embed_dim'])  # most parameters are here
        self.fc3 = nn.Linear(configs['embed_dim'], 1)
        

    def forward(self, x: torch.Tensor, pad_mask=None,signal=None) -> torch.Tensor:
        if x.ndim>3:
            B,N,_,_=x.shape
        h=x.squeeze(2)
        j=x.squeeze(2).clone()
        
        j=j.transpose(1,2)
        h=self.row_fc2(F.relu(self.row_fc(h)))
        j=self.col_fc2(F.relu(self.col_fc(j)))
        
#         row_enc,col_enc=row_enc.flatten(start_dim=0,end_dim=1),col_enc.flatten(start_dim=0,end_dim=1)
#         print(row_enc.shape,col_enc.shape)
        h=self.row_enc(h)
        j=self.col_enc(j)
        h=h.unsqueeze(-2).repeat(1,1,self.C,1)
        j=j.unsqueeze(-3).repeat(1,self.N,1,1)
        h=torch.cat((h,j),dim=-1)
#         print(h.shape)
        h=h.flatten(start_dim=-2)
        h=self.fc3(self.fc2(h)).squeeze(-1)
        if self.training:return h,torch.tensor(0)
        else:return h
    
