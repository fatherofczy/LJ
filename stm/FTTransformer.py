import torch
import torch.nn as nn

from ..BaseEncoder.layers import MHA, PE, Encoder


class FTTransformer(nn.Module):
    """Base encoder"""

    def __init__(
        self,
        configs
    ) -> None:
     
        super(FTTransformer, self).__init__()
        
        self.L, self.C =  configs['L'], configs['C']
        self.enc = Encoder(
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
#         self.pe = PE(configs['embed_dim'])
#         self.embed=nn.Linear(1,configs['embed_dim'])
        self.fc1 = nn.Linear(1, configs['embed_dim'])
        self.fc2 = nn.Linear(self.C *configs['embed_dim'], configs['embed_dim'])  # most parameters are here
        self.fc3 = nn.Linear(configs['embed_dim'], 1)
        

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor,signal=None) -> torch.Tensor:
        if x.ndim>3:
            B,N,_,_=x.shape
        h=x.squeeze(2).unsqueeze(-1)
#         h=self.embed(h)
        h = x.view(-1, self.C, 1)  # BN*L*C
#         x=(x-torch.mean(x,dim=0,keepdim=True))/torch.std(x,dim=0,keepdim=True)
        h = self.fc1(h)  # BN*L*embed_dim
#         h = h + self.pe(h)  # BN*L*embed_dim
#         print(h.shape)
        h = self.enc(h)
        h = h.flatten(start_dim=1)  # BN*(L*embed_dim)
        h = self.fc2(h)  # BN*embed_dim
        h = self.fc3(h).squeeze(-1)  # BN
        h = h.view(-1, N)  # B*N
        if self.training:return h,torch.tensor(0)
        else:return h