# from ..MogLSTM.lstm import MogLSTM
from ..BaseEncoder.model import BaseEncoder
import torch.nn as nn
import torch
from ..BaseEncoder.layers import MHA, PE, Encoder, MLP
from typing import Optional,Tuple
from enum import IntEnum
from torch.nn import Parameter
from torch.profiler import profile, record_function, ProfilerActivity
class Dim(IntEnum):
    batch = 0
    seq = 1
    feature = 2
class single_MogLSTM(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.L=configs['L']
        self.C=configs['C']
        self.input_size = configs['C']
        self.hidden_size = configs['hidden_sz']
        self.mog_iterations = configs['mog_iterations']
        #Define/initialize all tensors   
        self.Wih=nn.Linear(self.input_size,self.hidden_size*4)
        self.Wmx=nn.Linear(self.input_size,self.hidden_size)
        self.Wmh=nn.Linear(self.hidden_size,self.hidden_size)
        self.Whm=nn.Linear(self.hidden_size,self.hidden_size*4)
        self.Wih=nn.Linear(self.input_size,self.hidden_size*4)
        self.Q = Parameter(torch.Tensor(self.hidden_size,self.input_size))
        self.R = Parameter(torch.Tensor(self.input_size,self.hidden_size))
#         self.out=nn.Linear(self.hidden_size,1)
        self.init_weights()
    
    def init_weights(self):
        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

    def mogrify(self,xt,ht):
        for i in range(1,self.mog_iterations+1):
            if (i % 2 == 0):
                ht = (2*torch.sigmoid(xt @ self.R)) * ht
            else:
                xt = (2*torch.sigmoid(ht @ self.Q)) * xt
        return xt, ht

    
    #Define forward pass through all LSTM cells across all timesteps.
    #By using PyTorch functions, we get backpropagation for free.
    def forward(self, x: torch.Tensor, pad_mask=None,
                init_states: Optional[Tuple[torch.Tensor, torch.Tensor]]=None
               ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Assumes x is of shape (batch, sequence, feature)"""
        if x.ndim==4:
            B,N,seq_sz, _ = x.size()
            x = x.view(-1,*x.shape[2:])
        batch_sz, seq_sz, _ = x.size()
        hidden_seq = []
        #ht and Ct start as the previous states and end as the output states in each loop below
        if init_states is None:
            ht = torch.zeros((batch_sz,self.hidden_size)).to(x.device)
            Ct = torch.zeros((batch_sz,self.hidden_size)).to(x.device)
        else:
            ht, Ct = init_states
        for t in range(seq_sz): # iterate over the time steps
            xt = x[:, t, :]
            xt, ht = self.mogrify(xt,ht) #mogrification
            mt = self.Wmx(xt)+self.Wmh(ht)
#             mt = xt @ self.Wmx + ht@self.Wmh
            gates=self.Wih(xt)+self.Whm(mt)
#             gates = (xt @ self.Wih + self.bih) + (mt @ self.Whm + self.bhh)
#             gates = (xt @ self.Wih + self.bih) + (ht @ self.Whh + self.bhh)
            ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
            ft = torch.sigmoid(forgetgate)
            it = torch.sigmoid(ingate)
            Ct_candidate = torch.tanh(cellgate)
            ot = torch.sigmoid(outgate)
            #outputs
            Ct = (ft * Ct) + (it * Ct_candidate)
            ht = ot * torch.tanh(Ct)
            ###

            hidden_seq.append(ht.unsqueeze(Dim.batch))
        hidden_seq = torch.cat(hidden_seq, dim=Dim.batch)
#         reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
        hidden_seq = hidden_seq.transpose(Dim.batch, Dim.seq).contiguous()
        return hidden_seq
#         hidden_seq=hidden_seq.view(*hidden_seq.shape[:-2],self.L*self.hidden_size)
#         hidden_seq=self.out(hidden_seq).squeeze()
#         output_seq=hidden_seq[-1]
#         output_seq=self.out(output_seq)
#         output_seq=output_seq.view(-1,N)
#         return output_seq

class MogLSTM(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.config=configs
        self.L=configs['L']
        self.C=configs['C']
        self.layer=configs['layer']
        self.input_size = configs['C']
        self.hidden_size = configs['hidden_sz']
        self.mog_iterations = configs['mog_iterations']
        #Define/initialize all tensors   
        self.input_layer,self.layers=self.make_layer()
#         self.out=nn.Linear(self.hidden_size,1)
        self.out=nn.Linear(self.hidden_size,self.input_size)
    def make_layer(self):
        input_layer=single_MogLSTM(self.config)
        layers=nn.ModuleList()
        self.config['C']=self.hidden_size
        for i in range(self.layer-1):
            layers.append(single_MogLSTM(self.config))
        self.config['C']=self.input_size
        return input_layer,layers
    def forward(self,x,pad_mask):
        if x.ndim==4:
            B,N,seq_sz, _ = x.size()
            x = x.view(-1,*x.shape[2:])
        x=self.input_layer(x,pad_mask)
        for layer in self.layers:
            x=layer(x,pad_mask)
#         output_seq=x[:,-1,:]
#         output_seq=self.out(output_seq)
#         output_seq=output_seq.view(-1,N)
        output_seq=self.out(x)
#         output_seq=x
        return output_seq
        
class EncoderLayer(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        ff_shape: list,
        num_heads: int = 1,
        active_fn: str = "relu",
        eps: float = 1e-6,
        dropout: float = 0.0,
        norm_first: bool = False,
    ) -> None:
        super(EncoderLayer, self).__init__()
        ff_shape = [ff_shape] if isinstance(ff_shape, int) else ff_shape
        norm_shape=embed_dim
        self.sa = MHA(embed_dim, num_heads)
        self.ff = MLP([embed_dim, *ff_shape, embed_dim], active_fn)
        self.n1 = nn.LayerNorm(norm_shape, eps) if norm_shape else nn.Identity()
        self.n2 = nn.LayerNorm(norm_shape, eps) if norm_shape else nn.Identity()
        self.d1 = nn.Dropout(dropout)
        self.d2 = nn.Dropout(dropout)
        self.norm_first = norm_first
#         self.attn_mask=

    def forward(
        self, x_enc: torch.Tensor, enc_pad_mask: torch.Tensor = None
    ) -> torch.Tensor:
        if self.norm_first:
            x_enc = self.n1(x_enc)
            with record_function("MHA"):
                x_enc = x_enc + self.d1(self.sa(x_enc, x_enc, x_enc, enc_pad_mask))
            x_enc = self.n2(x_enc)
            
            x_enc = x_enc + self.d2(self.ff(x_enc))
        else:
            with record_function("MHA"):
                x_enc = x_enc + self.d1(self.sa(x_enc, x_enc, x_enc, enc_pad_mask))
            x_enc = self.n1(x_enc)
            x_enc = x_enc + self.d2(self.ff(x_enc))
            x_enc = self.n2(x_enc)
        return x_enc
class Encoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        ff_shape: list,
        num_heads: int = 1,
        active_fn: str = "relu",
        eps: float = 1e-6,
        dropout: float = 0.0,
        norm_first: bool = False,
        num_layers: int = 2,
    ) -> None:
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList(
            [
                EncoderLayer(
                    embed_dim=embed_dim,
                    ff_shape=ff_shape,
                    num_heads=num_heads,
                    active_fn=active_fn,
                    eps=eps,
                    dropout=dropout,
                    norm_first=norm_first,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(
        self, x_enc: torch.Tensor, enc_pad_mask: torch.Tensor = None
    ) -> torch.Tensor:
        for layer in self.layers:
            x_enc = layer(x_enc, enc_pad_mask)
        return x_enc

class lstm_transformer(nn.Module):
    def __init__(self,configs):
        super(lstm_transformer,self).__init__()
        self.config=configs
        self.L=configs['L']
        self.C=configs['C']
        self.layer=configs['layer']
        self.num_layer=configs['num_layer']
        self.input_size = configs['C']
        self.hidden_size = configs['hidden_sz']
        self.mog_iterations = configs['mog_iterations']
        self.embed_dim=configs['embed_dim']
        self.ff_shape=configs['ff_shape']
        self.num_heads=configs['num_heads']
        self.p=configs['dropout']
        self.lin=nn.Linear(self.C,self.embed_dim)
        self.encoder=Encoder(embed_dim=self.embed_dim,num_heads=self.num_heads,ff_shape=self.ff_shape,dropout=self.p,num_layers=self.num_layer)
        self.config['C']=self.embed_dim
        self.moglstm=MogLSTM(self.config)
        self.pe = PE(configs['embed_dim'])
#         self.moglstm,self.BaseEncoder=self.make_model()
#         self.out=nn.Linear(self.embed_dim,1)
        self.out=nn.Linear(self.hidden_size,1)
#         self.sa = MHA(self.embed_dim, self.num_heads)
        norm_shape=self.embed_dim
    
#         ff_shape = [self.ff_shape] if isinstance(self.ff_shape, int) else self.ff_shape
#         self.ff = MLP([self.embed_dim, *ff_shape, self.embed_dim], active_fn='relu')
#         self.n1 = nn.LayerNorm(norm_shape, eps=1e-6) if norm_shape else nn.Identity()
#         self.n2 = nn.LayerNorm(norm_shape, eps=1e-6) if norm_shape else nn.Identity()
        self.d1 = nn.Dropout(self.p)
#         self.out=nn.Linear(self.L*self.embed_dim,1)
#         self.d2 = nn.Dropout(self.p)
    def make_model(self):
        moglstm=MogLSTM(self.config)
        baseencoder=BaseEncoder(self.config)
        return moglstm,baseencoder
    def forward(self,x,pad_mask):
        B,N,L,C=x.shape
        x=x.transpose(1,2)
        x=x.view(-1,*x.shape[2:])
        x=self.lin(x)
#         x=x+self.pe(x)
        x=self.encoder(x)
        x=x.view(B,L,*x.shape[1:])
        x=x.transpose(1,2)
        x=self.moglstm(x,pad_mask)
#         temp=temp.view(B,N,*x.shape[2:])
#         x=x+self.d1(temp)
#         x=x.view(-1,*x.shape[2:])
#         x=x[...,self.L//2:,:]
        x=x[:,-1,:]
        x=self.out(x).squeeze()
        x=x.view(-1,N)
        return x
