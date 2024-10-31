from typing import Optional,Tuple
import torch
import torch.nn as nn
from enum import IntEnum
from torch.nn import Parameter
import torch.nn.functional as F
class Dim(IntEnum):
    batch = 0
    seq = 1
    feature = 2
class MogLSTM(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.L=configs['L']
        self.C=configs['C']
        self.input_size = configs['C']
        self.hidden_size = configs['hidden_sz']
        self.mog_iterations = configs['mog_iterations']
        #Define/initialize all tensors   
        self.Wih = Parameter(torch.Tensor(self.input_size,self.hidden_size * 4))
#         self.Whh = Parameter(torch.Tensor(self.hidden_size, self.hidden_size * 4))
        self.bih = Parameter(torch.Tensor(self.hidden_size * 4))
        self.bhh = Parameter(torch.Tensor(self.hidden_size * 4))
        self.Wmx = Parameter(torch.Tensor(self.input_size,self.hidden_size))
        self.Wmh = Parameter(torch.Tensor(self.hidden_size,self.hidden_size))
        self.Whm = Parameter(torch.Tensor(self.hidden_size,self.hidden_size*4))
        #Mogrifiers
        self.Q = Parameter(torch.Tensor(self.hidden_size,self.input_size))
        self.R = Parameter(torch.Tensor(self.input_size,self.hidden_size))
        self.out=nn.Linear(self.hidden_size,1)
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
    def forward(self, x: torch.Tensor, 
                init_states: Optional[Tuple[torch.Tensor, torch.Tensor]]=None,pad_mask=None
               ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Assumes x is of shape (batch, sequence, feature)"""
        B,N,seq_sz, _ = x.size()
        x = x.view(-1,self.L,self.C)
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
            mt = xt @ self.Wmx + ht@self.Wmh
            gates = (xt @ self.Wih + self.bih) + (mt @ self.Whm + self.bhh)
#             gates = (xt @ self.Wih + self.bih) + (ht @ self.Whh + self.bhh)
            ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

            ### The LSTM Cell!
            ft = torch.sigmoid(forgetgate)
            it = torch.sigmoid(ingate)
            Ct_candidate = torch.tanh(cellgate)
            ot = torch.sigmoid(outgate)
            #outputs
            Ct = (ft * Ct) + (it * Ct_candidate)
            ht = ot * torch.tanh(Ct)
            ###

            hidden_seq.append(ht.unsqueeze(Dim.batch))
#         hidden_seq = torch.cat(hidden_seq, dim=Dim.batch)
        
        # reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
#         hidden_seq = hidden_seq.transpose(Dim.batch, Dim.seq).contiguous()
#         hidden_seq=hidden_seq.view(*hidden_seq.shape[:-2],self.L*self.hidden_size)
#         hidden_seq=self.out(hidden_seq).squeeze()
        output_seq=hidden_seq[-1]
        output_seq=self.out(output_seq)
        output_seq=output_seq.view(-1,N)
        if self.training:
            return output_seq,0
        else:
            return output_seq
    

from typing import Optional,Tuple
from enum import IntEnum
from torch.nn import Parameter
class Dim(IntEnum):
    batch = 0
    seq = 1
    feature = 2
class lstm_encoder(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.L=configs['L']
        self.C=configs['C']
        self.input_size = configs['C']
        self.hidden_size = configs['hidden_sz']
        self.mog_iterations = configs['mog_iterations']
        self.Wih=nn.Linear(self.input_size,self.hidden_size*4)
        self.Wmx=nn.Linear(self.input_size,self.hidden_size)
        self.Wmh=nn.Linear(self.hidden_size,self.hidden_size)
        self.Whm=nn.Linear(self.hidden_size,self.hidden_size*4)
        self.Wih=nn.Linear(self.input_size,self.hidden_size*4)
#         self.Q = Parameter(torch.Tensor(self.hidden_size,self.input_size))
#         self.R = Parameter(torch.Tensor(self.input_size,self.hidden_size))
#         self.out=nn.Linear(self.hidden_size,1)
        self.init_weights()
        self.encoder_attn = nn.Linear(
            in_features=2 * self.hidden_size + self.L,
            out_features=1
        )
    
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
    def forward(self, x: torch.Tensor, 
                init_states: Optional[Tuple[torch.Tensor, torch.Tensor]]=None,pad_mask=None
               ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Assumes x is of shape (batch, sequence, feature)"""
        x = x.view(-1,self.L,self.C)
        hidden_seq=[]
        c_seq=[]
        x_tilde_seq=[]
        batch_sz, seq_sz, _ = x.size()
        #ht and Ct start as the previous states and end as the output states in each loop below
        if init_states is None:
            ht = torch.zeros((batch_sz,self.hidden_size)).to(x.device)
            Ct = torch.zeros((batch_sz,self.hidden_size)).to(x.device)
        else:
            ht, Ct = init_states
        for t in range(seq_sz): # iterate over the time steps
#             ht,Ct=ht.unsqueeze(1),Ct.unsqueeze(1)
            temp_ht,temp_Ct=ht.unsqueeze(1).repeat(1,self.C,1),Ct.unsqueeze(1).repeat(1,self.C,1)
            temp_x=x.transpose(-2,-1)
            temp_x=torch.cat((temp_ht,temp_Ct,temp_x),dim=-1)
            temp_x=self.encoder_attn(temp_x).squeeze()
            alpha = F.softmax(temp_x.view(-1, self.input_size))                  
            xt = x[:, t, :]
#             print(alpha.shape)
            x_tilde = torch.mul(alpha, xt)
            x_tilde_seq.append(x_tilde.unsqueeze(Dim.batch))
#             xt, ht = self.mogrify(xt,ht) #mogrification
            mt = self.Wmx(xt)+self.Wmh(ht)
            gates=self.Wih(xt)+self.Whm(mt)
            ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
            ft = torch.sigmoid(forgetgate)
            it = torch.sigmoid(ingate)
            Ct_candidate = torch.tanh(cellgate)
            ot = torch.sigmoid(outgate)
            Ct = (ft * Ct) + (it * Ct_candidate)
            ht = ot * torch.tanh(Ct)
            hidden_seq.append(ht.unsqueeze(Dim.batch))
            c_seq.append(Ct.unsqueeze(Dim.batch))
        hidden_seq = torch.cat(hidden_seq, dim=Dim.batch)
        c_seq=torch.cat(c_seq,dim=Dim.batch)
        x_tilde=torch.cat(x_tilde_seq,dim=Dim.batch)
        hidden_seq = hidden_seq.transpose(Dim.batch, Dim.seq).contiguous()
        c_seq=c_seq.transpose(Dim.batch,Dim.seq).contiguous()
        x_tilde=x_tilde.transpose(Dim.batch,Dim.seq).contiguous()
        return hidden_seq,c_seq,x_tilde
class single_MogLSTM(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.L=configs['L']
        self.C=configs['C']
        self.input_size = configs['C']
        self.hidden_size = configs['hidden_sz']
        self.mog_iterations = configs['mog_iterations']
        #Define/initialize all tensors   
        self.dilation=configs['dilation']
        self.Wih=nn.Linear(self.input_size,self.hidden_size*4)
        self.Wmx=nn.Linear(self.input_size,self.hidden_size)
        self.Wmh=nn.Linear(self.hidden_size,self.hidden_size)
        self.Whm=nn.Linear(self.hidden_size,self.hidden_size*4)
        self.Wih=nn.Linear(self.input_size,self.hidden_size*4)
        self.Q = Parameter(torch.Tensor(self.hidden_size,self.input_size))
        self.R = Parameter(torch.Tensor(self.input_size,self.hidden_size))
#         self.out=nn.Linear(self.hidden_size,1)
        self.init_weights()
#         self.c_history=[]
    
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
    def forward(self, x: torch.Tensor, 
                init_states: Optional[Tuple[torch.Tensor, torch.Tensor]]=None,pad_mask=None
               ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Assumes x is of shape (batch, sequence, feature)"""
#         B,N,seq_sz, _ = x.size()
        x = x.view(-1,self.L,self.C)
        hidden_seq=[]
        h_history=[]
        batch_sz, seq_sz, _ = x.size()
        #ht and Ct start as the previous states and end as the output states in each loop below
        if init_states is None:
            ht = torch.zeros((batch_sz,self.hidden_size)).to(x.device)
            Ct = torch.zeros((batch_sz,self.hidden_size)).to(x.device)
            h_history.append(ht)
#             self.c_history.append(Ct)
        else:
            ht, Ct = init_states
        for t in range(seq_sz): # iterate over the time steps
            xt = x[:, t, :]
            xt, ht = self.mogrify(xt,ht) #mogrification
            h_history.append(ht)
#             if t>=self.dilation:
#                 prev_ht=h_history[-self.dilation]
#             else:
#                 prev_ht=h_history[0]
#             ht=torch.cat((ht,prev_ht),dim=-1)
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
            hidden_seq.append(ht.unsqueeze(Dim.batch))
        hidden_seq = torch.cat(hidden_seq, dim=Dim.batch)
#         reshape from shape (sequence, batch, feature) to (batch, sequence, feature)
        hidden_seq = hidden_seq.transpose(Dim.batch, Dim.seq).contiguous()
        del h_history
        return hidden_seq
#         hidden_seq=hidden_seq.view(*hidden_seq.shape[:-2],self.L*self.hidden_size)
#         hidden_seq=self.out(hidden_seq).squeeze()
#         output_seq=hidden_seq[-1]
#         output_seq=self.out(output_seq)
#         output_seq=output_seq.view(-1,N)
#         return output_seq
# class MogLSTM(nn.Module):
#     def __init__(self, configs):
#         super().__init__()
#         self.config=configs
#         self.L=configs['L']
#         self.C=configs['C']
#         self.layer=configs['layer']
#         self.input_size = configs['C']
#         self.hidden_size = configs['hidden_sz']
#         self.mog_iterations = configs['mog_iterations']
#         #Define/initialize all tensors   
#         self.dilation=configs['dilation']
#         self.input_layer,self.layers=self.make_layer()
# #         self.out=nn.Linear(self.hidden_size,self.input_size)
#         self.out=nn.Linear(self.hidden_size,1)
#         self.lstm_encoder=lstm_encoder(configs)
#     def make_layer(self):
#         self.config['dilation']=self.dilation[0]
#         input_layer=single_MogLSTM(self.config)
#         layers=nn.ModuleList()
#         self.config['C']=self.hidden_size
#         for i in range(self.layer-1):
#             self.config['dilation']=self.dilation[i+1]
#             layers.append(single_MogLSTM(self.config))
#         self.config['C']=self.input_size
#         return input_layer,layers
#     def forward(self,x,pad_mask):
#         B,N,seq_sz, _ = x.size()
#         x = x.view(-1,self.L,self.C)
#         _,_,x=self.lstm_encoder(x)
#         x=self.input_layer(x,pad_mask=pad_mask)
#         for layer in self.layers:
#             x=layer(x,pad_mask=pad_mask)
#         output_seq=self.out(x).squeeze()
#         output_seq=output_seq.view(B,N,-1)
# #         output_seq=x[:,-1,:]
# #         output_seq=self.out(output_seq)
# #         output_seq=output_seq.view(B,N)
# #         output_seq=self.out(x)
#         return output_seq

        
        
        