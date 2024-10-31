import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from ..BaseEncoder.layers import MHA,Encoder
class VAE(nn.Module):
    def __init__(self, input_size,hidden_dim, latent_dim):
        super(VAE, self).__init__()
        
        # Encoder
#         self.fc1=nn.Linear(input_size,hidden_dim)
        self.fc2_mu = nn.Linear(hidden_dim, latent_dim)  # Mean vector of the latent space
        self.fc2_logvar = nn.Linear(hidden_dim, latent_dim)  # Log variance of the latent space

        # Decoder
        
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_size)

    def encode(self, x):
    
        mu = self.fc2_mu(x)
        logvar = self.fc2_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)  # Sample from a standard normal distribution
        return mu + eps * std  # Reparameterization trick

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return self.fc4(h3)  # Output reconstructed data

    def forward(self, x,temp):
        
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        
        recon_loss = F.mse_loss(self.decode(z), temp, reduction='mean')*temp.shape[-1]  # Mean Squared Error
        kl_loss = torch.mean(-0.5 * torch.sum((logvar - mu.pow(2) - logvar.exp()),dim=-1))*0
        return recon_loss + kl_loss



class WITRAN_2DPSGMU_Encoder(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, water_rows, water_cols, res_mode='layer_yes'):
        super(WITRAN_2DPSGMU_Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.water_rows = water_rows
        self.water_cols = water_cols
        self.res_mode = res_mode

        # parameter of row cell
        self.W_first_layer = torch.nn.Parameter(torch.empty(6 * hidden_size, input_size + 2 * hidden_size))
#         self.W_other_layer = torch.nn.Parameter(torch.empty(num_layers - 1, 6 * hidden_size, 4 * hidden_size))
        self.B = torch.nn.Parameter(torch.empty(num_layers, 6 * hidden_size))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, +stdv)

    def linear(self, input, weight, bias, batch_size, slice, Water2sea_slice_num):
        a = F.linear(input, weight)
        if slice < Water2sea_slice_num:
            a[:batch_size * (slice + 1), :] = a[:batch_size * (slice + 1), :] + bias
        return a

    def forward(self, input):
        flag=1
        batch_size,_,_,input_size=input.shape
        if flag == 1: # cols > rows
            input = input.permute(2, 0, 1, 3)
        else:
            input = input.permute(1, 0, 2, 3)
        Water2sea_slice_num, _, Original_slice_len, _ = input.shape
        Water2sea_slice_len = Water2sea_slice_num + Original_slice_len - 1
        hidden_slice_row = torch.zeros(Water2sea_slice_num * batch_size, self.hidden_size).to(input.device)
        hidden_slice_col = torch.zeros(Water2sea_slice_num * batch_size, self.hidden_size).to(input.device)
        input_transfer = torch.zeros(Water2sea_slice_num, batch_size, Water2sea_slice_len, input_size).to(input.device)
#         print(input_transfer.shape)
        for r in range(Water2sea_slice_num):
            input_transfer[r, :, r:r+Original_slice_len, :] = input[r, :, :, :]
        hidden_row_all_list = []
        hidden_col_all_list = []
        for layer in range(self.num_layers):
            if layer == 0:
                a = input_transfer.reshape(Water2sea_slice_num * batch_size, Water2sea_slice_len, input_size)
                W = self.W_first_layer
            else:
                a = F.dropout(output_all_slice, self.dropout, self.training)
                if layer == 1:
                    layer0_output = a
                W = self.W_other_layer[layer-1, :, :]
                hidden_slice_row = hidden_slice_row * 0
                hidden_slice_col = hidden_slice_col * 0
            B = self.B[layer, :]
            # start every for all slice
            output_all_slice_list = []
            for slice in range (Water2sea_slice_len):
                # gate generate
                gate = self.linear(torch.cat([hidden_slice_row, hidden_slice_col, a[:, slice, :]], 
                    dim = -1), W, B, batch_size, slice, Water2sea_slice_num)
                # gate
                sigmod_gate, tanh_gate = torch.split(gate, 4 * self.hidden_size, dim = -1)
                sigmod_gate = torch.sigmoid(sigmod_gate)
                tanh_gate = torch.tanh(tanh_gate)
                update_gate_row, output_gate_row, update_gate_col, output_gate_col = sigmod_gate.chunk(4, dim = -1)
                input_gate_row, input_gate_col = tanh_gate.chunk(2, dim = -1)
                # gate effect
                hidden_slice_row = torch.tanh(
                    (1-update_gate_row)*hidden_slice_row + update_gate_row*input_gate_row) * output_gate_row
                hidden_slice_col = torch.tanh(
                    (1-update_gate_col)*hidden_slice_col + update_gate_col*input_gate_col) * output_gate_col
                # output generate
#                 print(hidden_slice_row.shape)
               
                output_slice = torch.cat([hidden_slice_row, hidden_slice_col], dim = -1)
                # save output
                output_all_slice_list.append(output_slice)
                # save row hidden
                if slice >= Original_slice_len - 1:
                    need_save_row_loc = slice - Original_slice_len + 1
                    hidden_row_all_list.append(
                        hidden_slice_row[need_save_row_loc*batch_size:(need_save_row_loc+1)*batch_size, :])
                # save col hidden
                if slice >= Water2sea_slice_num - 1:
                    hidden_col_all_list.append(
                        hidden_slice_col[(Water2sea_slice_num-1)*batch_size:, :])
                # hidden transfer
                hidden_slice_col = torch.roll(hidden_slice_col, shifts=batch_size, dims = 0)
#                 print(hidden_slice_col.shape)
            if self.res_mode == 'layer_res' and layer >= 1: # layer-res
                output_all_slice = torch.stack(output_all_slice_list, dim = 1) + layer0_output
            else:
                output_all_slice = torch.stack(output_all_slice_list, dim = 1)
        hidden_row_all = torch.stack(hidden_row_all_list, dim = 1)
        hidden_col_all = torch.stack(hidden_col_all_list, dim = 1)
        hidden_row_all = hidden_row_all.reshape(batch_size, self.num_layers, Water2sea_slice_num, hidden_row_all.shape[-1])
        hidden_col_all = hidden_col_all.reshape(batch_size, self.num_layers, Original_slice_len, hidden_col_all.shape[-1])
        if flag == 1:
            return output_all_slice, hidden_col_all, hidden_row_all
        else:
            return output_all_slice, hidden_row_all, hidden_col_all
class VAE_WITRAN(torch.nn.Module):
    def __init__(self,configs):
        super(VAE_WITRAN, self).__init__()
#         print(configs)
        self.C=configs['C']
        self.L=configs['L']
        self.hidden_size=configs['hidden_size']
        self.row=configs['row']
        self.col=configs['col']
        self.dropout=configs['dropout']
        self.num_layers=configs['num_layers']
        self.type=configs['type']
        self.num_experts=configs['experts']
        self.wit= WITRAN_2DPSGMU_Encoder(input_size=self.hidden_size,hidden_size=self.hidden_size,num_layers=self.num_layers,dropout=self.dropout,water_rows=self.row,
                                         water_cols=self.col)
        self.fc3 = nn.Linear(self.C,4*self.hidden_size)
        self.fc4= nn.Linear(4*self.hidden_size,self.hidden_size)
        self.relu=nn.ReLU()
#         self.mha=MHA(self.L,2,0.2)
#         self.encoder=Encoder(
#             embed_dim=self.L,
#             ff_shape=configs['ff_shape'],
#             norm_shape=self.L,
#             num_heads=2,
#             active_fn=configs['active_fn'],
#             eps=configs['eps'],
#             dropout=configs['dropout'],
#             norm_first=configs['norm_first'],
#             num_layers=1,
#         )
#         self.attn_mask=torch.load('/dfs/data/attn_mask.pt')
        self.vae=VAE(self.C,self.hidden_size,2*self.hidden_size)
#         self.fc=nn.Linear(self.num_layers*(self.row+self.col)*self.hidden_size,1)
#         self.fc=nn.Linear(self.col*2*self.hidden_size,self.L)
#         self.fc=nn.Linear(self.hidden_size,1)
#         if self.type=='reg':
#             self.fc1=nn.Linear(self.hidden_size,1)
#             self.fc2=nn.Linear(self.hidden_size,1)
#         elif self.type=='class':
#             self.fc3=nn.Linear(2*self.hidden_size,2)
#         self.multi=nn.Linear(2*self.hidden_size,2)
#         self.router=nn.Linear(2*self.hidden_size,self.num_experts)
#         self.experts=nn.Linear(2*self.hidden_size,self.num_experts)
#         self.sigmoid=nn.Sigmoid()
#         self.batchnorm1d = nn.BatchNorm1d(num_features=2*self.hidden_size)
        self.fc1=nn.Linear(self.hidden_size,1)
        self.fc2=nn.Linear(self.hidden_size,1)
#         self.fc3=nn.Linear(2*self.hidden_size,2)
    
    def forward(self,x,pad_mask,signal=None):

 
        judge=x.ndim==4
        if x.ndim==4:
            B,N,L,C=x.shape
            x=x.reshape(B*N,L,C)
        temp=x.clone()
#         self.attn_mask=self.attn_mask.to(x.device)
#         x=torch.transpose(x,-2,-1)
#         x=x.reshape(-1,C,L)
#         x=self.encoder(x)
#         x=x.reshape(B,N,C,L)
#         x=torch.transpose(x,-2,-1)
#         x=self.fc3(x)
        x=self.fc4(self.relu(self.fc3(x)))
        x=x.reshape(B*N,L,-1)
        
        if signal=='vae':
            loss=self.vae(x,temp)
            return loss
        assert self.row*self.col==self.L and self.row>=self.col
        x=x.reshape(-1,self.row,self.col,self.hidden_size)
        _,row,col=self.wit(x)
        row,col=row[...,-1,:],col[...,-1,:]
#         output=torch.cat([row,col],dim=-1).squeeze()
# #         output=self.multi(output)
# #         temp=torch.min(output[...,1]).detach()
# #         output[...,1]=torch.log(output[...,1]-temp+1)
#         output,prob=self.experts(output),F.softmax(self.router(output),dim=-1)
#         prob_sum=prob.sum(dim=0)
#         max_values, max_indices = torch.max(prob, dim=1)
#         output = torch.gather(output, 1, max_indices.view(-1, 1)).view(-1)
#         count_indices=torch.bincount(max_indices,minlength=self.num_experts)
# #         count_indices=[0]*self.experts
# #         for i in max_indices:
# #             count_indices[i]+=1
# #         count_indices = count_indices.to(prob)
#         if judge:
# #             output=output.view(B,N,-1)
#             output=output.view(B,N)
#         if self.training:
# #             aux_loss=torch.sum(prob_sum*count_indices)*self.num_experts/(N**2)*0
#             return (output,0)
#         else:
# #             return output[...,0]
#             return output
        
#         return output
        
        if self.type=='reg':
            output=0.5*self.fc1(row)+0.5*self.fc2(col)
            output=output.squeeze()
            if judge:
                output=output.view(B,N)
        elif self.type=='class':
            output=torch.cat([row,col],dim=-1)
            output = self.fc3(output)
            output=output.squeeze()
            if judge:
                output=output.view(B,N,-1)
#         row=row[:,:,-1:,:].expand(-1,-1,self.col,-1)
#         output = torch.cat([row,col], dim = -1)
#         output=output[:,-1,...].reshape(output.shape[0],-1)
#         output=self.fc(output).squeeze()
#         output=output.view(B,N,-1)
#         temp=torch.concat([row,col],dim=2)
#         temp=temp.reshape(temp.shape[0],-1)
#         temp=self.fc(temp).squeeze()
#         temp=temp.view(B,temp.shape[0])
        if self.training:
            return output,torch.tensor(0)
        else:
            return output