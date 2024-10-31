import torch
import torch.nn as nn
import math
import torch.nn.functional as F
import torch.nn.init as init
class class_VAE(nn.Module):
    def __init__(self, input_size,hidden_dim, latent_dim,num):
        super(class_VAE, self).__init__()
        self.num=num
        self.input_size=input_size
        # Encoder
        self.fc1=nn.Linear(input_size,hidden_dim)
        self.fc=nn.Linear(hidden_dim,self.num)
        self.fc2_mu = nn.ModuleList([nn.Linear(hidden_dim, latent_dim) for _ in range(self.num)]) # Mean vector of the latent space
        self.fc2_logvar =nn.ModuleList([nn.Linear(hidden_dim, latent_dim) for _ in range(self.num)])  # Log variance of the latent space
        self.mean=torch.nn.Parameter(torch.randn(self.num,latent_dim))
        # Decoder
        init.xavier_uniform_(self.mean)
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_size)
    def encode(self, x,tau):
        h1=x[...,-1,:]
        mu = [linear(x) for linear in self.fc2_mu]
        mu=torch.stack(mu,dim=-2)
#         x_mean=torch.mean(h1,dim=-2)
        logvar = [linear(x) for linear in self.fc2_logvar]
        logvar=torch.stack(logvar,dim=-2)
        class_prob=F.softmax(self.fc(F.relu(h1))/tau,dim=-1)
        class_res=torch.argmax(class_prob,dim=-1)
        return mu, logvar,class_prob,class_res

    def reparameterize(self, mu,logvar,class_res):
#         print(class_res.shape)
        class_res=class_res.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).expand(*mu.shape[:-2],1,mu.shape[-1])
        mu,logvar=torch.gather(mu,3,class_res),torch.gather(logvar,3,class_res)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std) 
        return (mu + eps * std).squeeze(-2)

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return self.fc4(h3)  # Output reconstructed data
    def loss(self,x_decode,temp,mu,logvar,class_res,class_prob):
        vae_loss=torch.mean((x_decode-temp)**2)*self.input_size
        diff=0.5*(self.mean.unsqueeze(0).unsqueeze(0).unsqueeze(0).detach()-mu)**2-0.5*logvar+0.5*(torch.exp(logvar)+0.1*self.mean.unsqueeze(0).unsqueeze(0).unsqueeze(0)-mu.detach())**2
#         kl_loss=torch.mean(torch.sum(class_prob.unsqueeze(-2).unsqueeze(-1)*diff,dim=[-2,-1]))
#         cat_loss=torch.mean(torch.sum(class_prob*torch.log(class_prob+1e-7),dim=-1))
        return vae_loss
        
    def forward(self, x,temp,tau=1):
        mu, logvar,class_prob,class_res = self.encode(x,tau)
        z = self.reparameterize(mu,logvar,class_res)
#         print(class_prob.shape)
        loss=self.loss(self.decode(z),temp,mu,logvar,class_res,class_prob)
#         batch_res,_=torch.mode(class_res,dim=-1)
        return self.decode(z), mu, logvar,class_res,class_prob,loss
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
class stack_witran(torch.nn.Module):
    def __init__(self,configs):
        super(stack_witran, self).__init__()
#         print(configs)
        self.C=configs['C']
        self.L=configs['L']
        self.hidden_size=configs['hidden_size']
        self.row=configs['row']
        self.col=configs['col']
        self.dropout=configs['dropout']
        self.num_layers=configs['num_layers']
        self.type=configs['type']
        self.class_num=configs['num']
        self.wit_list=nn.ModuleList()
        self.fc3 = nn.Linear(self.C,self.hidden_size)
        self.fc4= nn.Linear(self.hidden_size,self.hidden_size)
        self.relu=nn.ReLU()
        self.vae=class_VAE(self.C,self.hidden_size,self.hidden_size,self.class_num)
#         self.wit=WITRAN_2DPSGMU_Encoder(input_size=self.hidden_size,hidden_size=self.hidden_size,num_layers=self.num_layers,dropout=self.dropout,water_rows=self.row,
#                                          water_cols=self.col)
        for i in range(self.class_num):
            module=nn.ModuleList()
            
            module.append(WITRAN_2DPSGMU_Encoder(input_size=self.hidden_size,hidden_size=self.hidden_size,num_layers=self.num_layers,dropout=self.dropout,water_rows=self.row,
                                         water_cols=self.col))
            module.append(nn.Linear(self.hidden_size,1))
            module.append(nn.Linear(self.hidden_size,1))
            self.wit_list.append(module)
 
        
#         self.fc1=nn.Linear(self.hidden_size,1)
#         self.fc2=nn.Linear(self.hidden_size,1)
#         elif self.type=='class':
#             self.fc3=nn.Linear(2*self.hidden_size,2)
    
    def forward(self,x,pad_mask,signal=None,y=None):
        judge=x.ndim==4
        temp=x.clone()
        x=self.fc4(self.relu(self.fc3(x)))
        tau=1
        x_decode,mu,logvar,class_res,prob,loss=self.vae(x,temp,tau)
        if signal=='vae':
            return loss
        assert self.row*self.col==self.L and self.row>=self.col
#         x=x.reshape(-1,self.row,self.col,self.hidden_size)
#         y_pred=torch.zeros(B,N).to(x.device)
#         print(class_res)
        if x.ndim==4:
            B,N,L,C=x.shape
            ### only for softmax version
#             x=x.reshape(B*N,L,C)
        if self.training:
#             y_pred=torch.zeros(B,N,self.class_num).to(x.device)
#             for i in range(self.class_num):
#                 target,fc1,fc2=self.wit_list[i]
#                 _,row,col=target(x.reshape(-1,self.row,self.col,self.hidden_size))
#                 row,col=row[...,-1,:],col[...,-1,:]
#                 output=0.5*fc1(row)+0.5*fc2(col)
#                 output=output.squeeze(-1)
#                 y_pred[...,i]=output.reshape(-1,N)
#             y_pred=(y_pred*prob).sum(dim=-1).squeeze(-1)

            y_pred=torch.zeros(B,N).to(x.device)
            for i in range(self.class_num):
                class_mask=(class_res==i)
                target,fc1,fc2=self.wit_list[i]
    #             print(x[class_mask].shape)
    #             print(x[class_mask].reshape(-1,self.row,self.col,self.hidden_size).shape)
                _,row,col=target(x[class_mask].reshape(-1,self.row,self.col,self.hidden_size))
                row,col=row[...,-1,-1,:],col[...,-1,-1,:]
                output=0.5*fc1(row)+0.5*fc2(col)
                output=output.squeeze(-1)
    #             print(output.shape)
    #             print(output.device,y_pred.device)
                y_pred[class_mask]=output
            return y_pred,torch.tensor(0)
        else:
            
            y_pred=torch.zeros(B,N,self.class_num).to(x.device)
            for i in range(self.class_num):
#                 class_mask=(class_res==i)
                target,fc1,fc2=self.wit_list[i]
    #             print(x[class_mask].shape)
    #             print(x[class_mask].reshape(-1,self.row,self.col,self.hidden_size).shape)
                _,row,col=target(x.reshape(-1,self.row,self.col,self.hidden_size))
                row,col=row[...,-1,:],col[...,-1,:]
                output=0.5*fc1(row)+0.5*fc2(col)
                output=output.squeeze(-1)
    #             print(output.shape)
    #             print(output.device,y_pred.device)
                y_pred[...,i]=output.reshape(-1,N)
            y_pred=(y_pred*prob).sum(dim=-1).squeeze(-1)
            return y_pred
#             _,row,col=self.wit(x.reshape(-1,self.row,self.col,self.hidden_size))
#             row,col=row[...,-1,-1,:],col[...,-1,-1,:]
#             output=0.5*self.fc1(row)+0.5*self.fc2(col)
#             output=output.squeeze(-1)
#             return output.reshape(B,N)
            
#         x=self.fc(x)
# #         _,row,col=self.wit(x)
# #         row,col=row[...,-1,:],col[...,-1,:]
# #         output=torch.cat([row,col],dim=-1).squeeze()
# #         output=self.tanh(output)
#         _,row,col=self.wit1(x)
#         row,col=row[...,-1,:],col[...,-1,:]
#         output=torch.cat([row,col],dim=-1).squeeze()
#         output=self.tanh(output)
#         pos_pred=self.fc1(output)
#         _,row,col=self.wit2(x)
#         row,col=row[...,-1,:],col[...,-1,:]
#         output=torch.cat([row,col],dim=-1).squeeze()
#         output=self.tanh(output)
#         neg_pred=self.fc2(output)
#         _,row,col=self.wit3(x)
#         row,col=row[...,-1,:],col[...,-1,:]
#         output=torch.cat([row,col],dim=-1).squeeze()
#         output=self.tanh(output)
#         class_pred=self.sigmoid(self.fc3(output).squeeze())
#         softmax_output = F.softmax(class_pred, dim=-1)

#         y_pred=torch.cat((neg_pred,pos_pred),dim=-1)
#         y_pred=torch.mul(softmax_output,y_pred).mean(dim=-1)
        return y_pred.reshape(B,N)
#         if self.training:
#             if signal=='class':
#                 x=self.tanh(x)
#                 x=x.reshape(B*N,-1)
#                 pred=self.fc3(x)     
#                 pred=pred.reshape(B,N,2)
#             elif signal=='pos':
#                 _,row,col=self.wit1(x)
#                 row,col=row[...,-1,:],col[...,-1,:]
#                 output=torch.cat([row,col],dim=-1).squeeze()
#                 output=self.tanh(output)
#                 pred=self.fc1(output).squeeze()
#             else:
#                 _,row,col=self.wit2(x)
#                 row,col=row[...,-1,:],col[...,-1,:]
#                 output=torch.cat([row,col],dim=-1).squeeze()
#                 output=self.tanh(output)
#                 pred=self.fc2(output).squeeze()
#             return pred
#         else:
#             temp=x.clone()
#             temp=self.tanh(temp)
#             temp=temp.reshape(B*N,-1)
#             _,row1,col1=self.wit1(x)
#             _,row2,col2=self.wit2(x)
#             row1,col1=row1[...,-1,:],col1[...,-1,:]
#             row2,col2=row2[...,-1,:],col2[...,-1,:]
#             output1=self.tanh(torch.cat([row1,col1],dim=-1).squeeze())
#             output2=self.tanh(torch.cat([row2,col2],dim=-1).squeeze())
#             class_pred,pos_pred,neg_pred=self.fc3(temp),self.fc1(output1),self.fc2(output2)
#             softmax_output = F.softmax(class_pred, dim=-1)
# #             print(softmax_output.shape)
# #             print(pos_pred.shape)
# #             print(neg_pred.shape)
#             y_pred=torch.cat((neg_pred,pos_pred),dim=-1)
#             y_pred=torch.mul(softmax_output,y_pred).mean(dim=-1)
#             return y_pred.reshape(B,N)
            
#         return output
        
#         if self.type=='reg':
#             output=0.5*self.fc1(row)+0.5*self.fc2(col)
#             output=output.squeeze()
#             if judge:
#                 output=output.view(B,N)
#         elif self.type=='class':
#             output=torch.cat([row,col],dim=-1)
#             output = self.fc3(output)
#             output=output.squeeze()
#             if judge:
#                 output=output.view(B,N,-1)
#         row=row[:,:,-1:,:].expand(-1,-1,self.col,-1)
#         output = torch.cat([row,col], dim = -1)
#         output=output[:,-1,...].reshape(output.shape[0],-1)
#         output=self.fc(output).squeeze()
#         output=output.view(B,N,-1)
#         temp=torch.concat([row,col],dim=2)
#         temp=temp.reshape(temp.shape[0],-1)
#         temp=self.fc(temp).squeeze()
#         temp=temp.view(B,temp.shape[0])
        return (output)