import torch
import torch.nn as nn
import math
import torch.nn.functional as F
class WITRAN_2DPSGMU_Encoder(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, water_rows, water_cols,mog_iterations, res_mode='layer_yes'):
        super(WITRAN_2DPSGMU_Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.water_rows = water_rows
        self.water_cols = water_cols
        self.res_mode = res_mode
        # parameter of row cell
        self.W_first_layer=nn.Linear(input_size+2*hidden_size,6*hidden_size)
#         self.W_first_layer = torch.nn.Parameter(torch.empty(6 * hidden_size, input_size + 2 * hidden_size))
#         self.W_other_layer = torch.nn.Parameter(torch.empty(num_layers - 1, 6 * hidden_size, 4 * hidden_size))
        self.row_m=nn.Linear(self.hidden_size,self.hidden_size)
        self.col_m=nn.Linear(self.hidden_size,self.hidden_size)
        self.row_xm=nn.Linear(self.input_size,self.hidden_size)
        self.col_xm=nn.Linear(self.input_size,self.hidden_size)
#         self.B = torch.nn.Parameter(torch.empty(num_layers, 6 * hidden_size))
#         self.row_R=nn.Linear(self.input_size,self.hidden_size)
#         self.col_R=nn.Linear(self.input_size,self.hidden_size)
#         self.row_Q=nn.Linear(self.hidden_size,self.input_size)
#         self.col_Q=nn.Linear(self.hidden_size,self.input_size)
        self.mog_iterations=mog_iterations
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
    def mogrify(self,xt,ht,r,q):
        for i in range(1,self.mog_iterations+1):
            if (i % 2 == 0):
                ht = (2*torch.sigmoid(r(xt))) * ht
            else:
                xt = (2*torch.sigmoid(q(ht))) * xt
        return xt, ht
    def forward(self, input):
    
        batch_size,row,col,input_size=input.shape
        if col>row: # cols > rows
            input = input.permute(2, 0, 1, 3)
        else:
            input = input.permute(1, 0, 2, 3)
        Water2sea_slice_num, _, Original_slice_len, _ = input.shape
        Water2sea_slice_len = Water2sea_slice_num + Original_slice_len - 1
        hidden_slice_row = torch.zeros(Water2sea_slice_num * batch_size, self.hidden_size).to(input.device)
        hidden_slice_col = torch.zeros(Water2sea_slice_num * batch_size, self.hidden_size).to(input.device)
#         print(hidden_slice_row.shape)
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
#             B = self.B[layer, :]
            # start every for all slice
            output_all_slice_list = []
            for slice in range (Water2sea_slice_len):
                # gate generate
                x=a[:,slice,:]
#                 x,hidden_slice_row=self.mogrify(x,hidden_slice_row,self.row_R,self.row_Q)
#                 x,hidden_slice_col=self.mogrify(x,hidden_slice_col,self.col_R,self.col_Q)
                rowm=self.row_xm(x)+self.row_m(hidden_slice_row)
                colm=self.col_xm(x)+self.col_m(hidden_slice_col)
                gate=self.W_first_layer(torch.cat([rowm, colm, x],dim=-1))
                
#                 gate = self.linear(torch.cat([hidden_slice_row, hidden_slice_col, x], 
#                     dim = -1), W, B, batch_size, slice, Water2sea_slice_num)
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
        if col>row:
            return output_all_slice, hidden_col_all, hidden_row_all
        else:
            return output_all_slice, hidden_row_all, hidden_col_all
class MOGWITRAN(torch.nn.Module):
    def __init__(self,configs):
        super(MOGWITRAN, self).__init__()
        self.C=configs['C']
        self.L=configs['L']
        self.hidden_size=configs['hidden_size']
        self.row=configs['row']
        self.col=configs['col']
        self.dropout=configs['dropout']
        self.num_layers=configs['num_layers']
        self.mog_iterations=configs['mog_iterations']
        
        self.wit= WITRAN_2DPSGMU_Encoder(input_size=self.C,hidden_size=self.hidden_size,num_layers=self.num_layers,dropout=self.dropout,water_rows=self.row,
                                         water_cols=self.col,mog_iterations=self.mog_iterations)
#         self.fc=nn.Linear(self.num_layers*(self.row+self.col)*self.hidden_size,1)
#         self.fc=nn.Linear(self.col*2*self.hidden_size,self.L)
        self.fc1=nn.Linear(self.hidden_size*self.row,1)
        self.fc2=nn.Linear(self.hidden_size*self.col,1)
    def forward(self,x,pad_mask):
        if x.ndim==4:
            B,N,L,C=x.shape
            x=x.reshape(B*N,L,C)
        assert self.row*self.col==self.L and self.row>=self.col
        x=x.reshape(-1,self.row,self.col,self.C)
        _,row,col=self.wit(x)
#         row,col=row[...,-1,-1,:],col[...,-1,-1,:]
        row,col=row[:,-1],col[:,-1]
        row,col=row.reshape(B*N,-1),col.reshape(B*N,-1)
        output=0.5*self.fc1(row)+0.5*self.fc2(col)
        output=output.squeeze().view(B,N)
#         output=torch.cat([row,col],dim=-2)
#         output = self.fc(output).squeeze()
#         output=output.view(B,N,-1)
#         print(output.shape)
#         row=row[:,:,-1:,:].expand(-1,-1,self.col,-1)
#         output = torch.cat([row,col], dim = -1)

#         output=output[:,-1,...].reshape(output.shape[0],-1)
#         output=self.fc(output).squeeze()
#         output=output.view(B,N,-1)
#         output=output.view(B,output.shape[0])
#         temp=torch.concat([row,col],dim=2)
#         temp=temp.reshape(temp.shape[0],-1)
#         temp=self.fc(temp).squeeze()
#         temp=temp.view(B,temp.shape[0])
        if self.training:
            return output,0
        else:
            return output