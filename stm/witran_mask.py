import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from ..BaseEncoder.layers import MHA
class WITRAN_2DPSGMU_Encoder(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, water_rows, water_cols, res_mode='layer_yes'):
        super(WITRAN_2DPSGMU_Encoder, self).__init__()
        self.mha=MHA(input_size,2,0.2)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.water_rows = water_rows
        self.water_cols = water_cols
        self.res_mode = res_mode
        self.attn_mask=torch.load('/dfs/data/attn_mask.pt')
        # parameter of row cell
        self.W_first_layer = torch.nn.Parameter(torch.empty(6 * hidden_size, 2*input_size + 2 * hidden_size))
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
        self.attn_mask=self.attn_mask.to(input.device)
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
                x=a[:,slice,:]
                y=x.reshape(Water2sea_slice_num,batch_size,-1)
                y=self.mha(y,y,y,attn_mask=self.attn_mask)
                y=y.reshape(-1,y.shape[-1])
                gate = self.linear(torch.cat([hidden_slice_row, hidden_slice_col,x,y], 
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
class witran_mask(torch.nn.Module):
    def __init__(self,configs):
        super(witran_mask, self).__init__()
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
        self.wit= WITRAN_2DPSGMU_Encoder(input_size=self.C,hidden_size=self.hidden_size,num_layers=self.num_layers,dropout=self.dropout,water_rows=self.row,
                                         water_cols=self.col)
        self.fc1=nn.Linear(2*self.hidden_size,100)
        self.fc2=nn.Linear(100,100)
        self.relu=nn.ReLU()
#         self.fc3=nn.Linear(2*self.hidden_size,2)
        self.layernorm=nn.LayerNorm(100)
        self.fc3=nn.Linear(2*self.hidden_size,100)
        self.out=nn.Linear(100,1)
        self.tanh=nn.Tanh()
    def forward(self,x,pad_mask,signal=None):
        judge=x.ndim==4
        if x.ndim==4:
            B,N,L,C=x.shape
            x=x.reshape(B*N,L,C)
        assert self.row*self.col==self.L and self.row>=self.col
        x=x.reshape(-1,self.row,self.col,self.C)
        _,row,col=self.wit(x)
        row,col=row[...,-1,:],col[...,-1,:]
        output=torch.cat([row,col],dim=-1)
        mask=self.fc2(self.relu(self.fc1(output)))
        output=self.tanh(self.layernorm(self.fc3(output)))
        output=output*mask
        output=self.out(output)
        output=output.squeeze(-1)
        if judge:
            output=output.view(B,N)
        if self.training:
            return output,torch.tensor(0)
        else:
            return output