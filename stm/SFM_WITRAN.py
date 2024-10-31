import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from torch.autograd import Variable
class SFM_WITRAN_2DPSGMU_Encoder(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, water_rows, water_cols,freq_size, res_mode='layer_yes'):
        super(SFM_WITRAN_2DPSGMU_Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = nn.Dropout(dropout)
        self.water_rows = water_rows
        self.water_cols = water_cols
        self.res_mode = res_mode
        # parameter of row cell
        self.freq_size=freq_size
        self.W_first_layer = torch.nn.Parameter(torch.empty(6 * hidden_size, input_size + 2 * hidden_size))
        self.W_freq=torch.nn.Parameter(torch.empty(2 * self.freq_size, input_size + 2 * hidden_size))
#         self.W_other_layer = torch.nn.Parameter(torch.empty(num_layers - 1, 6 * hidden_size, 4 * hidden_size))
        self.B = torch.nn.Parameter(torch.empty(num_layers, 6 * hidden_size))
        self.B_freq = torch.nn.Parameter(torch.empty(num_layers, 2 * self.freq_size))
        self.row_c=nn.Linear(self.freq_size,1)
        self.col_c=nn.Linear(self.freq_size,1)
        self.row_o=nn.Linear(self.hidden_size,self.hidden_size)
        self.col_o=nn.Linear(self.hidden_size,self.hidden_size)
        self.reset_parameters()
        self.omega = torch.nn.Parameter(2*torch.pi*torch.arange(1,self.freq_size+1)/self.freq_size).float().unsqueeze(0)
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
        self.omega=self.omega.to(input.device)
        batch_size,row,col,input_size=input.shape
        if col>row: # cols > rows
            input = input.permute(2, 0, 1, 3)
        else:
            input = input.permute(1, 0, 2, 3)
        Water2sea_slice_num, _, Original_slice_len, _ = input.shape
        Water2sea_slice_len = Water2sea_slice_num + Original_slice_len - 1
        hidden_slice_row = torch.zeros(Water2sea_slice_num * batch_size, self.hidden_size).to(input.device)
        hidden_slice_col = torch.zeros(Water2sea_slice_num * batch_size, self.hidden_size).to(input.device)
        hidden_slice_row_re=0.1*torch.ones(Water2sea_slice_num * batch_size, self.hidden_size,self.freq_size).to(input.device)
        hidden_slice_row_im=0.1*torch.ones(Water2sea_slice_num * batch_size, self.hidden_size,self.freq_size).to(input.device)
        hidden_slice_col_re=0.1*torch.ones(Water2sea_slice_num * batch_size, self.hidden_size,self.freq_size).to(input.device)
        hidden_slice_col_im=0.1*torch.ones(Water2sea_slice_num * batch_size, self.hidden_size,self.freq_size).to(input.device)
        
#         print(hidden_slice_row.shape)
        input_transfer = torch.zeros(Water2sea_slice_num, batch_size, Water2sea_slice_len, input_size).to(input.device)
        time=torch.zeros(Water2sea_slice_num,batch_size,Water2sea_slice_len).to(input.device)
#         print(input_transfer.shape)
        for r in range(Water2sea_slice_num):
            input_transfer[r, :, r:r+Original_slice_len, :] = input[r, :, :, :]
            time[r,:,r:r+Original_slice_len]=torch.arange(r*Original_slice_len,(r+1)*Original_slice_len)
        time=time.reshape(Water2sea_slice_num * batch_size,Water2sea_slice_len)
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
            B_freq=self.B_freq[layer,:]
            # start every for all slice
            output_all_slice_list = []
            for slice in range (Water2sea_slice_len):
                # gate generate
                
                x=torch.cat([hidden_slice_row, hidden_slice_col, a[:, slice, :]], dim = -1)
#                 if slice%4==0:
#                     x=self.dropout(x)
                gate = self.linear(x, W, B, batch_size, slice, Water2sea_slice_num)
                # gate
                freq_gate=self.linear(torch.cat([hidden_slice_row, hidden_slice_col, a[:, slice, :]], 
                    dim = -1), self.W_freq, B_freq, batch_size, slice, Water2sea_slice_num)
                freq_gate=torch.sigmoid(freq_gate)
                row_freq,col_freq=torch.split(freq_gate,self.freq_size,dim=-1)
                
                sigmod_gate, tanh_gate = torch.split(gate, 4 * self.hidden_size, dim = -1)
                sigmod_gate = torch.sigmoid(sigmod_gate)
                tanh_gate = torch.tanh(tanh_gate)
                input_gate_row, input_gate_col = tanh_gate.chunk(2, dim = -1)
                update_gate_row, output_gate_row, update_gate_col, output_gate_col = sigmod_gate.chunk(4, dim = -1)
                row_ft=torch.matmul(update_gate_row.unsqueeze(-1),row_freq.unsqueeze(-2))
                col_ft=torch.matmul(update_gate_col.unsqueeze(-1),col_freq.unsqueeze(-2))
                cur_time=time[:,slice].unsqueeze(1)
                hidden_slice_row_re=torch.mul(row_ft,hidden_slice_row_re)+torch.matmul(torch.mul(output_gate_row,input_gate_row).unsqueeze(-1),\
                                                                                       torch.cos(torch.mul(cur_time,self.omega)).unsqueeze(1))
                hidden_slice_row_im=torch.mul(row_ft,hidden_slice_row_im)+torch.matmul(torch.mul(output_gate_row,input_gate_row).unsqueeze(-1),\
                                                                                       torch.sin(torch.mul(cur_time,self.omega)).unsqueeze(1))
                hidden_slice_col_re=torch.mul(col_ft,hidden_slice_col_re)+torch.matmul(torch.mul(output_gate_col,input_gate_col).unsqueeze(-1),\
                                                                                       torch.cos(torch.mul(cur_time,self.omega)).unsqueeze(1))
                hidden_slice_col_im=torch.mul(col_ft,hidden_slice_col_im)+torch.matmul(torch.mul(output_gate_col,input_gate_col).unsqueeze(-1),\
                                                                                       torch.sin(torch.mul(cur_time,self.omega)).unsqueeze(1))
                
                # gate effect
                row_a=hidden_slice_row_re**2+hidden_slice_row_im**2
                col_a=hidden_slice_col_re**2+hidden_slice_col_im**2
                row_c=torch.tanh(self.row_c(row_a)).squeeze(-1)
                col_c=torch.tanh(self.col_c(col_a)).squeeze(-1)
                
                hidden_slice_row = torch.sigmoid(
                    (1-update_gate_row)*hidden_slice_row + update_gate_row*input_gate_row+self.row_o(row_c)) * row_c
                hidden_slice_col = torch.sigmoid(
                    (1-update_gate_col)*hidden_slice_col + update_gate_col*input_gate_col+self.col_o(col_c)) * col_c
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
                time=torch.roll(time,shifts=batch_size,dims=0)
                hidden_slice_col_re=torch.roll(hidden_slice_col_re,shifts=batch_size,dims=0)
                hidden_slice_col_im=torch.roll(hidden_slice_col_im,shifts=batch_size,dims=0)
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
class SFM_WITRAN(torch.nn.Module):
    def __init__(self,configs):
        super(SFM_WITRAN, self).__init__()
        self.C=configs['C']
        self.L=configs['L']
        self.hidden_size=configs['hidden_size']
        self.row=configs['row']
        self.col=configs['col']
        self.dropout=configs['dropout']
        self.num_layers=configs['num_layers']
        self.freq_size=configs['freq_size']
        self.wit= SFM_WITRAN_2DPSGMU_Encoder(input_size=self.C,hidden_size=self.hidden_size,num_layers=self.num_layers,dropout=self.dropout,water_rows=self.row,
                                         water_cols=self.col,freq_size=self.freq_size)
        self.dropout = nn.Dropout(self.dropout)
        self.fc1=nn.Linear(self.hidden_size,1)
        self.fc2=nn.Linear(self.hidden_size,1)
    def forward(self,x,pad_mask):
        if x.ndim==4:
            B,N,L,C=x.shape
            x=x.reshape(B*N,L,C)
        assert self.row*self.col==self.L and self.row>=self.col
        x=x.reshape(-1,self.row,self.col,self.C)
        _,row,col=self.wit(x)
        row,col=row[...,-1,:],col[...,-1,:]
        row,col=self.dropout(row),self.dropout(col)
        output=0.5*self.fc1(row)+0.5*self.fc2(col)
        output=output.view(B,N)
        return output