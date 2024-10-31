
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class WITRAN_2DPSGMU_Encoder_ConvLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, water_rows, water_cols, C,H,W,kernel_size=3, res_mode='layer_yes'):
        super(WITRAN_2DPSGMU_Encoder_ConvLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.water_rows = water_rows
        self.water_cols = water_cols
        self.kernel_size = kernel_size
        self.res_mode = res_mode
        self.C=C
        self.H=H
        self.W=W
        self.shape=[self.C,self.H,self.W]
        # ConvLSTM Layers
        
        assert self.C*self.H*self.W==self.input_size
        self.conv_lstm_layers = nn.ModuleList()
        for i in range(num_layers):
            in_channels = self.C*3
            self.conv_lstm_layers.append(ConvLSTMCell(input_dim=self.C, hidden_dim=self.C, kernel_size=self.kernel_size))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            weight.data.uniform_(-stdv, +stdv)

    def forward(self, input):
        batch_size,row,col,input_size=input.shape
        flag=col>row
        if col>row: # cols > rows
            input = input.permute(2, 0, 1, 3)
        else:
            input = input.permute(1, 0, 2, 3)
        Water2sea_slice_num, _, Original_slice_len, _ = input.shape
        Water2sea_slice_len = Water2sea_slice_num + Original_slice_len - 1
        hidden_slice_row = torch.zeros(Water2sea_slice_num * batch_size, *self.shape).to(input.device)
        hidden_slice_col = torch.zeros(Water2sea_slice_num * batch_size, *self.shape).to(input.device)
#         print(hidden_slice_row.shape)
        input_transfer = torch.zeros(Water2sea_slice_num, batch_size, Water2sea_slice_len, input_size).to(input.device)
#         print(input_transfer.shape)
        for r in range(Water2sea_slice_num):
            input_transfer[r, :, r:r+Original_slice_len, :] = input[r, :, :, :]
        hidden_row_all_list = []
        hidden_col_all_list = []
        for layer in range(self.num_layers):
            if layer == 0:
                a = input_transfer.reshape(Water2sea_slice_num * batch_size, Water2sea_slice_len, *self.shape)
                
            else:
                a = F.dropout(output_all_slice, self.dropout, self.training)
                if layer == 1:
                    layer0_output = a
                
                hidden_slice_row = hidden_slice_row * 0
                hidden_slice_col = hidden_slice_col * 0
#             B = self.B[layer, :]
            # start every for all slice
            output_all_slice_list = []
            hidden_slice_row = hidden_slice_row * 0
            hidden_slice_col = hidden_slice_col * 0

            # Get ConvLSTMCell layer
            conv_lstm = self.conv_lstm_layers[layer]

            # Process each slice of the input sequentially
            output_all_slice_list = []
            for slice in range(Water2sea_slice_len):
                # Perform ConvLSTM operation on each slice
                hidden_slice_row,hidden_slice_col = conv_lstm(a[:, slice, ...], [hidden_slice_row, hidden_slice_col])
               
                output_slice = torch.cat([hidden_slice_row, hidden_slice_col], dim = -3)
                # save output
                output_all_slice_list.append(output_slice)
                # save row hidden
                if slice >= Original_slice_len - 1:
                    need_save_row_loc = slice - Original_slice_len + 1
                    hidden_row_all_list.append(
                        hidden_slice_row[need_save_row_loc*batch_size:(need_save_row_loc+1)*batch_size, ...])
                # save col hidden
                if slice >= Water2sea_slice_num - 1:
                    hidden_col_all_list.append(
                        hidden_slice_col[(Water2sea_slice_num-1)*batch_size:, ...])
                # hidden transfer
                hidden_slice_col = torch.roll(hidden_slice_col, shifts=batch_size, dims = 0)
#                 print(hidden_slice_col.shape)
            if self.res_mode == 'layer_res' and layer >= 1: # layer-res
                output_all_slice = torch.stack(output_all_slice_list, dim = 1) + layer0_output
            else:
                output_all_slice = torch.stack(output_all_slice_list, dim = 1)
        hidden_row_all = torch.stack(hidden_row_all_list, dim = 1)
        hidden_col_all = torch.stack(hidden_col_all_list, dim = 1)
        hidden_row_all = hidden_row_all.reshape(batch_size, self.num_layers, Water2sea_slice_num, *self.shape)
        hidden_col_all = hidden_col_all.reshape(batch_size, self.num_layers, Original_slice_len, *self.shape)
        if flag == 1:
            return output_all_slice, hidden_col_all, hidden_row_all
        else:
            return output_all_slice, hidden_row_all, hidden_col_all


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True):
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        self.bias = bias

        # Convolutional gate layers
        self.conv = nn.Conv2d(in_channels=self.input_dim + 2*self.hidden_dim,
                              out_channels=6 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        hidden_slice_row,hidden_slice_col = cur_state
    
        # Concatenate input and hidden state
#         print(hidden_slice_row.shape)
        combined = torch.cat([input_tensor,hidden_slice_row,hidden_slice_col], dim=-3)  # concatenate along channel axis
#         print(combined.shape)
        combined_conv = self.conv(combined)
        sigmod_gate, tanh_gate = torch.split(combined_conv, 4 * self.hidden_dim, dim = -3)
        sigmod_gate = torch.sigmoid(sigmod_gate)
        tanh_gate = torch.tanh(tanh_gate)
        update_gate_row, output_gate_row, update_gate_col, output_gate_col = sigmod_gate.chunk(4, dim = -3)
        input_gate_row, input_gate_col = tanh_gate.chunk(2, dim = -3)
        hidden_slice_row = torch.tanh(
            (1-update_gate_row)*hidden_slice_row + update_gate_row*input_gate_row) * output_gate_row
        hidden_slice_col = torch.tanh(
            (1-update_gate_col)*hidden_slice_col + update_gate_col*input_gate_col) * output_gate_col

        return hidden_slice_row,hidden_slice_col

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))
class conv_witran(nn.Module):
    def __init__(self,configs):
        super(conv_witran, self).__init__()
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
        self.C,self.H,self.W=2,16,8
        self.wit=WITRAN_2DPSGMU_Encoder_ConvLSTM(input_size=self.hidden_size,hidden_size=self.hidden_size,num_layers=self.num_layers,dropout=self.dropout,water_rows=self.row,
                                         water_cols=self.col,C=self.C,H=self.H,W=self.W)

        
        self.fc1=nn.Linear(self.hidden_size,1)
        self.fc2=nn.Linear(self.hidden_size,1)
#         elif self.type=='class':
#             self.fc3=nn.Linear(2*self.hidden_size,2)
    
    def forward(self,x,pad_mask,signal=None,y=None):
        judge=x.ndim==4
        temp=x.clone()
        x=self.fc4(self.relu(self.fc3(x)))
        assert self.row*self.col==self.L and self.row>=self.col
#         x=x.reshape(-1,self.row,self.col,self.hidden_size)
#         y_pred=torch.zeros(B,N).to(x.device)
#         print(class_res)
        if x.ndim==4:
            B,N,L,C=x.shape
            x=x.reshape(B*N,L,C)
        
        _,row,col=self.wit(x.reshape(-1,self.row,self.col,self.hidden_size))
        row,col=row[...,-1,:,:,:],col[...,-1,:,:,:]
#         print(row.shape)
        row,col=row.reshape(*row.shape[:-3],-1),col.reshape(*col.shape[:-3],-1)
        output=0.5*self.fc1(row)+0.5*self.fc2(col)
        output=output.squeeze(-1)
        if self.training:
            return output.reshape(B,N),torch.tensor(0)
        else:
            return output.reshape(B,N)
       