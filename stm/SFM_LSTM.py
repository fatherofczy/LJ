
import math
import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np

class SFM_LSTM(nn.Module):
    def __init__(self, configs):
        super(SFM_LSTM, self).__init__()
        self.L=configs['L']
        self.input_size = configs['C']
        self.hidden_size = configs['hidden_size']
        self.freq_size = configs['freq_size']
        self.output_size = configs['output_size']
        self.sigmoid = nn.Sigmoid()
        self.drop=nn.Dropout(p=0.2)
        self.tanh = nn.Tanh()
        self.output = nn.Linear(self.hidden_size,self.output_size)
        self.mog_iterations=5
#         self.omega = torch.tensor(2 * np.pi * np.arange(1, self.freq_size + 1) / self.freq_size).float()

        # Define linear layers
        self.i_linear = nn.Linear(self.input_size + self.hidden_size, self.hidden_size)
        self.g_linear = nn.Linear(self.input_size + self.hidden_size, self.hidden_size)
        self.o_linear = nn.Linear(self.input_size + 2 * self.hidden_size, self.hidden_size)
        self.omega_linear=nn.Linear(self.input_size + self.hidden_size, self.freq_size)
        self.fre_linear = nn.Linear(self.input_size + self.hidden_size, self.freq_size)
        self.ste_linear = nn.Linear(self.input_size +self. hidden_size, self.hidden_size)
        self.Q = nn.Parameter(torch.Tensor(self.hidden_size,self.input_size))
        self.R =nn.Parameter(torch.Tensor(self.input_size,self.hidden_size))
        self.a_linear = nn.Linear(self.freq_size, 1)
        
        
        self.reset_parameters()
    def mogrify(self,xt,ht):
        for i in range(1,self.mog_iterations+1):
            if (i % 2 == 0):
                ht = (2*torch.sigmoid(xt @ self.R)) * ht
            else:
                xt = (2*torch.sigmoid(ht @ self.Q)) * xt
        return xt, ht
    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            if weight.dim() > 1:
                nn.init.uniform_(weight, -stdv, stdv)
    def init_state(self,batch_size):
        h = torch.zeros(batch_size, self.hidden_size)
        c = torch.zeros(batch_size, self.hidden_size)
        re_s = torch.zeros(batch_size,self.hidden_size, self.freq_size)
        im_s = torch.zeros(batch_size,self.hidden_size, self.freq_size)
#         omega=torch.zeros(batch_size, self.freq_size)
        time = torch.ones(1)
        return h, c, re_s, im_s,time
    def forward(self, input,pad_mask=None):
        b=input.shape[0]
        input=self.drop(input)
        input=input.reshape(-1,*input.shape[2:])
        batch_size, seq_len, _ = input.size()
        h, c, re_s, im_s, time = self.init_state(batch_size)
        h,c,re_s,im_s=h.to(input),c.to(input),re_s.to(input),im_s.to(input)
        outputs = []

        for t in range(seq_len):
            input_t = input[:, t, :]
            input_t,h=self.mogrify(input_t,h)
            combined_ih = torch.cat((input_t, h), dim=-1)
            omega_t = self.omega_linear(combined_ih)  # Linear combination
            omega_t = 2 * np.pi * torch.sigmoid(omega_t)  # Scale and apply sigmoid to ensure values are within [0, 2*pi]

            i_t = self.sigmoid(self.i_linear(combined_ih))
            c_hat_t = self.sigmoid(self.g_linear(combined_ih))

            f_ste = self.sigmoid(self.ste_linear(combined_ih))  # belong R^D
            f_fre = self.sigmoid(self.fre_linear(combined_ih))  # belong R^K
            f_t = torch.matmul(f_ste.view(-1, self.hidden_size, 1), f_fre.view(-1, 1, self.freq_size))
            
            re_s = torch.mul(f_t, re_s) + torch.matmul(torch.mul(i_t, c_hat_t).unsqueeze(-1), torch.cos(omega_t).unsqueeze(1))
            im_s = torch.mul(f_t, im_s) + torch.matmul(torch.mul(i_t, c_hat_t).unsqueeze(-1), torch.sin(omega_t).unsqueeze(1))

            a_t = torch.sqrt(re_s**2 + im_s**2)
            c_t = self.tanh(self.a_linear(a_t)).squeeze()
            combined_oh = torch.cat((input_t, h, c_t), dim=-1)
            o_t = self.sigmoid(self.o_linear(combined_oh))
            h = torch.mul(o_t, self.tanh(c_t))

            outputs.append(h)
            time += 1
        

        outputs = self.output(outputs[-1]).squeeze()  # (batch_size, seq_len, output_size)
        return outputs.reshape(b,-1)

