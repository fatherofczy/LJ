import torch.nn as nn
import torch
from torch.nn import init
import torch.nn.functional as F
# from torch._jit_internal import weak_module, weak_script_method
from torch.nn.parameter import Parameter
import math
from ..MTSMixer.layers.Invertible import RevIN

# @weak_module
class Maxout(nn.Module):
    __constants__ = ['bias']

    def __init__(self, in_features, out_features, pieces=2, bias=True):
        super(Maxout, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.pieces = pieces
        self.weight = Parameter(torch.Tensor(pieces, in_features,out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(pieces, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

#     @weak_script_method
    def forward(self, input):
        output=torch.einsum('...i,aij->...aj', input, self.weight)+self.bias
        output = torch.max(output, dim=-2)[0]
    
        return output

class TSBatchNorm2d(nn.Module):

    def __init__(self):
        super(TSBatchNorm2d, self).__init__()
        self.bn = nn.BatchNorm2d(num_features=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x: (batch_size, time, features)

        # Reshape input_data to (batch_size, 1, timepoints, features)
        x = x.unsqueeze(1)

        # Forward pass
        output = self.bn(x)

        # Reshape the output back to (batch_size, timepoints, features)
        output = output.squeeze(1)
        return output


class TSTimeMixingResBlock(nn.Module):

    def __init__(self, width_time: int,width_time_hidden:int, dropout: float):
        super(TSTimeMixingResBlock, self).__init__()
        self.norm = TSBatchNorm2d()

        self.lin1 = nn.Linear(in_features=width_time, out_features=width_time_hidden)
        self.lin2 = nn.Linear(in_features=width_time_hidden, out_features=width_time)
#         self.lin=Maxout(in_features=width_time, out_features=width_time)
        self.dropout = nn.Dropout(p=dropout)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x: (batch_size, time, features)
        y = self.norm(x)
        
        # Now rotate such that shape is (batch_size, features, time)
        y = torch.transpose(y, 1, 2)
        
        # Apply MLP to time dimension
        y = self.lin1(y)
        y = self.act(y)
        y = self.lin2(y)
        # Rotate back such that shape is (batch_size, time, features)
        y = torch.transpose(y, 1, 2)

        # Dropout
        y = self.dropout(y)
                
        # Add residual connection
        return x + y


class TSFeatMixingResBlock(nn.Module):

    def __init__(self, width_feats: int, width_feats_hidden: int, dropout: float):
        super(TSFeatMixingResBlock, self).__init__()
        self.norm = TSBatchNorm2d()

        self.lin_1 = nn.Linear(in_features=width_feats, out_features=width_feats_hidden)
        self.lin_2 = nn.Linear(in_features=width_feats_hidden, out_features=width_feats)
#         self.lin_1=Maxout(in_features=width_feats, out_features=width_feats_hidden)
#         self.lin_2=Maxout(in_features=width_feats_hidden, out_features=width_feats)
        self.dropout_1 = nn.Dropout(p=dropout)
        self.dropout_2 = nn.Dropout(p=dropout)
        self.act = nn.ReLU()


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x: (batch_size, time, features)
        y = self.norm(x)
        
        # Apply MLP to feat dimension
        y = self.lin_1(y)
        y = self.act(y)
#         y = self.dropout_1(y)
        y = self.lin_2(y)
        y = self.dropout_2(y)
                
        # Add residual connection
        return x + y


class TSMixingLayer(nn.Module):

    def __init__(self, input_length: int, no_feats:int, time_mixing_hidden_channels:int,feat_mixing_hidden_channels: int, dropout: float):
        super(TSMixingLayer, self).__init__()
     
        self.time_mixing = TSTimeMixingResBlock(width_time=input_length, width_time_hidden=time_mixing_hidden_channels,dropout=dropout)
        self.feat_mixing = TSFeatMixingResBlock(width_feats=no_feats, width_feats_hidden=feat_mixing_hidden_channels, dropout=dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Input x: (batch_size, time, features)
        x = self.time_mixing(x)
        x = self.feat_mixing(x)
        return x


class TSTemporalProjection(nn.Module):

    def __init__(self, input_length: int, forecast_length: int,feature_num:int):
        super(TSTemporalProjection, self).__init__()
        
#         self.lin = nn.Linear(in_features=input_length, out_features=forecast_length)
        self.out = nn.Linear(in_features=feature_num*input_length, out_features=forecast_length)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        # Input x: (batch_size, time, features)
        x=x.view(*x.shape[:-2],x.shape[-2]*x.shape[-1])
        # Now rotate such that shape is (batch_size, features, time=input_length)
#         x = torch.transpose(x, 1, 2)

        # Apply linear projection -> shape is (batch_size, features, time=forecast_length)
        

        # Rotate back such that shape is (batch_size, time=forecast_length, features)
#         x = torch.transpose(x, 1, 2)
#         x=self.out(x).squeeze()
        x = self.out(x).squeeze()
        
        return x


class TSMixerModel(nn.Module):

    def __init__(self, configs):
        super(TSMixerModel, self).__init__()
        L=configs['L']
        C=configs['C']
        forecast_length=configs['forecast_length']
        time_mixing_hidden_channels=configs['time_mixing_hidden_channels']
        feat_mixing_hidden_channels=configs['feat_mixing_hidden_channels']
        dropout=configs['dropout']
        no_feats=configs['no_feats']
        no_mixer_layers=configs['no_mixer_layers']
        self.lin=nn.Linear(in_features=C,out_features=no_feats)
        self.rev=RevIN(C) if configs['rev'] else None
        
        mixer_layers = []
        for _ in range(no_mixer_layers):
            mixer_layers.append(TSMixingLayer(input_length=L, no_feats=no_feats, time_mixing_hidden_channels=time_mixing_hidden_channels,feat_mixing_hidden_channels=feat_mixing_hidden_channels, dropout=dropout))
        self.mixer_layers = nn.ModuleList(mixer_layers)
  
        self.feat_proj=nn.Linear(no_feats,C)
        self.temp_proj = TSTemporalProjection(input_length=L, 
                                              forecast_length=forecast_length,
                                             feature_num=C)
   
    def forward(self, x: torch.Tensor, pad_mask:torch.Tensor) -> torch.Tensor:
        B,N,L,C=x.shape
        x = x.view(-1,L,C)
        x = self.rev(x,'norm') if self.rev else x
        x = self.lin(x)
        # Input x: (batch_size, time, features)
        for mixer_layer in self.mixer_layers:
            x = mixer_layer(x)
        
        # Apply temporal projection -> shape is (batch_size, time=forecast_length, features)
        x=self.rev(self.feat_proj(x),'denorm') if self.rev else x
        x = self.temp_proj(x)
        
        x=x.view(-1,N)
        return x


