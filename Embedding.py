import math
import torch
from torch import nn

class FixedEmbedding(nn.Module):
    def __init__(self, num_embed, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(num_embed, d_model).float()
        w.require_grad = False

        position = torch.arange(num_embed).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.embedding = nn.Embedding(num_embed, d_model)
        self.embedding.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        # no gradients
        return self.embedding(x).detach()

# 具体可以看一下Time-Series-Library中TemporalEmbedding和TimeFeatureEmbedding这两个类
# https://github.com/thuml/Time-Series-Library/blob/main/layers/Embed.py#L29
# todo: TimeFeatureEmbedding这个class的含义, freq_map value不同数字的含义是什么?
class CalendarEmbedding(nn.Module):
    def __init__(self, d_model, dropout, embed_type='fixed'):
        super(CalendarEmbedding, self).__init__()
        day_size = 31
        week_size = 7
        month_size = 12
        self.dropout = nn.Dropout(p=dropout)

        embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        self.day_embedding = embed(day_size, d_model)
        self.week_embedding = embed(week_size, d_model)
        self.month_embedding = embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        day_embedding = self.day_embedding(x[:, :, 0]) # day time embedding
        week_embedding = self.week_embedding(x[:, :, 1]) # week time embedding
        month_embedding = self.month_embedding(x[:, :, 2]) # month time embedding

        calendar_embedding = day_embedding + week_embedding + month_embedding

        return self.dropout(calendar_embedding)

class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6,
                    'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]

# Token embedding by using Conv1d
class TokenEmbedding(nn.Module):
    def __init__(self, input_size, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= '1.5.0' else 2
        self.tokenConv = nn.Conv1d(in_channels=input_size, out_channels=d_model,
                                   kernel_size=3, padding=padding, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x

# from https://github.com/thuml/Time-Series-Library/blob/main/layers/Embed.py
class DataEmbedding(nn.Module):
    def __init__(self, input_size, d_model, dropout, embed_type='fixed', freq='h'):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(input_size=input_size, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = CalendarEmbedding(d_model=d_model, dropout=dropout, embed_type=embed_type)\
            if embed_type != 'timeF' else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark=None):
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x_mark) + self.position_embedding(x)

        return self.dropout(x)