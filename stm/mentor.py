import torch
import torch.nn as nn
import torch.nn.functional as F

class mentor(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=100):
        super(mentor, self).__init__()
        # 第一层 MLP
        self.mlp1 = nn.Linear(input_dim, hidden_dim)
        # 第二层 MLP
        self.mlp2 = nn.Linear(hidden_dim, hidden_dim)
        # 输出层
        self.output_layer = nn.Linear(hidden_dim, 1)
    
    def forward(self, loss, true_label):
        # 将损失和标签作为输入，拼接成一个向量
        x = torch.cat((loss, true_label), dim=-1)
        x=(x-torch.mean(x,dim=1))/torch.std(x,dim=1)
        # 第一层 MLP
        x = self.mlp1(x)
        # Tanh 激活函数
        x = torch.tanh(x)
        
        # 第二层 MLP
        x = self.mlp2(x)
        
        # 最后输出层经过 Sigmoid 激活函数得到权重
        weight = torch.sigmoid(self.output_layer(x))
        
        return weight