import torch.nn as nn
import torch
import torch.nn.functional as F
from ..Embed.Embed import TokenEmbedding,TemporalEmbedding,FixedEmbedding
class TRA(nn.Module):
    def __init__(self, input_size,num_states=5, hidden_size=256, tau=1.0, src_info="LR_TPE"):
        super(TRA,self).__init__()

        self.num_states = num_states
        self.tau = tau
        self.src_info = src_info
        if num_states > 1:
#             self.router = nn.LSTM(
#                 input_size=num_states,
#                 hidden_size=hidden_size,
#                 num_layers=1,
#                 batch_first=True,
#             )
            self.fc = nn.Linear(hidden_size + input_size, num_states)
        self.embed=TemporalEmbedding(embed_type='naive',d_model=hidden_size)
        self.predictors = nn.Linear(input_size, num_states)
        self.relu=nn.ReLU()

    def forward(self, hidden, hist_loss):
#         print('hidden',hidden.shape)
        preds = self.predictors(hidden)
        
        if self.num_states == 1:
            return preds.squeeze(-1), preds, None

        # information type
#         router_out, _ = self.router(hist_loss)
#         print(router_out.shape)
#         print('loss',hist_loss.shape)
        if "LR" in self.src_info:
            latent_representation = hidden
        else:
            latent_representation = torch.randn(hidden.shape).to(hidden)
        if "TPE" in self.src_info:
            temporal_pred_error = self.embed(hist_loss)
        
        else:
            temporal_pred_error = torch.randn(router_out[:, -1].shape).to(hidden)
        temporal_pred_error=temporal_pred_error.reshape(-1,temporal_pred_error.shape[-1])
#         print('latent',latent_representation.shape)
        out = self.fc(self.relu(torch.cat([temporal_pred_error, latent_representation], dim=-1)))
        prob = F.gumbel_softmax(out, dim=-1, tau=self.tau, hard=False)

        if self.training:
            final_pred = (preds * prob).sum(dim=-1)
        else:
            final_pred = preds[range(len(preds)), prob.argmax(dim=-1)]

        return final_pred, preds, prob