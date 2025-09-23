
import torch
from torch import nn
from einops import rearrange
from ts_benchmark.baselines.pre_train.model_plugin.layers.moe import SoftMoE
import numpy as np
from transformers.activations import ACT2FN
from ts_benchmark.baselines.pre_train.model_plugin.layers.star import STAR

class CosPositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, max_id=60 ):
        super().__init__()
        pe = torch.zeros(max_id, embedding_dim)
        position = torch.arange(0, max_id, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-np.log(10000.0) / embedding_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.cuda()

    def forward(self, inputs):
        # inputs 是你的ID张量
        return self.pe[inputs.long()]
    
class StatusEmbedding(nn.Module):
    def __init__(self, num_shared_experts, num_experts, seq_len, enc_in, K ,d_model,patch_len, stride):
        super().__init__()
        
        self.cpe = CosPositionalEncoding(d_model//2, max(len(enc_in)+1,sum(enc_in)) )
        self.soft_moe = SoftMoE(num_shared_experts, num_experts, seq_len, enc_in, K, hidden_size=d_model)
        self.temp_emb = nn.Linear(d_model*patch_len, d_model)
        self.patch_len = patch_len
        self.stride = stride
        self.star = STAR(d_model, d_model)
        
    def forward(self, x):
        b,l,n=x.shape
        value_identy = self.cpe(rearrange(x, 'b l n  -> (b l) n'))
        channel_identy = self.cpe(torch.arange(x.size(-1))).unsqueeze(0).expand(value_identy.size(0),-1,-1)

        x = torch.cat((value_identy,channel_identy),dim=-1)

        x, L_importance, weight = self.soft_moe(x)
        x = rearrange(x, '(b l) n d -> b l n d', b=b)
        x_patch = x.unfold(dimension=1, size=self.patch_len, step=self.stride)
        x_patch = rearrange(x_patch,'batch_size num_patch num_vars d_model patch_len -> batch_size num_vars num_patch (patch_len d_model)')
        
        # # 线性映射：变成 [B, L, d_model]
        status_token = self.temp_emb(x_patch)  # [B,n,num_patch, d_model]
        status_token = self.star(status_token)
        return x, status_token,  L_importance , weight



# class TimeMoeInputEmbedding(nn.Module):
#     """
#     Use a mlp layer to embedding the time-series.
#     """

#     def __init__(self, input_size, hidden_size ):
#         super().__init__()

#         self.emb_layer = nn.Linear(input_size, hidden_size, bias=False)
#         self.gate_layer = nn.Linear(input_size, hidden_size, bias=False)
#         self.act_fn = ACT2FN['silu']

#     def forward(self, x):
#         b, t, c = x.shape
#         x = rearrange(x, 'b t c -> (b c) t 1')
#         emb = self.act_fn(self.gate_layer(x)) * self.emb_layer(x)
#         emb = rearrange(emb, '(b c) t d-> b t c d', b=b)
#         return emb