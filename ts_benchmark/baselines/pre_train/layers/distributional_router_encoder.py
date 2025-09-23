import torch
import torch.nn as nn


class encoder(nn.Module):
    def __init__(self, config):
        super(encoder, self).__init__()
        input_size = config.seq_len
        num_experts = config.num_experts
        encoder_hidden_size = config.hidden_size

        # TODO input_size换成（b*n*t）
        # self.distribution_fit = nn.Sequential(nn.Linear(638976, encoder_hidden_size, bias=False), nn.ReLU(),
        #                                       nn.Linear(encoder_hidden_size, num_experts, bias=False))
        self.distribution_fit = nn.Sequential(
                nn.Linear(1, encoder_hidden_size, bias=False),  # 👈 只处理每个 patch
                nn.ReLU(),
                nn.Linear(encoder_hidden_size, num_experts, bias=False)
            )

    def forward(self, x):
       # mean = torch.mean(x, dim=-1)
        out = self.distribution_fit(x)
        return out
