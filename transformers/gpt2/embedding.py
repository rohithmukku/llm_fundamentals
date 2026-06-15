import torch.nn as nn
import torch

# Reference: https://github.com/aju22/RoPE-PyTorch/blob/main/RoPE.ipynb

class RotaryEmbedding(nn.Module):
    def __init__(self, max_seq_len, d, base=10000):
        super().__init__()
        self.d = d
        self.base = base

        theta = 1./(self.base ** (torch.arange(0, self.d, 2)/ self.d))  # (d//2, 1)
        seq_idx = torch.arange(0, max_seq_len)                          # (t, 1)

        idx_theta = torch.outer(seq_idx, theta)                         # (t, d//2)
        idx_theta_2 = torch.cat([idx_theta, idx_theta], dim=-1)

        self.register_buffer('cos_table', torch.cos(idx_theta_2))
        self.register_buffer('sin_table', torch.sin(idx_theta_2))

    def forward(self, x):
        t = x.shape[-2]
        d = self.d

        rotate_x = torch.cat([-x[..., d//2:], x[..., :d//2]], dim=-1)

        cos_table = self.cos_table[:t, :]
        sin_table = self.sin_table[:t, :]

        embed = x * cos_table + rotate_x * sin_table

        return embed