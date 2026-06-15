from .attention import CausalSelfAttention
from .config import ModelConfig
import torch.nn as nn
import torch

class Block(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embed, bias=config.bias)
        self.attention = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embed, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))

        return x

class MLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_embed = config.n_embed
        self.linear1 = nn.Linear(config.n_embed, 4 * config.n_embed, bias=config.bias)
        self.linear2 = nn.Linear(4 * config.n_embed, config.n_embed)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.linear1(x)
        x = self.act(x)
        x = self.linear2(x)
        x = self.dropout(x)

        return x

def test_block():
    config = ModelConfig(n_embed=128, n_heads=4, max_seq_len=32, 
                         dropout=0.0, bias=True, use_torch_dot_product=True)
    block = Block(config)
    block.eval()
    
    x = torch.randn(2, 16, 128)
    out = block(x)
    
    print("Input shape:", x.shape)
    print("Output shape:", out.shape)
    print("Shape match:", x.shape == out.shape)
    print("Has residual signal:", not torch.allclose(out, x))

if __name__ == "__main__":
    test_block()