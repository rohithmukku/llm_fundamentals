import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from .config import ModelConfig
from .embedding import RotaryEmbedding

# B: Batch size
# T: Sequence Length
# D: Embedding dimension
# nH: Number of heads
# nD: Dimension per head

class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_embed = config.n_embed
        self.n_heads = config.n_heads
        self.max_seq_len = config.max_seq_len
        self.dropout = config.dropout
        self.bias = config.bias
        self.use_torch_dot_product = config.use_torch_dot_product
        self.use_rope = config.use_rope

        self.attention = nn.Linear(self.n_embed, self.n_embed * 3, bias=self.bias)

        if self.use_rope:
            self.rope = RotaryEmbedding(config.max_seq_len, config.n_embed // config.n_heads)
        else:
            self.rope = nn.Identity()

        self.attn_dropout = nn.Dropout(self.dropout)
        self.resid_dropout = nn.Dropout(self.dropout)

        self.projection = nn.Linear(self.n_embed, self.n_embed)

        if not self.use_torch_dot_product:
            self.register_buffer("causal_mask", torch.tril(torch.ones(self.max_seq_len, self.max_seq_len)).view(1, 1, self.max_seq_len, self.max_seq_len))
        

    def forward(self, x, past_kv=None, return_kv_cache=False):
        # x: (B, T, D)
        B, T, D = x.size()

        if past_kv is None:
            q, k, v = torch.split(self.attention(x), self.n_embed, dim=2) # split along D
            # q, k, v: (B, nH, T, nD)
            q = q.view(B, T, self.n_heads, D // self.n_heads).transpose(1, 2)
            k = k.view(B, T, self.n_heads, D // self.n_heads).transpose(1, 2)
            v = v.view(B, T, self.n_heads, D // self.n_heads).transpose(1, 2)

            # Apply RoPE transformation
            q = self.rope(q)
            k = self.rope(k)
        else:
            k_cached, v_cached = past_kv
            q, k, v = torch.split(self.attention(x[:,[-1],:]), self.n_embed, dim=2)

            q = q.view(B, 1, self.n_heads, D // self.n_heads).transpose(1, 2)
            k = k.view(B, 1, self.n_heads, D // self.n_heads).transpose(1, 2)
            v = v.view(B, 1, self.n_heads, D // self.n_heads).transpose(1, 2)

            # Apply RoPE transformation
            q = self.rope(q)
            k = self.rope(k)

            k = torch.cat((k_cached, k), dim=2)
            v = torch.cat((v_cached, v), dim=2)

        if self.use_torch_dot_product:
            if past_kv is None:
                o = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0, is_causal=True)
            else:
                o = F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0, is_causal=False)
        else:
            scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
            scores = scores.masked_fill(self.causal_mask[:,:,:T,:T] == 0, -torch.inf)
            w = scores.softmax(dim=-1)
            if self.training and self.dropout > 0:
                w = self.attn_dropout(w)
            o = w @ v

        if past_kv is None:
            o = o.transpose(1, 2).contiguous().view(B, T, D)
        else:
            o = o.transpose(1, 2).contiguous().view(B, 1, D)
        o = self.resid_dropout(self.projection(o))

        return o, (k, v) if return_kv_cache else None

def test_attention():
    config = ModelConfig(n_embed=128, n_heads=4, max_seq_len=32, dropout=0.0, bias=True, use_torch_dot_product=False)
    
    # both versions should produce identical output with dropout=0
    attn_manual = CausalSelfAttention(config)
    
    config.use_torch_dot_product = True
    attn_fast = CausalSelfAttention(config)
    attn_fast.load_state_dict(attn_manual.state_dict(), strict=False)
    
    x = torch.randn(2, 16, 128)
    out_manual = attn_manual(x)
    out_fast = attn_fast(x)
    
    print("Match:", torch.allclose(out_manual, out_fast, atol=1e-5))

if __name__ == "__main__":
    test_attention()