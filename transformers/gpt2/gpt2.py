import torch
import torch.nn as nn
import torch.nn.functional as F
from .block import Block
from .config import ModelConfig

class GPT2(nn.Module):
    def __init__(self, config: ModelConfig, vocab_size: int):
        super().__init__()
        self.config = config
        self.vocab_size = vocab_size

        self.token_embedding = nn.Embedding(vocab_size, config.n_embed)
        if not config.use_rope:
            self.position_embedding = nn.Embedding(config.max_seq_len, config.n_embed)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layers)])
        self.ln = nn.LayerNorm(config.n_embed, bias=config.bias)
        self.lm_head = nn.Linear(config.n_embed, vocab_size, bias=False)

        # self.lm_head.weight = self.token_embedding.weight

    def forward(self, input_ids, targets=None, past_kvs=None, return_kv_cache=False):
        new_kvs = []
        B, T = input_ids.size()

        if T > self.config.max_seq_len:
            raise ValueError(f"Sequence length {T} exceeds max_seq_len {self.config.max_seq_len}")

        pos = torch.arange(T, device=input_ids.device)  # T
        tok_emb = self.token_embedding(input_ids)       # B, T, D

        if not self.config.use_rope:
            pos_emb = self.position_embedding(pos)          # T, D
            x = self.dropout(tok_emb + pos_emb)
        else:
            x = self.dropout(tok_emb)

        for i, block in enumerate(self.blocks):
            past_kv = past_kvs[i] if past_kvs is not None else None
            x, new_kv = block(x, past_kv)
            new_kvs.append(new_kv)
        
        x = self.ln(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                targets.view(-1),
                ignore_index=-1
            )

        if return_kv_cache:
            ret = logits, loss, new_kvs
        else:
            ret = logits, loss

        return ret

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, do_sample=False, top_k=None, use_kv_cache=False):
        kvs = None
        for _ in range(max_new_tokens):
            idx = idx if idx.shape[1] <= self.config.max_seq_len else idx[:,:self.config.max_seq_len]
            if use_kv_cache:
                current_idx = idx if kvs is None else idx[:,[-1]]
                logits, _, kvs = self.forward(current_idx, past_kvs=kvs, return_kv_cache=True)
            else:
                logits, _ = self.forward(idx)
            logits = logits[:, -1, :] / temperature             # B, V

            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float('Inf')     # v[:, [-1]] instead of v[:, -1] preserve dimension as (B, 1)

            probs = F.softmax(logits, dim=-1)
            if do_sample:
                pred = torch.multinomial(probs, num_samples=1)
            else:
                _, pred = torch.topk(logits, 1)

            idx = torch.cat((idx, pred), dim=-1)
            
        return idx



def test_gpt2():
    config = ModelConfig(
        n_embed=128, n_heads=4, n_layers=4,
        max_seq_len=64, dropout=0.0, bias=True,
        use_torch_dot_product=True
    )
    vocab_size = 65  # TinyShakespeare char-level
    model = GPT2(config, vocab_size)
    
    # forward without targets
    idx = torch.randint(0, vocab_size, (2, 16))
    logits, loss = model(idx)
    print("Logits shape:", logits.shape)
    print("Loss (no targets):", loss)
    
    # forward with targets
    targets = torch.randint(0, vocab_size, (2, 16))
    logits, loss = model(idx, targets)
    print("Loss with targets:", loss.item())
    
    # parameter count
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")
    
    # backward pass
    loss.backward()
    print("Backward pass: OK")

if __name__ == "__main__":
    test_gpt2()