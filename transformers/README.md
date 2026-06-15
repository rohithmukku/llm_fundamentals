# Transformers from Scratch

GPT-2 implemented from scratch in PyTorch, trained on TinyShakespeare. No HuggingFace model weights — architecture, training loop, and tokenizer all written by hand.

## Architecture

- Decoder-only transformer (GPT-2 style)
- Causal self-attention with pre-LayerNorm
- Learned positional embeddings (RoPE in progress)
- Character-level tokenization (vocab size: 65)

Default small config (~10M params):

| Param | Value |
|---|---|
| n_embed | 384 |
| n_heads | 6 |
| n_layers | 6 |
| max_seq_len | 256 |

## Training Results

Trained on TinyShakespeare (~1M chars, 80/10/10 split) with cosine LR decay and gradient clipping.

| Steps | Final Loss |
|---|---|
| 5000 | ~1.38 |

Random init baseline: `log(65) ≈ 4.17`. Char-level entropy lower bound: ~1.0.

## Usage

```bash
# Train
python train.py --n_embed 384 --n_heads 6 --n_layers 6 \
  --max_seq_len 256 --steps 5000 --batch_size 32

# Eval
python train.py --mode eval --n_embed 384 --n_heads 6 --n_layers 6 \
  --max_seq_len 256
```

## TODO

- [x] RoPE positional encoding
- [ ] BPE tokenizer

## Structure

```
gpt2/
  config.py      - ModelConfig, TrainerConfig dataclasses
  attention.py   - Causal multi-head self-attention
  block.py       - Transformer block (attention + MLP + LayerNorm + residual)
  gpt2.py        - Full GPT-2 model
train.py         - Training loop with cosine decay, char-level dataset
tokenizer/
  bpe.py         - BPE tokenizer (in progress)
```
