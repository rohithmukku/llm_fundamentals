# llm_fundamentals

Core pieces of modern LLMs, built from scratch — autograd, transformers, and the training/inference machinery — to understand them from first principles rather than calling library functions. The autograd engine, the GPT-2 architecture, the BPE tokenizer, and the KV cache are all written by hand, with results verified against PyTorch and published baselines.

## Highlights

- **GPT-2 small (124M) trained from scratch to 24.8 validation perplexity on WikiText** — beating the published GPT-2-small reference (~29), in ~72 minutes on a single A10 with BF16 + Flash Attention.
- **A hand-written tensor autograd engine** whose attention backward pass is verified element-wise against PyTorch.
- **KV cache implemented and benchmarked** — up to **4.8× decoding speedup** on CPU, with a written analysis of why it *regresses* on Apple MPS.
- Real training engineering: traced a ~13 GB activation leak causing OOM, applied BF16 + Flash Attention, and verified the fast path was actually engaging.

## Contents

### 1. [`backpropagation/`](backpropagation/) — autograd from scratch
- **micrograd** — a scalar-valued autograd engine (reimplementation of Karpathy's micrograd).
- **tensorgrad** — a tensor-valued autograd engine over NumPy (matmul, broadcasting, reductions, activations), with gradients verified against PyTorch.
- **MNIST** — a 2-layer MLP trained with tensorgrad to **90.96% validation accuracy**.
- **Attention from scratch** — scaled dot-product attention built on tensorgrad (batched matmul backward, softmax, transpose with correct gradient flow), with all gradients verified element-wise against PyTorch.

### 2. [`transformers/`](transformers/) — GPT-2 from scratch
GPT-2 implemented from scratch in PyTorch (no Hugging Face weights): decoder-only, causal self-attention with pre-LayerNorm, configurable RoPE, a hand-written BPE tokenizer, and a KV cache.

| Model | Data | Result |
|---|---|---|
| ~10M (char-level) | TinyShakespeare | loss ~1.38 @ 5k steps |
| **124M (GPT-2 small)** | **WikiText** | **24.8 perplexity** (ref ~29) |

Includes a KV-cache benchmark (2.7×–4.8× speedup on CPU, growing with sequence length) and a written analysis of the MPS regression — tracing it to per-op dispatch overhead and `torch.cat` allocations, and noting how production engines (vLLM, TGI, SGLang) use static buffers to avoid it.

Full results, configs, and analysis are in the per-folder READMEs:
[`backpropagation/README.md`](backpropagation/README.md) · [`transformers/README.md`](transformers/README.md)

## Setup

```bash
pip install -r requirements.txt
```