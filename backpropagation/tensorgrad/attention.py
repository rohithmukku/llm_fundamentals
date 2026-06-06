
import numpy as np
import torch
from .engine import Matrix

class Attention:
    def __init__(self, dim):
        self.W_q = Matrix(np.random.randn(dim, dim) * np.sqrt(1.0 / dim))
        self.W_k = Matrix(np.random.randn(dim, dim) * np.sqrt(1.0 / dim))
        self.W_v = Matrix(np.random.randn(dim, dim) * np.sqrt(1.0 / dim))

    def __call__(self, x):
        Q = x @ self.W_q         # (B, T, D) @ (D, D) => (B, T, D)
        K = x @ self.W_k
        V = x @ self.W_v

        scores = Q @ K.transpose(-2, -1) / np.sqrt(Q.shape[-1])
        weights = scores.softmax(axis=-1)
        output = weights @ V

        return output

def verify():
    # tests/test_attention.py
    np.random.seed(42)
    torch.manual_seed(42)

    dim = 8
    seq_len = 4
    batch = 2

    # Build attention with deterministic weights
    W_q = np.random.randn(dim, dim) * np.sqrt(1.0/dim)
    W_k = np.random.randn(dim, dim) * np.sqrt(1.0/dim)
    W_v = np.random.randn(dim, dim) * np.sqrt(1.0/dim)
    x_data = np.random.randn(batch, seq_len, dim)

    # Your impl
    attn = Attention(dim)
    attn.W_q = Matrix(W_q)
    attn.W_k = Matrix(W_k)
    attn.W_v = Matrix(W_v)
    x = Matrix(x_data)
    out = attn(x)
    loss = out.sum()
    loss.backward()

    # PyTorch reference
    x_t = torch.tensor(x_data, requires_grad=True)
    Wq_t = torch.tensor(W_q, requires_grad=True)
    Wk_t = torch.tensor(W_k, requires_grad=True)
    Wv_t = torch.tensor(W_v, requires_grad=True)

    Q = x_t @ Wq_t
    K = x_t @ Wk_t
    V = x_t @ Wv_t
    scores = Q @ K.transpose(-2, -1) / (dim ** 0.5)
    weights = torch.softmax(scores, dim=-1)
    out_t = weights @ V
    out_t.sum().backward()

    # Compare
    print("x grad match:", np.allclose(x.grad, x_t.grad.numpy(), atol=1e-5))
    print("W_q grad match:", np.allclose(attn.W_q.grad, Wq_t.grad.numpy(), atol=1e-5))
    print("W_k grad match:", np.allclose(attn.W_k.grad, Wk_t.grad.numpy(), atol=1e-5))
    print("W_v grad match:", np.allclose(attn.W_v.grad, Wv_t.grad.numpy(), atol=1e-5))

if __name__ == "__main__":
    verify()
