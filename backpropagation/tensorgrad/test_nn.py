import pytest
import numpy as np
import torch
import torch.nn as tnn          # was: import torch.nn as nn
from engine import Matrix
from nn import MLP

RNG = np.random.default_rng(42)
def rand(*shape): return RNG.standard_normal(shape)


# ── helpers ──────────────────────────────────────────────────────────────────

def build_torch_mlp(dims, act_fns):
    """Equivalent torch.nn.Sequential for a given dims/act_fns spec."""
    layers = []
    for d_in, d_out, fn in zip(dims[:-1], dims[1:], act_fns):
        layers.append(tnn.Linear(d_in, d_out))
        layers.append(tnn.ReLU() if fn == "relu" else tnn.Tanh())
    return tnn.Sequential(*layers).double()


def sync_weights(mlp, torch_mlp):
    """Copy weights from our MLP into the torch model.

    Our  W : (in, out)   — forward is  x @ W
    Torch W : (out, in)  — forward is  x @ W.T
    So torch.weight = our_W.T  and the grad relationship is symmetric.
    """
    torch_linears = [m for m in torch_mlp.modules() if isinstance(m, tnn.Linear)]
    for (ours, _), theirs in zip(mlp.layers, torch_linears):
        theirs.weight = tnn.Parameter(torch.tensor(ours.W.data.T, dtype=torch.float64))
        theirs.bias   = tnn.Parameter(torch.tensor(ours.b.data,   dtype=torch.float64))


# ── structural tests ──────────────────────────────────────────────────────────

def test_parameter_count():
    mlp = MLP([2, 4, 3], ["relu", "tanh"])
    params = mlp.parameters()
    assert len(params) == 4                    # W and b for each of 2 layers
    assert params[0].shape == (2, 4)           # W layer 0
    assert params[1].shape == (4,)             # b layer 0
    assert params[2].shape == (4, 3)           # W layer 1
    assert params[3].shape == (3,)             # b layer 1

def test_zero_grad():
    mlp = MLP([2, 4, 1], ["relu", "tanh"])
    mlp(Matrix(rand(4, 2))).backward()
    assert any(np.any(p.grad != 0) for p in mlp.parameters()), "no grad after backward"
    mlp.zero_grad()
    for p in mlp.parameters():
        np.testing.assert_array_equal(p.grad, np.zeros_like(p.data))


# ── correctness vs torch ──────────────────────────────────────────────────────

def run(dims, act_fns, batch=6):
    """Forward + backward comparison against torch for the given architecture."""
    mlp       = MLP(dims, act_fns)
    torch_mlp = build_torch_mlp(dims, act_fns)
    sync_weights(mlp, torch_mlp)

    X_np = rand(batch, dims[0])

    # forward
    out_m = mlp(Matrix(X_np))
    out_t = torch_mlp(torch.tensor(X_np, dtype=torch.float64))
    np.testing.assert_allclose(out_m.data, out_t.detach().numpy(), atol=1e-6,
                               err_msg="forward mismatch")

    # backward
    out_m.backward()
    out_t.backward(torch.ones_like(out_t))

    torch_linears = [m for m in torch_mlp.modules() if isinstance(m, tnn.Linear)]
    for i, ((ours, _), theirs) in enumerate(zip(mlp.layers, torch_linears)):
        np.testing.assert_allclose(ours.W.grad, theirs.weight.grad.numpy().T, atol=1e-5,
                                   err_msg=f"W grad mismatch layer {i}")
        np.testing.assert_allclose(ours.b.grad, theirs.bias.grad.numpy(), atol=1e-5,
                                   err_msg=f"b grad mismatch layer {i}")

def test_single_layer_relu():  run([4, 3],       ["relu"])
def test_single_layer_tanh():  run([4, 3],       ["tanh"])
def test_two_layer():          run([3, 8, 4],    ["relu", "tanh"])
def test_three_layer():        run([2, 8, 8, 1], ["relu", "relu", "tanh"])
def test_wide_batch():         run([4, 16, 8],   ["tanh", "relu"], batch=32)


# ── integration: loss decreases on XOR ───────────────────────────────────────

def test_xor_converges():
    random = __import__('random'); random.seed(0); np.random.seed(0)
    X = Matrix(np.array([[0,0],[0,1],[1,0],[1,1]], dtype=np.float64))
    y = np.array([[0],[1],[1],[0]], dtype=np.float64)

    mlp = MLP([2, 8, 1], ["relu", "tanh"])
    lr, losses = 0.1, []

    for _ in range(200):
        mlp.zero_grad()
        pred = mlp(X)
        loss = ((pred - Matrix(y)) ** 2).mean()
        loss.backward()
        losses.append(float(loss.data))
        for p in mlp.parameters():
            p.data -= lr * p.grad

    assert losses[-1] < losses[0],  "loss did not decrease"
    assert losses[-1] < 0.1,        f"loss did not converge (final={losses[-1]:.4f})"