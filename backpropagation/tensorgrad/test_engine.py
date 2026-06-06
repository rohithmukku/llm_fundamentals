import pytest
import numpy as np
import torch
from tensorgrad.engine import Matrix


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def run(np_vals, fn_matrix, fn_torch, atol=1e-5):
    """
    np_vals   : dict[str, ndarray]  —  shared float64 inputs
    fn_matrix : dict[str, Matrix]   -> Matrix  (output node)
    fn_torch  : dict[str, Tensor]   -> Tensor

    Runs forward + backward on both engines and asserts that every forward
    output value and every input gradient match within atol.
    """
    # ── Matrix engine ──────────────────────────────────────────────────────
    m = {k: Matrix(v.copy()) for k, v in np_vals.items()}
    m_out = fn_matrix(m)
    m_out.backward()                          # seeds grad with np.ones_like

    # ── PyTorch ────────────────────────────────────────────────────────────
    t = {k: torch.tensor(v.copy(), dtype=torch.float64, requires_grad=True)
         for k, v in np_vals.items()}
    t_out = fn_torch(t)
    t_out.backward(torch.ones_like(t_out))    # same seed as Matrix

    # ── Compare ────────────────────────────────────────────────────────────
    np.testing.assert_allclose(
        m_out.data, t_out.detach().numpy(), atol=atol,
        err_msg="forward pass mismatch")

    for k in np_vals:
        np.testing.assert_allclose(
            m[k].grad, t[k].grad.numpy(), atol=atol,
            err_msg=f"gradient mismatch for '{k}'")


RNG = np.random.default_rng(0)
def rand(*shape): return RNG.standard_normal(shape)


# ─────────────────────────────────────────────────────────────────────────────
# Elementwise / binary ops
# ─────────────────────────────────────────────────────────────────────────────

def test_add():
    run({'A': rand(3,4), 'B': rand(3,4)},
        lambda v: v['A'] + v['B'],
        lambda v: v['A'] + v['B'])

def test_add_broadcast_row():
    run({'A': rand(1,4), 'B': rand(3,4)},
        lambda v: v['A'] + v['B'],
        lambda v: v['A'] + v['B'])

def test_add_broadcast_col():
    run({'A': rand(3,1), 'B': rand(3,4)},
        lambda v: v['A'] + v['B'],
        lambda v: v['A'] + v['B'])

def test_sub():
    run({'A': rand(3,4), 'B': rand(3,4)},
        lambda v: v['A'] - v['B'],
        lambda v: v['A'] - v['B'])

def test_neg():
    run({'A': rand(3,4)},
        lambda v: -v['A'],
        lambda v: -v['A'])

def test_mul():
    run({'A': rand(3,4), 'B': rand(3,4)},
        lambda v: v['A'] * v['B'],
        lambda v: v['A'] * v['B'])

def test_mul_broadcast():
    run({'A': rand(1,4), 'B': rand(3,4)},
        lambda v: v['A'] * v['B'],
        lambda v: v['A'] * v['B'])

def test_div():
    vals = {'A': rand(3,4), 'B': rand(3,4) + 2.0}   # +2 avoids division by zero
    run(vals,
        lambda v: v['A'] / v['B'],
        lambda v: v['A'] / v['B'])

def test_pow():
    run({'A': rand(3,4) + 1.0},                       # +1 keeps values positive
        lambda v: v['A'] ** 3,
        lambda v: v['A'] ** 3)

def test_matmul():
    run({'A': rand(3,4), 'B': rand(4,5)},
        lambda v: v['A'] @ v['B'],
        lambda v: v['A'] @ v['B'])


# ─────────────────────────────────────────────────────────────────────────────
# Reductions
# ─────────────────────────────────────────────────────────────────────────────

def test_sum_all():
    run({'A': rand(3,4)},
        lambda v: v['A'].sum(),
        lambda v: v['A'].sum())

def test_sum_axis0():
    run({'A': rand(3,4)},
        lambda v: v['A'].sum(axis=0),
        lambda v: v['A'].sum(dim=0))

def test_sum_axis1():
    run({'A': rand(3,4)},
        lambda v: v['A'].sum(axis=1),
        lambda v: v['A'].sum(dim=1))

def test_mean_all():
    run({'A': rand(3,4)},
        lambda v: v['A'].mean(),
        lambda v: v['A'].mean())

def test_mean_axis0():
    run({'A': rand(3,4)},
        lambda v: v['A'].mean(axis=0),
        lambda v: v['A'].mean(dim=0))

def test_mean_axis1():
    run({'A': rand(3,4)},
        lambda v: v['A'].mean(axis=1),
        lambda v: v['A'].mean(dim=1))


# ─────────────────────────────────────────────────────────────────────────────
# Activations
# ─────────────────────────────────────────────────────────────────────────────

def test_relu():
    run({'A': rand(3,4)},
        lambda v: v['A'].relu(),
        lambda v: torch.relu(v['A']))

def test_tanh():
    run({'A': rand(3,4)},
        lambda v: v['A'].tanh(),
        lambda v: torch.tanh(v['A']))


# ─────────────────────────────────────────────────────────────────────────────
# Composed expressions
# ─────────────────────────────────────────────────────────────────────────────

def test_linear():
    # out = X @ W + b  (b broadcasts over the batch)
    run({'X': rand(4,8), 'W': rand(8,6), 'b': rand(1,6)},
        lambda v: v['X'] @ v['W'] + v['b'],
        lambda v: v['X'] @ v['W'] + v['b'])

def test_linear_relu():
    run({'X': rand(4,8), 'W': rand(8,6), 'b': rand(1,6)},
        lambda v: (v['X'] @ v['W'] + v['b']).relu(),
        lambda v: torch.relu(v['X'] @ v['W'] + v['b']))

def test_linear_tanh():
    run({'X': rand(4,8), 'W': rand(8,6), 'b': rand(1,6)},
        lambda v: (v['X'] @ v['W'] + v['b']).tanh(),
        lambda v: torch.tanh(v['X'] @ v['W'] + v['b']))

def test_diamond():
    # A feeds two branches that recombine: grad must accumulate correctly
    run({'A': rand(3,4)},
        lambda v: (v['A'] + v['A']) * v['A'],
        lambda v: (v['A'] + v['A']) * v['A'])

def test_shared_intermediate():
    # M = A @ B is a non-leaf node used in two branches
    def build(v):
        M = v['A'] @ v['B']
        return M + M * v['C']
    run({'A': rand(3,4), 'B': rand(4,3), 'C': rand(3,3)}, build, build)

def test_two_layer_mlp():
    # hidden = relu(X @ W1 + b1),  out = mean(hidden @ W2 + b2)
    def build_m(v):
        h = (v['X'] @ v['W1'] + v['b1']).relu()
        return (h @ v['W2'] + v['b2']).mean()

    def build_t(v):
        h = torch.relu(v['X'] @ v['W1'] + v['b1'])
        return (h @ v['W2'] + v['b2']).mean()

    run({'X':  rand(8,4), 'W1': rand(4,6), 'b1': rand(1,6),
                          'W2': rand(6,3), 'b2': rand(1,3)},
        build_m, build_t)