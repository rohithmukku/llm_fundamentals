import pytest
from engine import Value


# =========================================================================
# Forward pass
# =========================================================================

def test_construction_defaults():
    v = Value(2.0)
    assert v.data == 2.0
    assert v.grad == 0
    assert v._op == ''

def test_repr():
    assert repr(Value(2.0)) == "Value(data=2.0, grad=0)"

# ---------- addition ----------

def test_add_value_value():
    out = Value(2.0) + Value(3.0)
    assert out.data == 5.0
    assert out._op == '+'

def test_add_value_scalar():
    assert (Value(2.0) + 3.0).data == 5.0

def test_radd_scalar_value():
    assert (3.0 + Value(2.0)).data == 5.0

# ---------- multiplication ----------

def test_mul_value_value():
    out = Value(2.0) * Value(3.0)
    assert out.data == 6.0
    assert out._op == '*'

def test_mul_value_scalar():
    assert (Value(2.0) * 3.0).data == 6.0

def test_rmul_scalar_value():
    assert (3.0 * Value(2.0)).data == 6.0

# ---------- negation & subtraction ----------

def test_neg():
    assert (-Value(2.0)).data == -2.0

def test_sub_value_value():
    assert (Value(5.0) - Value(3.0)).data == 2.0

def test_sub_value_scalar():
    assert (Value(5.0) - 3.0).data == 2.0

def test_rsub_scalar_value():
    assert (5.0 - Value(3.0)).data == 2.0

# ---------- power & division ----------

def test_pow():
    assert (Value(2.0) ** 3).data == 8.0

def test_truediv_value_value():
    assert (Value(6.0) / Value(2.0)).data == pytest.approx(3.0)

def test_truediv_value_scalar():
    assert (Value(6.0) / 2.0).data == pytest.approx(3.0)

def test_rtruediv_scalar_value():
    assert (12.0 / Value(3.0)).data == pytest.approx(4.0)

# ---------- a slightly bigger expression ----------

def test_chain_expression():
    a, b, c, d = Value(2.0), Value(3.0), Value(4.0), Value(1.0)
    out = a * b + c - d        # 2*3 + 4 - 1 = 9
    assert out.data == 9.0


# =========================================================================
# Backward pass: compare engine gradients to central-difference numerical
# gradients of the same expression written in plain floats. No torch needed.
# =========================================================================

def _engine_grad(build, point):
    """build: dict[str, Value] -> Value (the output). Returns d(out)/d(input)."""
    vs = {k: Value(x) for k, x in point.items()}
    out = build(vs)
    out.backward()
    return {k: vs[k].grad for k in point}

def _numeric_grad(plain, point, h=1e-6):
    """plain: dict[str, float] -> float. Central differences on each input."""
    grads = {}
    for k in point:
        hi = dict(point); hi[k] += h
        lo = dict(point); lo[k] -= h
        grads[k] = (plain(hi) - plain(lo)) / (2 * h)
    return grads

def _assert_grads(build, plain, point, tol=1e-4):
    eng = _engine_grad(build, point)
    num = _numeric_grad(plain, point)
    for k in point:
        assert eng[k] == pytest.approx(num[k], abs=tol), \
            f"d/d{k}: engine={eng[k]} numeric={num[k]}"


def test_grad_add():
    _assert_grads(lambda v: v['a'] + v['b'],
                  lambda v: v['a'] + v['b'], {'a': 2.0, 'b': -3.0})

def test_grad_mul():
    _assert_grads(lambda v: v['a'] * v['b'],
                  lambda v: v['a'] * v['b'], {'a': 2.0, 'b': -3.0})

def test_grad_pow():
    _assert_grads(lambda v: v['a'] ** 3,
                  lambda v: v['a'] ** 3, {'a': 1.7})

def test_grad_div():
    _assert_grads(lambda v: v['a'] / v['b'],
                  lambda v: v['a'] / v['b'], {'a': 6.0, 'b': 2.5})

def test_grad_chain_a4():
    _assert_grads(lambda v: v['a'] * v['a'] * v['a'] * v['a'],
                  lambda v: v['a'] ** 4, {'a': 2.0})

def test_grad_diamond_shared_leaf():
    _assert_grads(lambda v: (v['a'] + v['a']) * (v['a'] + 1),
                  lambda v: (2 * v['a']) * (v['a'] + 1), {'a': 3.0})

def test_grad_shared_intermediate():
    def build(v):
        m = v['a'] * v['b']
        return m * v['c'] + m * v['d']
    _assert_grads(build,
                  lambda v: (v['a']*v['b'])*v['c'] + (v['a']*v['b'])*v['d'],
                  {'a': 2.0, 'b': 3.0, 'c': 4.0, 'd': 5.0})

def test_grad_mixed_with_reflected_ops():
    _assert_grads(lambda v: 5 - 2 / v['a'] + 3 * v['a'],
                  lambda v: 5 - 2 / v['a'] + 3 * v['a'], {'a': 0.9})

def test_grad_messy():
    _assert_grads(lambda v: (v['a']*v['b'] + v['b']**2) / v['a'] - v['a'],
                  lambda v: (v['a']*v['b'] + v['b']**2) / v['a'] - v['a'],
                  {'a': 1.3, 'b': -2.1})