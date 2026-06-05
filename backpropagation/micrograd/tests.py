import sys
import pytest
from engine import Value

# ---------- construction & repr ----------

def test_construction_defaults():
    v = Value(2.0)
    assert v.data == 2.0
    assert v.grad == 0
    assert v.op == ''

def test_repr():
    assert repr(Value(2.0)) == "Value(data=2.0, grad=0)"

# ---------- addition ----------

def test_add_value_value():
    out = Value(2.0) + Value(3.0)
    assert out.data == 5.0
    assert out.op == '+'

def test_add_value_scalar():
    assert (Value(2.0) + 3.0).data == 5.0

def test_radd_scalar_value():
    assert (3.0 + Value(2.0)).data == 5.0

# ---------- multiplication ----------

def test_mul_value_value():
    out = Value(2.0) * Value(3.0)
    assert out.data == 6.0
    assert out.op == '*'

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
    # 5 - 3 should be 2
    assert (5.0 - Value(3.0)).data == 2.0

# ---------- power & division ----------

def test_pow():
    assert (Value(2.0) ** 3).data == 8.0

def test_truediv_value_value():
    # 6 / 2 should be 3
    assert (Value(6.0) / Value(2.0)).data == pytest.approx(3.0)

def test_truediv_value_scalar():
    assert (Value(6.0) / 2.0).data == pytest.approx(3.0)

# ---------- a slightly bigger expression ----------

def test_chain_expression():
    a, b, c, d = Value(2.0), Value(3.0), Value(4.0), Value(1.0)
    out = a * b + c - d        # 2*3 + 4 - 1 = 9
    assert out.data == 9.0