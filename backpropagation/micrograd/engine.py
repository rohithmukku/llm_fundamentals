class Value:
    def __init__(self, data, op = ''):
        self.data = data
        self.op = op
        self.grad = 0

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    def __add__(self, other):
        other = Value(other) if not isinstance(other, Value) else other

        ret = self.data + other.data
        ret = Value(ret, '+')

        return ret

    def __mul__(self, other):
        other = Value(other) if not isinstance(other, Value) else other

        ret = self.data * other.data
        ret = Value(ret, '*')

        return ret

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * (other**-1)

    def __rtruediv__(self, other):
        return (self**-1) * other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __neg__(self):
        return self * -1

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only int/float powers for now"
        return Value(self.data ** other, '**')