class Value:
    def __init__(self, data, op = '', children=()):
        self.data = data
        self._op = op
        self.grad = 0
        self._children = set(children)
        self._backward = lambda: None

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"

    def __add__(self, other):
        other = Value(other) if not isinstance(other, Value) else other

        ret = self.data + other.data
        ret = Value(ret, '+', (self, other))

        def _backward():
            self.grad += ret.grad
            other.grad += ret.grad
    
        ret._backward = _backward

        return ret

    def __mul__(self, other):
        other = Value(other) if not isinstance(other, Value) else other

        ret = self.data * other.data
        ret = Value(ret, '*', (self, other))

        def _backward():
            self.grad += other.data * ret.grad
            other.grad += self.data * ret.grad
        
        ret._backward = _backward

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
        
        ret = self.data ** other
        ret = Value(ret, '**', (self,))

        def _backward():
            self.grad += other * (self.data ** (other - 1)) * ret.grad
        
        ret._backward = _backward

        return ret

    def backward(self):
        visited = set()
        array = []
        def dfs(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    dfs(child)
                array.append(v)
        
        dfs(self)

        self.grad = 1
        for node in reversed(array):
            node._backward()