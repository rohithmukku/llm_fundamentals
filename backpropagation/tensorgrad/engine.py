import numpy as np

class Matrix:
    def __init__(self, data, op='', children=()):
        self.data = np.asarray(data, dtype=np.float64)
        self.op = op
        self.grad = np.zeros_like(self.data)
        self._children = set(children)
        self._backward = lambda: None

    @property
    def shape(self):
        return self.data.shape

    @property
    def size(self):
        return self.data.size

    @property
    def T(self):
        return np.transpose(self.data)

    def __repr__(self):
        return f"Tensor(shape={self.data.shape}, op='{self.op}')"

    @staticmethod
    def _unbroadcast(grad, target_shape):
        """
        A (3,) -> broadcasts to (1, 3) -> expands to (2, 3)
        B (2,3)
        C = A + B -> (2, 3)
        dL/dC -> (2,3)
        grad.shape -> (2,3)
        target_shape -> (3,) for A
        """
        # handle missing dimensions
        extra_dims = grad.ndim - len(target_shape)
        for _ in range(extra_dims):
            grad = grad.sum(axis=0)

        # handle dimensions with 1
        for i, s in enumerate(target_shape):
            if s == 1:
                grad = grad.sum(axis=i, keepdims=True)

        return grad


    def __add__(self, other):
        if not isinstance(other, Matrix):
            other = Matrix(other)

        ret = self.data + other.data
        ret = Matrix(ret, '+', (self, other))

        def _backward():
            self.grad += self._unbroadcast(ret.grad, self.shape)
            other.grad += self._unbroadcast(ret.grad, other.shape)
        ret._backward = _backward

        return ret

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __matmul__(self, other):
        """
        Matrix Multiplication

        backward
        C = A @ B → dA = dC @ B.T, dB = A.T @ dC
        """
        if not isinstance(other, Matrix):
            other = Matrix(other)

        assert self.data.shape[1] == other.data.shape[0], "Matrices columns, rows should match"

        ret = self.data @ other.data
        ret = Matrix(ret, '@', (self, other))

        def _backward():
            self.grad += ret.grad @ other.T
            other.grad += self.T @ ret.grad
        ret._backward = _backward

        return ret

    def __rmatmul__(self, other):
        return self @ other

    def __mul__(self, other):
        """
        Element wise multiplication

        C = A * B → dA = dC * B, dB = A * dC
        """
        if not isinstance(other, Matrix):
            other = Matrix(other)

        ret = self.data * other.data
        ret = Matrix(ret, '*', (self, other))

        def _backward():
            self.grad += self._unbroadcast(ret.grad * other.data, self.shape)
            other.grad += self._unbroadcast(self.data * ret.grad, other.shape)
        ret._backward = _backward

        return ret

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        return self * (other ** -1)

    def __rtruediv__(self, other):
        return other * (self ** -1)

    def __pow__(self, other):
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"

        ret = self.data ** other
        ret = Matrix(ret, '**', (self,))
    
        def _backward():
            self.grad += other * (self.data ** (other - 1)) * ret.grad
        ret._backward = _backward

        return ret

    def __neg__(self):
        return self * -1

    def sum(self, axis=None):
        ret = Matrix(self.data.sum(axis=axis), 'sum', (self,))
        def _backward():
            # grad flows back to every element that was summed,
            # so broadcast it back out to self.shape
            if axis is None:
                self.grad += np.broadcast_to(ret.grad, self.shape)
            else:
                # re-insert the axis that sum() collapsed so broadcast works
                self.grad += np.broadcast_to(
                    np.expand_dims(ret.grad, axis=axis), self.shape
                )
        ret._backward = _backward
        return ret

    def mean(self, axis=None):
        ret = Matrix(self.data.mean(axis=axis), 'mean', (self,))
        n = self.size if axis is None else self.shape[axis]
        def _backward():
            # grad flows back to every element that was summed,
            # so broadcast it back out to self.shape
            if axis is None:
                self.grad += np.broadcast_to(ret.grad, self.shape)/n
            else:
                # re-insert the axis that mean() collapsed so broadcast works
                self.grad += np.broadcast_to(
                    np.expand_dims(ret.grad, axis=axis), self.shape
                )/n
        ret._backward = _backward
        return ret

    def relu(self):
        ret = Matrix(np.maximum(0, self.data), 'relu', (self,))
        def _backward():
            self.grad += (ret.data > 0) * ret.grad
        ret._backward = _backward
        return ret
 
    def tanh(self):
        t = np.tanh(self.data)
        ret = Matrix(t, 'tanh', (self,))
        def _backward():
            self.grad += (1 - t**2) * ret.grad
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

        self.grad = np.ones_like(self.data)
        for node in reversed(array):
            node._backward()