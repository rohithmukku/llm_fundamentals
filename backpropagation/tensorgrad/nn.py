import random
from tensorgrad.engine import Matrix
import numpy as np

class Module:
    def zero_grad(self):
        for p in self.parameters():
            p.grad = np.zeros_like(p.data)

    def parameters(self):
        return []

class LinearLayer(Module):
    def __init__(self, in_shape, out_shape):
        # W = [[random.uniform(-1, 1) for _ in range(out_shape)] for _ in range(in_shape)]
        W = np.random.randn(in_shape, out_shape) * np.sqrt(1.0 / in_shape)  # need He/Kaiming init for stability
        self.W = Matrix(W) # (in_shape, out_shape)
        self.b = Matrix([0 for _ in range(out_shape)]) # (out_shape)

    def __call__(self, x):
        # x: (B, in_shape)
        out = x @ self.W + self.b
        return out

    def parameters(self):
        return [self.W, self.b]
    
    def __repr__(self):
        return f"LinearLayer(W={self.W.shape})"

class NonLinearLayer(Module):
    def __init__(self, fn_name):
        self.fn_name = fn_name

    def __call__(self, x):
        if self.fn_name == "relu":
            out = x.relu()
        elif self.fn_name == "tanh":
            out = x.tanh()
        else:
            raise NotImplementedError
        return out

    def __repr__(self):
        return f"NonLinearLayer(fn={self.fn_name})"

class MLP(Module):
    def __init__(self, dims, act_fns):
        assert len(dims) == len(act_fns) + 1, "len(dims) should be len(act_fns) + 1"
        linear_layers = []
        non_linear_layers = [NonLinearLayer(fn_name) for fn_name in act_fns]
        for i, o in zip(dims[:-1], dims[1:]):
            linear_layers.append(LinearLayer(i, o))
        
        self.layers = list(zip(linear_layers, non_linear_layers))

    def __call__(self, x):
        for linear, non_linear in self.layers:
            x = linear(x)
            x = non_linear(x)
        return x

    def __repr__(self):
        ret = "MLP [\n"
        for i, (linear, non_linear) in enumerate(self.layers):
            delim = ",\n" if i != len(self.layers) - 1 else "\n"
            ret += str(linear) + "," + str(non_linear) + delim
        ret += "]"

        return ret

    def parameters(self):
        return [p for linear, _ in self.layers for p in linear.parameters()]