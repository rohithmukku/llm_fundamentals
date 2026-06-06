# Backpropagation Implementation

## Micrograd
Scalar-valued autograd engine. Reimplementation of Karpathy's micrograd 
with support for +, -, *, /, **, pow.

## Tensorgrad  
Tensor-valued autograd extending micrograd to numpy arrays. Supports 
matmul, broadcasting, sum, mean, and common activation functions. 
Gradients verified against PyTorch.

## Training with MNIST

2-layer MLP (784 → 256 → 10), ReLU + Tanh activations
Final validation accuracy: 90.96%

```
python train.py --dims 784 256 10 --act-fns relu tanh
step    0  loss 2.3654
step  100  loss 1.3107
step  200  loss 1.1468
step  300  loss 1.2423
step  400  loss 1.1496
step  500  loss 1.0941
step  600  loss 1.0513
step  700  loss 1.0597
step  800  loss 1.1132
step  900  loss 1.0280
val accuracy: 0.9096  (9096/10000)
```