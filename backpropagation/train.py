"""
Sample MNIST training with tensorgrad MLP
"""
import random
import argparse
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
 
from tensorgrad.engine import Matrix
from tensorgrad.nn import MLP

def cross_entropy_loss(y_labels, pred):
    """
    y_labels : (B,) int numpy array of class indices
    pred     : Matrix (B, C) logits (any real values)
    returns  : scalar Matrix, mean negative log-likelihood
    """
    B, C = pred.shape
 
    # one-hot target — plain numpy, not tracked
    y_onehot = np.zeros((B, C), dtype=np.float64)
    y_onehot[np.arange(B), y_labels] = 1.0
 
    # subtract row-max for numerical stability (constant, not tracked)
    shift = pred.data.max(axis=1, keepdims=True)     # (B, 1) numpy
    shifted = pred + Matrix(-shift)                   # (B, C)
 
    exp_shifted = shifted.exp()                       # (B, C)
 
    # sum over classes: (B, C) @ (C, 1) = (B, 1)
    sum_exp = exp_shifted @ Matrix(np.ones((C, 1)))   # (B, 1)
    log_sum_exp = sum_exp.log()                        # (B, 1)
 
    log_softmax = shifted - log_sum_exp               # (B, C)  ← broadcasts
 
    # -1/B * sum(y_onehot * log_softmax)
    loss = (Matrix(y_onehot) * log_softmax).sum() * (-1.0 / B)
    return loss

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def parse_args():
    parser = argparse.ArgumentParser(description="Train an MLP on MNIST with tensorgrad")

    parser.add_argument("--steps",      type=int,   default=1000)
    parser.add_argument("--lr",         type=float, default=0.01)
    parser.add_argument("--batch-size", type=int,   default=64)
    parser.add_argument("--seed",       type=int,   default=0)
    parser.add_argument("--data-dir",   type=str,   default="./data")
    parser.add_argument("--dims",    type=int, nargs="+", default=[784, 128, 10])
    parser.add_argument("--act-fns", type=str, nargs="+", default=["relu", "tanh"],
                        choices=["relu", "tanh"])

    args = parser.parse_args()

    if len(args.dims) != len(args.act_fns) + 1:
        parser.error(
            f"--dims must have exactly one more entry than --act-fns "
            f"(got {len(args.dims)} dims, {len(args.act_fns)} act-fns)"
        )

    return args

def create_MLP(args):
    return MLP(args.dims, args.act_fns)

def get_data(args):
    # maybe directly import from torch?
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),   # MNIST channel mean / std
    ])
    train_data = torchvision.datasets.MNIST(
        root=args.data_dir, train=True,  download=True, transform=transform)
    val_data = torchvision.datasets.MNIST(
        root=args.data_dir, train=False, download=True, transform=transform)
    return train_data, val_data

def get_dataloader(data, batch_size):
    train_data, val_data = data
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_data,   batch_size=256,        shuffle=False)
    return train_loader, val_loader

def train(args, model, dataloader):
    lr = args.lr
    losses = []
 
    data_iter = iter(dataloader)
    for step in range(args.steps):
        try:
            X, y = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            X, y = next(data_iter)
 
        X = Matrix(X.view(X.shape[0], -1).numpy())    # (B, 784)
        y = y.numpy()                                  # (B,)
 
        model.zero_grad()
        pred = model(X)                                # (B, 10)
        loss = cross_entropy_loss(y, pred)
        loss.backward()
 
        for p in model.parameters():
            p.data -= lr * p.grad
 
        losses.append(float(loss.data))
        if step % 100 == 0:
            print(f"step {step:>4d}  loss {float(loss.data):.4f}")
 
    return losses

def val(model, dataloader):
    all_labels, all_preds = [], []
 
    for X, y in dataloader:
        X = Matrix(X.view(X.shape[0], -1).numpy())    # (B, 784)
        pred = model(X)                                # (B, 10)
        all_labels.append(y.numpy())
        all_preds.append(np.argmax(pred.data, axis=1))
 
    labels = np.concatenate(all_labels)
    preds  = np.concatenate(all_preds)
    correct = (labels == preds).sum()
    acc = correct / len(labels)
    print(f"val accuracy: {acc:.4f}  ({correct}/{len(labels)})")
    return acc


if __name__ == "__main__":
    args = parse_args()

    set_seed(args.seed)

    model = create_MLP(args)
    data = get_data(args)
    train_loader, val_loader = get_dataloader(data, args.batch_size)

    train(args, model, train_loader)
    val(model, val_loader)