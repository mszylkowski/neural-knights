import torch
from torchinfo import summary


def model_summary(model: torch.nn.Module, batchsize=1000):
    return repr(summary(model, input_size=(1, 12, 8, 8), verbose=0))


def accuracy(output: torch.Tensor, target: torch.Tensor):
    """Computes the precision@k for the specified values of k"""
    correct = torch.argmax(output, dim=-1).eq(target).sum() * 1.0
    acc = correct / target.shape[0]
    return acc
