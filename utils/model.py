import torch
from torchinfo import summary


def model_summary(model: torch.nn.Module,
                  batchsize=1000,
                  consecutive_positions=1):
    if consecutive_positions == 1:
        return repr(summary(model, input_size=(batchsize, 12, 8, 8), verbose=0))
    # For transformers use input_data argument to include both src and tgt.
    src = torch.zeros((batchsize, consecutive_positions, 12, 8, 8))
    tgt = torch.zeros((batchsize, consecutive_positions))
    return repr(summary(model, input_data=[src,tgt], verbose=0))


def accuracy(output: torch.Tensor, target: torch.Tensor):
    """Computes the precision@k for the specified values of k"""
    correct = torch.argmax(output, dim=-1).eq(target).sum() * 1.0
    acc = correct / target.shape[0]
    return acc


def get_parameters_count(self) -> int:
    return sum([param.nelement() for param in self.parameters()])
