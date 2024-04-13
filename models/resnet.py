import torch
from torch import nn
from torch.nn import functional as F

from utils.moves import get_all_moves, NUM_OF_SQUARES


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)

class SimpleSkipLayer(nn.Module):
    """Same-padding layer with a skip connection.

    By limiting functionality to same-padding input/outputs and an equal
    number of channels, we can implement a simpler version of the ResNet basic
    block, without having to worry about fitting shortuct dimensions.
    """
    def __init__(self, channels, apply_batch_norm=False):
        super().__init__()
        self.outputs = len(get_all_moves())
        self._apply_bnorm = apply_batch_norm
        
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        x = x.type(torch.float32)
        # First convolution block
        out = self.conv1(x)
        if self._apply_bnorm:
            out = self.bn1(out)
        out = F.relu(out)
        # Second convolution block
        out = self.conv2(out)
        if self._apply_bnorm:
            out = self.bn2(out)
        # Skip connection
        out += x
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, device: torch.device | None = None,
                 apply_batch_norm=False) -> None:
        super().__init__()
        self.outputs = len(get_all_moves())

        num_filters = 64

        self.first_conv = nn.Conv2d(12, num_filters, kernel_size=3, padding=1)
        self.res = nn.Sequential(
            SimpleSkipLayer(num_filters, apply_batch_norm=apply_batch_norm),
            SimpleSkipLayer(num_filters, apply_batch_norm=apply_batch_norm),
            SimpleSkipLayer(num_filters, apply_batch_norm=apply_batch_norm),
            SimpleSkipLayer(num_filters, apply_batch_norm=apply_batch_norm),
            SimpleSkipLayer(num_filters, apply_batch_norm=apply_batch_norm),
        )
        self.linear = nn.Linear(num_filters * NUM_OF_SQUARES, self.outputs)
        self.apply(init_weights)

    def forward(self, x):
        x = x.type(torch.float32)
        out = self.first_conv(x)
        out = self.res(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
