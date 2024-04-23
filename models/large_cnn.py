from torch import nn
import torch

from utils.moves import NUM_POSSIBLE_MOVES


class LargeCNN(nn.Module):
    def __init__(
        self, device: torch.device | None = None, num_filters=64, blocks=10
    ) -> None:
        super().__init__()
        self.outputs = NUM_POSSIBLE_MOVES

        self.seq = nn.Sequential(
            *[
                (
                    nn.Conv2d(num_filters, num_filters, kernel_size=3, padding=1)
                    if i % 2 == 0
                    else nn.ELU()
                )
                for i in range((blocks - 1) * 2)
            ]
        )
        self.cnn1 = nn.Conv2d(12, num_filters, kernel_size=3, padding=1)
        self.activation = nn.ELU()
        self.linear = nn.Linear(64 * num_filters, self.outputs)

        if device:
            self.to(device)

    def forward(self, x: torch.Tensor):
        x = x.type(torch.float32)
        x = self.cnn1(x)
        x = self.activation(x)
        x = self.seq(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x
