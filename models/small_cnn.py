from torch import nn
import torch

from utils.moves import NUM_POSSIBLE_MOVES


class SmallCNN(nn.Module):
    def __init__(self, device: torch.device | None = None) -> None:
        super().__init__()
        self.outputs = NUM_POSSIBLE_MOVES

        self.cnn1 = nn.Conv2d(12, 64, kernel_size=3, padding=1)
        self.activation = nn.ELU()
        self.cnn2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.cnn3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.linear = nn.Linear(128 * 8 * 8, self.outputs)

        if device:
            self.to(device)

    def forward(self, x: torch.Tensor):
        x = x.type(torch.float32)
        x = self.cnn1(x)
        x = self.activation(x)
        x = self.cnn2(x)
        x = self.activation(x)
        x = self.cnn3(x)
        x = self.activation(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x
