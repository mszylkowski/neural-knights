from torch import nn
import torch

from utils.moves import NUM_POSSIBLE_MOVES, NUM_OF_SQUARES, NUM_OF_PIECE_TYPES


class Linear(nn.Module):
    def __init__(self, device: torch.device | None = None, hid_scaler: int = 5) -> None:
        super().__init__()
        self.inputs = NUM_OF_SQUARES * NUM_OF_PIECE_TYPES
        self.outputs = NUM_POSSIBLE_MOVES
        hidden_size = self.inputs * hid_scaler

        self.fc1 = nn.Linear(self.inputs, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, self.outputs)
        self.activation = nn.ELU()

        if device:
            self.to(device)

    def forward(self, x: torch.Tensor):
        x = x.view(x.size(0), -1)
        x = x.type(torch.float32)
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.fc3(x)
        return x

