from torch import nn
import torch

from utils.moves import NUM_POSSIBLE_MOVES, NUM_OF_SQUARES, NUM_OF_PIECE_TYPES


class Linear(nn.Module):
    def __init__(
        self,
        device: torch.device | None = None,
        hidden_size: int = 512,
        hidden_layers=2,
    ) -> None:
        super().__init__()
        self.inputs = NUM_OF_SQUARES * NUM_OF_PIECE_TYPES
        self.outputs = NUM_POSSIBLE_MOVES

        self.fc_first = nn.Linear(self.inputs, hidden_size)
        self.seq = nn.Sequential(
            *[
                (
                    nn.Linear(hidden_size, hidden_size, device=device)
                    if i % 2 == 0
                    else nn.ELU()
                )
                for i in range((hidden_layers - 1) * 2)
            ]
        )
        self.fc_last = nn.Linear(hidden_size, self.outputs)
        self.activation = nn.ELU()

        if device:
            self.to(device)

    def forward(self, x: torch.Tensor):
        x = x.view(x.size(0), -1)
        x = x.type(torch.float32)
        x = self.fc_first(x)
        x = self.activation(x)
        x = self.seq(x)
        x = self.fc_last(x)
        return x
