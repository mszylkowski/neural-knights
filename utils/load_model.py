import sys
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn
import re
from os import path

from models import Linear, ResNet, LargeCNN


def _try_load_model(model: nn.Module, state_dict: dict[str, torch.Tensor]) -> bool:
    try:
        model.load_state_dict(state_dict)
        return True
    except RuntimeError as e:
        print(e)
        return False


NAME_TO_MODEL = {
    "resnet": "ResNet",
    "largecnn": "LargeCNN",
    "linear": "Linear",
    "transformer": "Transformer",
}

LETTER_TO_ARGNAME = {
    "b": "model_blocks",
    "f": "model_num_filters",
    "l": "model_hidden_layers",
    "h": "model_hidden_size",
    "t": None,
}

REMOVE_NUMBERING = re.compile(r"\(\d+\)")


def _infer_args(path_to_run: str):
    assert path_to_run.endswith(".pt"), "Model file ends with .pt"
    file_name = path.basename(path_to_run)[:-3]
    file_name = REMOVE_NUMBERING.sub("", file_name)
    parts = file_name.split("_")
    kwargs: dict[str, Any] = dict(model=NAME_TO_MODEL[parts[0]])
    for part in parts[1:]:
        num, letter = part[:-1], part[-1:]
        arg_name = LETTER_TO_ARGNAME[letter] if letter in LETTER_TO_ARGNAME else None
        if arg_name:
            kwargs[arg_name] = int(num)
        else:
            print(f"Warning: could not process part {part} into an arg")
    return SimpleNamespace(**kwargs)


def load_model_from_saved_run(
    path_to_run: str, args=None, device: torch.device | None = None
) -> nn.Module:
    if args is None:
        args = _infer_args(path_to_run)
    model = get_empty_model(args, device)
    state_dict = torch.load(path_to_run)
    if _try_load_model(model, state_dict):
        return model
    else:
        sys.exit("This run is incompatible with '{model_name}' arch.")


def get_empty_model(args, device: torch.device | None = None) -> nn.Module:
    """Returns a model instantiation based on config args."""
    if args.model == "Linear":
        return Linear(
            device=device,
            hidden_size=args.model_hidden_size or 512,
            hidden_layers=args.model_hidden_layers or 2,
        )
    if args.model == "ResNet":
        return ResNet(
            device=device,
            blocks=args.model_blocks or 6,
            num_filters=args.model_num_filters or 64,
        )
    if args.model == "LargeCNN":
        return LargeCNN(
            device=device,
            blocks=args.model_blocks or 10,
            num_filters=args.model_num_filters or 64,
        )
    raise Exception(f"Model {args.model} not found")
