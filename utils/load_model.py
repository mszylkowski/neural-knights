import sys

import torch
from torch import nn

from models import Linear, ResNet, SmallCNN, Transformer

def _try_load_model(model: nn.Module, state_dict: dict[str, torch.Tensor]) -> bool:
    model_name = model.__class__.__name__
    try:
        model.load_state_dict(state_dict)
        return True
    except RuntimeError as e:
        print(e)
        return False


def load_model_from_saved_run(path_to_run, args, DEVICE) -> nn.Module:
    model = get_empty_model(args, DEVICE)
    state_dict = torch.load(path_to_run, map_location=DEVICE)
    if _try_load_model(model, state_dict):
        return model
    else:
        sys.exit("This run is incompatible with '{model_name}' arch.")


def get_empty_model(args, DEVICE) -> nn.Module:
    """Returns a model instantiation based on config args."""
    if args.model == "Linear":
        return Linear(device=DEVICE)
    if args.model == "SmallCNN":
        return SmallCNN(device=DEVICE)
    if args.model == "ResNet":
        return ResNet(device=DEVICE, blocks=args.model_blocks or 6)
    if args.model == "Transformer":
        return Transformer(
            device=DEVICE,
            num_heads=args.num_heads,
            dim_feedforward=args.dim_feedforward,
            num_layers_enc=args.num_layers_enc,
            num_layers_dec=args.num_layers_dec,
            dropout=args.dropout,
            sequence_length=args.consecutive_positions,
        )
    raise Exception("Model {args.model} not found")
