# Runs

This directory will store all the runs from the command `python -m train`.

Each run contains a summary MarkDown document, as well as a Pytorch checkpoint `*.pt`.

## Version Control

Add any hyperparameter fields that you want to keep in `args` (in `train.py`) so they get added to the summary.

To upload good models, use `git add -f <path/to/file>` since files in this folder are gitignored by default. Only do this with the best models, so the repository is not congested. Feel free to delete old models if new ones perform better.

## Naming

In order to restore the models correctly, we use a specific naming scheme that needs to be passed to `python -m train --name <name>`. The name is the type of network (`linear`, `largecnn`, `resnet`) and then the properties indicated by a letter split by underscores. The name must match the configuration passed in `--config`, otherwise the model won't be read properly.

Examples:
- `--name resnet_6b_256f`: Resnet with 6 blocks and 256 filters per block.
- `--name largecnn_10b_128f`: CNN with 10 layers and 128 filters per block.

> Note: Check `utils/load_model.py` to see how letters correspond to options. The number always goes before the letter that represents the YAML config.
