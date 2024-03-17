# Runs

This directory will store all the runs from the command `python -m train`.

Each run contains a summary MarkDown document, as well as a Pytorch checkpoint `*.pt`.

## Version Control

Add any hyperparameter fields that you want to keep in `args` (in `train.py`) so they get added to the summary.

To upload good models, use `git add -f <path/to/file>` since files in this folder are gitignored by default. Only do this with the best models, so the repository is not congested. Feel free to delete old models if new ones perform better.
