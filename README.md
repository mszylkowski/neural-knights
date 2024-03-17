# DL Final Project - Chess

Authors:

- Matias Szylkowski
- Reuven Brener

## Install instructions

Steps:

- Optionally, create a new [Conda](https://www.anaconda.com) environment (eg:`dl_final`).
- Run `pip install -r requirements.txt`.
- Install [Pytorch](https://pytorch.org/get-started/locally) with Cuda (if the device has GPU, otherwise install regular Pytorch).
- Download training datasets from the [Lichess Database](https://database.lichess.org/#standard_games) ([read more](./README.md)).
- Convert the dataset to a `data/*.fen.txt` using `python -m pgn_to_positions`, which can take 15 minutes or so. Make sure the input files in the source match the files you downloaded. This needs to be done for each dataset downloaded as a `data/*.pgn.zst`.
- Encode the resulting `data/*.fen.txt` into a `data/*.fen.zst` using `python -m utils.zstd -i data/fen_1500_<date>.fen.txt -o data/fen_1500_<date>.fen.zst -m compress`. Optionally, you can delete the `data/*.fen.txt` since we'll only use the `data/*.fen.zst` from now on.
- Run `python -m train --test` (or remove the flag `--test` to store the results in `runs/`).
- Play against the bot using `python -m play -m <model>`

## Additional Resources

Maia Chess (Human Engine):

- https://maiachess.com/
- https://arxiv.org/pdf/2006.01855.pdf
- https://github.com/CSSLab/maia-chess
- https://saumikn.com/blog/a-brief-guide-to-maia/
