# DL Final Project - Neural Knights

![Pytorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Python](https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue)

<a style="display:inline-flex;gap:10px;padding:5px 10px 5px 5px;background:#8883;border-radius:40px;line-height:20px" href="https://github.com/mszylkowski">
<img src="https://avatars.githubusercontent.com/u/22420856" height="20" style="border-radius:50%">
Matias Szylkowski
</a>
<a style="display:inline-flex;gap:10px;padding:5px 10px 5px 5px;background:#8883;border-radius:40px;line-height:20px" href="https://github.com/rabrener">
<img src="https://avatars.githubusercontent.com/u/16889614" height="20" style="border-radius:50%">
Reuven Brener
</a>

- Matias Szylkowski
- Reuven Brener

## Install instructions

Steps:

- Optionally, create a new [Conda](https://www.anaconda.com) environment (eg:`dl_final`).
- Run `pip install -r requirements.txt`.
- Install [Pytorch](https://pytorch.org/get-started/locally) with Cuda (if the device has GPU, otherwise install regular Pytorch).

## Training instructions

Steps:

- Download training datasets from the [Lichess Database](https://database.lichess.org/#standard_games) ([read more](./data/README.md)).
- Run `python -m train --test` (or remove the flag `--test` to store the results in `runs/`).

## Playing

Play against the bot using `python -m play_online -m <model>`, and go to `http://localhost:8080`.

Or, use `python -m play -m <model>` to play in the CLI.

## Additional Resources

Maia Chess (Human Engine):

- https://maiachess.com/
- https://arxiv.org/pdf/2006.01855.pdf
- https://github.com/CSSLab/maia-chess
- https://saumikn.com/blog/a-brief-guide-to-maia/
