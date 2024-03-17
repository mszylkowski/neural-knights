from datetime import datetime
from math import inf
from time import time
import torch
from torch import nn
import torch.optim as optim
import argparse

from fenloader import FenDataLoader
from model import NeuralKnight
from utils.meters import AverageMeter
from utils.model import model_summary, accuracy

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def make_arg_parser():
    parser = argparse.ArgumentParser(
        prog="Train loop", description="Runs the training loop for Neural Knight"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=argparse.FileType("rb"),
        default="data/fen_1500_2018-02.fen.zst",
        help="Input file with FEN boards and moves. Should have the format .fen.zst",
    )
    parser.add_argument(
        "--name",
        "-n",
        type=str,
        default=datetime.now().strftime("%b%d_%H%M"),
        help="Name of the current experiment. Used to save the model and details. Defaults to the date.",
    )
    parser.add_argument(
        "--positions",
        "-p",
        type=int,
        default=inf,
        help="Maximum number of positions to train on. Defaults to inf, which trains on the entire dataset.",
    )
    parser.add_argument(
        "--batchsize",
        "-b",
        type=int,
        default=1000,
        help="Size of each batch. Defaults to 1000.",
    )
    return parser


if __name__ == "__main__":
    # Parse arguments
    args = make_arg_parser().parse_args()

    # Create model and helpers
    model = NeuralKnight(device=DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Load data
    dataloader = FenDataLoader(args.input, DEVICE, max_positions=args.positions)
    summary_str = model_summary(model, batchsize=args.batchsize)
    args.criterion = criterion
    args.optimizer = optimizer
    args.positions = len(dataloader) * args.batchsize

    # Save markdown summary in `runs/{name}.md`
    name = args.name.replace(" ", "_")
    output = open(f"runs/{name}.md", "w", encoding="utf-8")
    model_path = f"runs/{name}.pt"
    arg_str = "\n".join([f"- **`--{k}`**: {v}" for k, v in vars(args).items()])

    output.write(
        f"# Training `{name}`\n\nStarted on `{datetime.now()}`\n\n"
        f"**Description:** Add description here\n\nArguments:\n{arg_str}\n\n"
        f"## Model\n\nSaved in `{model_path}`\n\n```\n{summary_str}\n```\n\n"
        "## Training\n\n| Epoch | Loss | Acc | Time |\n| - | - | - | - |\n"
    )
    output.flush()

    # Set up trackers
    losses = AverageMeter()
    acc = AverageMeter()
    minibatch = 0
    start = time()

    # Training loop
    for _ in range(10000):
        for _, (batch_x, batch_y) in enumerate(dataloader):
            minibatch += 1
            optimizer.zero_grad()

            outputs = model.forward(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            losses.update(loss.item(), outputs.shape[0])

            batch_acc = accuracy(outputs, batch_y)
            acc.update(batch_acc, outputs.shape[0])
            curr_time = time() - start
            if minibatch % 100 == 0:
                print(
                    f"[Epoch {minibatch // 100:05d}] loss: {losses.avg:.3f}, acc: {
                        acc.avg:.3f}, time: {curr_time:.3f}"
                )
            if minibatch % 10000 == 0 or minibatch == 1:
                output.write(
                    f"| {minibatch // 100:05d} | {losses.avg:.3f} | {acc.avg:.3f} | {curr_time:.3f} |\n"
                )
                output.flush()
                torch.save(model.state_dict(), model_path)
        losses.reset()
        acc.reset()
