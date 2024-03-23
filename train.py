from datetime import datetime
from io import StringIO
from time import time
import numpy as np
import torch
from torch import nn
import torch.optim as optim
import argparse
from model import NeuralKnight
from utils.pgnpipeline import get_datapipeline_pgn
from utils.meters import AverageMeter
from utils.model import model_summary, accuracy

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_args():
    parser = argparse.ArgumentParser(
        prog="Train loop", description="Runs the training loop for Neural Knight"
    )
    parser.add_argument(
        "--name",
        "-n",
        type=str,
        default=datetime.now().strftime("%b%d_%H%M"),
        help="Name of the current experiment. Used to save the model and details. Defaults to the date and time.",
    )
    parser.add_argument(
        "--batchsize",
        "-b",
        type=int,
        default=512,
        help="Size of each batch. Defaults to 512.",
    )
    parser.add_argument(
        "--test",
        "-t",
        type=bool,
        default=False,
        help="Do a test run (don't store anything).",
        action=argparse.BooleanOptionalAction,
    )
    return parser.parse_args()


if __name__ == "__main__":
    # Parse arguments
    args = get_args()

    # Create model and helpers
    model = NeuralKnight(device=DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Load data
    dataloader = get_datapipeline_pgn(batch_size=args.batchsize)
    summary_str = model_summary(model, batchsize=args.batchsize)
    args.criterion = criterion
    args.optimizer = optimizer

    # Save markdown summary in `runs/{name}.md`
    name = args.name.replace(" ", "_")
    output = (
        open(f"runs/{name}.md", "w", encoding="utf-8") if not args.test else StringIO()
    )
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
    start = time()

    # Training loop
    for batch_number, batch in enumerate(dataloader, 1):
        batch_x, batch_y = zip(*batch)
        batch_x = torch.tensor(np.array(batch_x), device=DEVICE)
        batch_y = torch.tensor(batch_y, device=DEVICE)
        optimizer.zero_grad()

        outputs = model.forward(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()

        # Update training loss and accuracy
        losses.update(loss.item(), outputs.shape[0])
        acc.update(accuracy(outputs, batch_y), outputs.shape[0])

        curr_time = time() - start
        if batch_number % 100 == 0:
            print(
                f"[Epoch {batch_number // 100:05d}] "
                f"loss: {loss.item():.3f}, acc: {batch_acc:.3f}, time: {curr_time:.1f}"
            )
        if batch_number % 10000 == 0 or batch_number == 1:
            output.write(
                f"| {batch_number // 100:05d} | {losses.avg:.3f} | {acc.avg:.3f} | {int(curr_time)} |\n"
            )
            output.flush()
            if not args.test:
                torch.save(model.state_dict(), model_path)
            losses.reset()
            acc.reset()
