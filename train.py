from datetime import datetime
from io import StringIO
from time import time
import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import argparse
from model import NeuralKnight
from utils.pgnpipeline import get_datapipeline_pgn, get_validation_pgns
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
    parser.add_argument(
        "--lr",
        "-l",
        type=float,
        default=0.001,
        help="Learning rate of stochastic gradient descent.",
    )
    parser.add_argument(
        "--momentum",
        "-m",
        type=float,
        default=0.9,
        help="Momentum of stochastic gradient descent.",
    )
    return parser.parse_args()


def update_validation_meters(model, dataloader, val_losses, val_acc, max_num_batches=1000):
    for batch_number in range(max_num_batches):
      batch = next(dataloader)
        batch_x, batch_y = zip(*batch)
        batch_x = torch.tensor(np.array(batch_x), device=DEVICE)
        batch_y = torch.tensor(batch_y, device=DEVICE)

    with torch.no_grad():
        outputs = model.forward(batch_x)
        loss = criterion(outputs, batch_y)

        # Update training loss and accuracy
        val_losses.update(loss.item(), outputs.shape[0])
        batch_acc = accuracy(outputs, batch_y)
        val_acc.update(batch_acc, outputs.shape[0])
    return loss.item(), batch_acc


if __name__ == "__main__":
    # Parse arguments
    args = get_args()

    # Create model and helpers
    model = NeuralKnight(device=DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    # Load data
    dataloader = get_datapipeline_pgn(batch_size=args.batchsize)
    val_dataloader = get_validation_pgns()
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
    val_losses = AverageMeter()
    val_acc = AverageMeter()
    writer = SummaryWriter(f"runs/{name}")
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
        batch_acc = accuracy(outputs, batch_y)
        acc.update(batch_acc, outputs.shape[0])

        # Run validate scores
        update_validation_meters(model, val_dataloader, val_losses, val_acc)

        curr_time = time() - start
        epoch = round(batch_number // 100)
        if batch_number % 100 == 0:
            print(
                f"[Epoch {epoch:05d}] "
                f"train loss: {loss.item():.3f}, acc: {batch_acc:.3f}, time: {curr_time:.1f}"
            )

            # Update writer
            writer.add_scalar("Loss/train", loss.item(), epoch)
            writer.add_scalar("Loss/test", loss.item(), epoch)
            writer.add_scalar("Accuracy/train", batch_acc, epoch)
            writer.add_scalar("Accuracy/test", batch_acc, epoch)
        if batch_number == 1:
            writer.add_graph(model, batch_x)

        if batch_number % 10000 == 0 or batch_number == 1:
            output.write(
                f"| {epoch:05d} | training    | {losses.avg:.3f} | {acc.avg:.3f} | ---------------- |\n"
            )
            output.write(
                f"| ----------- | validation  | {val_losses.avg:.3f} | {val_acc.avg:.3f} | {int(curr_time)} |\n"
            )
            output.flush()
            if not args.test:
                torch.save(model.state_dict(), model_path)
            losses.reset()
            acc.reset()
            val_losses.reset()
            val_acc.reset()
