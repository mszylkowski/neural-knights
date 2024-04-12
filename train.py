import argparse
import yaml

from datetime import datetime
from io import StringIO
from time import time

import numpy as np
import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR

from models import SmallCNN
from utils.pgnpipeline import get_datapipeline_pgn, get_validation_pgns
from utils.prettyprint import config_to_markdown_list
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
        "--test",
        "-t",
        type=bool,
        default=False,
        help="Do a test run (don't store anything).",
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--config",
        "-c",
        type=argparse.FileType(),
        default="./configs/small_cnn.yaml",
        help="Train config spec. Used to define model and hyperparameter values.",
    )
    return parser.parse_args()


def parse_config(config_path):
    """Load config.yaml and parse."""
    return 

def save_config_to_args(config, args):
    """Save config back into args for convenience."""
    for key in config:
        for k, v in config[key].items():
            setattr(args, k, v)


def get_validation_scores(model, criterion, dataloader, max_num_batches=10):
    all_losses = np.zeros(max_num_batches, dtype=np.float32)
    all_accs = np.zeros(max_num_batches, dtype=np.float32)
    for batch_number in range(max_num_batches):
        batch = next(iter(dataloader))
        batch_x, batch_y = zip(*batch)
        batch_x = torch.tensor(np.array(batch_x), device=DEVICE)
        batch_y = torch.tensor(batch_y, device=DEVICE)

        with torch.no_grad():
            outputs = model.forward(batch_x)
            loss = criterion(outputs, batch_y)
            batch_acc = accuracy(outputs, batch_y)
        all_losses[batch_number] = loss.item()
        all_accs[batch_number] = batch_acc
    return np.mean(all_losses), np.mean(all_accs)


def get_model(args):
    """Returns a model instantiation based on config args."""
    if args.model == "SmallCNN":
        return SmallCNN(device=DEVICE)


if __name__ == "__main__":
    # Parse arguments
    args = get_args()

    config = yaml.safe_load(args.config)
    save_config_to_args(config, args)

    # Create model and helpers
    model = get_model(args)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.reg_l2
    )
    scheduler = ExponentialLR(optimizer, gamma=args.exponential_decay)

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
        "## Training\n\n| Epoch | Train/Val | Loss | Acc | Time |\n| - | - | - | - | - |\n"
    )
    output.flush()

    # Set up trackers
    losses = AverageMeter()
    acc = AverageMeter()
    writer = SummaryWriter(f"runs/{name}")
    start = time()

    # Write summary of trainiing config to Tensorboard
    writer.add_text("Training Configuration", config_to_markdown_list(config))

    # Training loop
    for batch_number, batch in enumerate(dataloader, 1):
        epoch = round(batch_number // 100)

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
        scheduler.step(epoch)

        curr_time = time() - start
        if batch_number % 100 == 0:
            print(
                f"[Epoch {epoch:05d}] "
                f"train loss: {loss.item():.3f}, acc: {batch_acc:.3f}, time: {curr_time:.1f}"
            )
        if batch_number == 1:
            writer.add_graph(model, batch_x)

        # Training - Update Tensorboard writer
        writer.add_scalar("Loss/train", loss.item(), epoch)
        writer.add_scalar("Accuracy/train", batch_acc, epoch)

        # Every 10 epochs
        if batch_number % 1000 == 0 or batch_number == 1:
            # Run validate scores
            val_loss, val_acc = get_validation_scores(model, criterion, val_dataloader)
            curr_time = time() - start
            print(
                "           "
                f"-> valid loss: {val_loss:.3f}, acc: {val_acc:.3f}, time: {curr_time:.1f}"
            )

            # Validation - Update Tensorboard writer
            writer.add_scalar("Loss/test", val_loss, epoch)
            writer.add_scalar("Accuracy/test", val_acc, epoch)
            writer.add_scalar("Optimizer/LR", scheduler.get_last_lr()[0], epoch)

            output.write(
                f"| {epoch:05d} | training | {losses.avg:.3f} | {acc.avg:.3f} | ---------- |\n"
            )
            output.write(
                f"| -------- | validation | {val_loss:.3f} | {val_acc:.3f} | {int(curr_time)} |\n"
            )
            output.flush()
            if not args.test:
                torch.save(model.state_dict(), model_path)
            losses.reset()
            acc.reset()
