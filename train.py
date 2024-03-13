from math import inf
from time import time
import torch
from torch import nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchinfo import summary as model_summary

from load_fen_zstd import read_fens
from model import NeuralKnight
from utils.meters import AverageMeter

BATCH_SIZE = 1_000
MAX_POSITIONS = inf  # Set to infinity to train with the entire dataset
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    input_stream = open("data/fen_1500_2018-02.fen.zst", "rb")
    xs, ys = read_fens(input_stream, max_positions=MAX_POSITIONS)

    print(f"Training on ({len(xs)}, {len(ys)}) positions")

    model = NeuralKnight()
    model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    dataset = TensorDataset(
        torch.tensor(xs, device=DEVICE),
        torch.tensor(ys, dtype=torch.long, device=DEVICE),
    )
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    model_summary(
        model,
        input_size=(BATCH_SIZE, 12, 8, 8),
        col_names=("output_size", "num_params", "mult_adds"),
        col_width=20,
    )

    def accuracy(output: torch.Tensor, target: torch.Tensor):
        """Computes the precision@k for the specified values of k"""
        correct = torch.argmax(output, dim=-1).eq(target).sum() * 1.0
        acc = correct / target.shape[0]
        return acc

    iter_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()

    minibatch = 0
    for _ in range(10000):
        for _, (batch_x, batch_y) in enumerate(dataloader):
            minibatch += 1
            start = time()
            optimizer.zero_grad()

            outputs = model.forward(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            losses.update(loss.item(), outputs.shape[0])

            batch_acc = accuracy(outputs, batch_y)
            acc.update(batch_acc, outputs.shape[0])
            iter_time.update(time() - start)
            if minibatch % 100 == 0:
                print(
                    f"[Epoch {minibatch // 100:05d}] loss: {losses.avg:.3f}, acc: {acc.avg:.3f}, time: {iter_time.avg:.3f}"
                )
        iter_time.reset()
        losses.reset()
        acc.reset()
