from math import inf
from multiprocessing import Pool
from os import fstat
import signal
import time
from typing import BinaryIO
import numpy as np
import pyzstd
from chess import BaseBoard
import re
from torch.utils.data import DataLoader, TensorDataset
import torch

from utils import moves as MoveEncoder
from utils.ProgressBar import ProgressBar, format_number
from utils.board import board_to_np

NEWLINES_SPLITTER = re.compile("\r\n+")


def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def fen_to_bitboards(line: str) -> tuple[np.ndarray, int]:
    fen, move = line.split(",")
    board = BaseBoard(fen)
    return board_to_np(board), MoveEncoder.encode(move)


class FenReader(BinaryIO):
    def __init__(self, max_positions=inf) -> None:
        super().__init__()

        self.xys: list[tuple[np.ndarray, int]] = []
        self.last_part = ""
        self.pool = Pool(None, initializer=init_worker)
        self.total = 0
        self.max_positions = max_positions

    def write(self, n: memoryview) -> int:
        if self.total >= self.max_positions:
            return len(n)
        lines_str = self.last_part + n.tobytes().decode()
        lines = NEWLINES_SPLITTER.split(lines_str)
        self.last_part = lines[-1]
        lines = lines[:-1]
        self.total += len(lines)
        self.last_res = self.pool.map_async(
            fen_to_bitboards, lines, callback=self.xys.extend
        )
        return len(n)

    def __len__(self) -> int:
        return len(self.xys)

    def close(self) -> None:
        self.last_res.get()
        self.pool.close()
        self.pool.join()


class FenDataLoader(DataLoader):
    def __init__(
        self,
        zst_input_stream: BinaryIO,
        device: torch.device | None,
        max_positions=inf,
        batchsize=1000,
    ) -> None:
        xs, ys = read_fens(zst_input_stream, max_positions=max_positions)

        self.dataset = TensorDataset(
            torch.tensor(xs, device=device),
            torch.tensor(ys, dtype=torch.long, device=device),
        )
        DataLoader.__init__(self, self.dataset, batch_size=batchsize, shuffle=True)


def read_fens(zst_input_stream: BinaryIO, max_positions=inf):
    processor = FenReader(max_positions=max_positions)

    bar = ProgressBar(
        total=fstat(zst_input_stream.fileno()).st_size,
        desc="Reading PGNs",
        unit="B",
        colour="yellow",
    )

    def cb(total_input, total_output, read_data, write_data):
        bar.set(total_input)
        bar.set_postfix_str(f"{format_number(len(processor))} positions")

    pyzstd.decompress_stream(zst_input_stream, processor, callback=cb)
    bar.close()
    bar = ProgressBar(
        total=processor.total, desc="Processing FENs", unit="positions", colour="green"
    )

    while len(processor) < processor.total:
        time.sleep(0.01)
        bar.set(len(processor))

    processor.close()
    bar.close()

    xs, ys = zip(*processor.xys)
    xs_arr = np.array(xs)
    ys_arr = np.array(ys)

    return xs_arr, ys_arr


if __name__ == "__main__":
    input_stream = open("data/fen_1500_2018-02.fen.zst", "rb")
    xs, ys = read_fens(input_stream, max_positions=1_000_000)
    print(
        f"xs = {xs.shape} ({format_number(xs.nbytes)}B), ys = {ys.shape} ({format_number(ys.nbytes)}b)"
    )
