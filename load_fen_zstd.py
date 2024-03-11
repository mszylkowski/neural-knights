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

    def write(self, n: memoryview) -> int:
        lines_str = self.last_part + n.tobytes().decode()
        lines = NEWLINES_SPLITTER.split(lines_str)
        self.last_part = lines[-1]
        self.total += len(lines) - 1
        self.last_res = self.pool.map_async(
            fen_to_bitboards, lines[:-1], callback=self.xys.extend
        )
        return len(n)

    def __len__(self) -> int:
        return len(self.xys)

    def close(self) -> None:
        self.last_res.get()
        self.pool.close()
        self.pool.join()


if __name__ == "__main__":
    input_stream = open("data/fen_1500_2018-02.fen.zst", "rb")
    processor = FenReader()

    bar = ProgressBar(
        total=fstat(input_stream.fileno()).st_size,
        desc="Reading PGNs",
        unit="lines",
    )

    def cb(total_input, total_output, read_data, write_data):
        bar.set(total_input)
        bar.set_postfix_str(f"{format_number(len(processor))} positions")

    pyzstd.decompress_stream(input_stream, processor, callback=cb)
    bar.desc = "Processing FENs"
    bar.total = processor.total
    while len(processor) < processor.total:
        time.sleep(0.1)
        bar.set(len(processor))
        bar.set_postfix_str(f"{format_number(len(processor))} positions")

    processor.close()
    bar.close()

    print(f"Found {len(processor)} training positions")

    xs, ys = zip(*processor.xys)
    print(len(xs), len(ys))
    xs_arr = np.array(xs)
    print(xs_arr.shape, format_number(xs_arr.nbytes) + "B")
