from math import inf
from multiprocessing import Pool
from os import fstat
import signal
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

        self.xs: list[tuple[np.ndarray, int]] = []
        self.ys: list[int] = []
        self.last_part = ""
        self.pool = Pool(None, initializer=init_worker)

    def write(self, n: memoryview) -> int:
        lines_str = self.last_part + n.tobytes().decode()
        lines = NEWLINES_SPLITTER.split(lines_str)
        self.last_part = lines[-1]
        self.pool.map_async(fen_to_bitboards, lines[:-1], callback=self.xs.extend)
        return len(n)

    def __len__(self) -> int:
        return len(self.xs)

    def close(self) -> None:
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

    bar.close()

    print(f"Found {len(processor)} training positions")
