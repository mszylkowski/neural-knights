import signal
from chess import BaseBoard
import numpy as np
from torchdata.datapipes.iter import (
    FileLister,
    FileOpener,
    MultiplexerLongest,
    Shuffler,
)
from torch.utils.data.datapipes.datapipe import IterDataPipe
from torchdata.datapipes.utils import StreamWrapper
from pyzstd import ZstdDecompressor as ZstdDec
import re

from board import board_to_np
import moves as MoveEncoder
from progressbar import ProgressBar

WINDOW_SIZE = 100_000
SHUFFLER_SIZE = 100_000
NEWLINES_SPLITTER = re.compile("\r\n+")


def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def fen_to_bitboards(line: str) -> tuple[np.ndarray, int]:
    fen, move = line.split(",")
    board = BaseBoard(fen)
    return board_to_np(board), MoveEncoder.encode(move)


class ZstdDecompressor(IterDataPipe):
    def __init__(self, file_opener: FileOpener) -> None:
        self.file_opener = file_opener

    def __iter__(self):
        for filename, file_stream in self.file_opener:
            decompressor = Decompressor(file_stream)
            yield filename, decompressor


class Decompressor:
    def __init__(self, stream: StreamWrapper) -> None:
        self.zstd = ZstdDec()
        self.stream = stream

    def __iter__(self):
        last_part = ""
        while True:
            compressed_chunk = self.stream.read(WINDOW_SIZE)
            if not compressed_chunk:
                break
            chunk = self.zstd.decompress(compressed_chunk)
            if not chunk:
                break
            lines_str = last_part + chunk.decode("utf-8")
            lines: list[str] = NEWLINES_SPLITTER.split(lines_str)
            last_part = lines.pop()
            for line in lines:
                yield fen_to_bitboards(line)


def get_datapipeline_fen():
    file_lister = FileLister(root="data/", masks="*.fen.zst")
    file_opener = FileOpener(file_lister, mode="b")
    file_decompressor = ZstdDecompressor(file_opener)
    multiplexer = MultiplexerLongest(*[x for _, x in file_decompressor])
    shuffler = Shuffler(multiplexer, buffer_size=SHUFFLER_SIZE)
    xys = []
    for b in ProgressBar(shuffler, desc="Decompressing FENs", unit="pos"):
        # xys.append(b)
        pass
    return shuffler


if __name__ == "__main__":
    get_datapipeline_fen()
