import signal
from typing import ByteString
from chess import BaseBoard
from multiprocessing import Pool
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
SHUFFLER_SIZE = 10_000
NEWLINES_SPLITTER = re.compile("\r\n+")
MAX_POOLS = None  # Set to None to use all CPU cores, or to a positive integer to limit the number of pools.


def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def fen_to_bitboards(line: str) -> tuple[np.ndarray, int]:
    fen, move = line.split(",")
    board = BaseBoard(fen)
    return board_to_np(board), MoveEncoder.encode(move)


class ZstdDecompressor(IterDataPipe):
    class Decompressor:
        def __init__(self, stream: StreamWrapper) -> None:
            self.zstd = ZstdDec()
            self.stream = stream
            self.last_part = ""
            self.__pool = Pool(MAX_POOLS, initializer=init_worker)

        def __iter__(self):
            for chunk in self.__iter_chunk():
                lines: list[str] = NEWLINES_SPLITTER.split(self.last_part + chunk)
                self.last_part = lines.pop()
                yield from self.__pool.map(fen_to_bitboards, lines)

        def __iter_chunk(self):
            while True:
                compressed_chunk: ByteString = self.stream.read(WINDOW_SIZE)
                if not compressed_chunk:
                    return
                chunk = self.zstd.decompress(compressed_chunk)
                if not chunk:
                    return
                yield chunk.decode("utf-8")

    def __init__(self, file_opener: FileOpener) -> None:
        self.file_opener = file_opener

    def __iter__(self):
        for filename, file_stream in self.file_opener:
            decompressor = ZstdDecompressor.Decompressor(file_stream)
            yield filename, decompressor


def get_datapipeline_fen():
    file_lister = FileLister(root="data/", masks="*.fen.zst")
    file_opener = FileOpener(file_lister, mode="b")
    file_decompressor = ZstdDecompressor(file_opener)
    multiplexer = MultiplexerLongest(*[x for _, x in file_decompressor])
    shuffler = Shuffler(multiplexer, buffer_size=SHUFFLER_SIZE)
    for b in ProgressBar(shuffler, desc="Decompressing FENs", unit="pos"):
        pass
    return shuffler


if __name__ == "__main__":
    get_datapipeline_fen()
