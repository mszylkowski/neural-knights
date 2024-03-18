from multiprocessing.pool import Pool as PoolType
import signal
from typing import ByteString
from chess import BaseBoard
from multiprocessing import Pool
import numpy as np
from torchdata.datapipes.iter import (
    FileLister,
    FileOpener,
    MultiplexerLongest,
)
from torch.utils.data.datapipes.datapipe import IterDataPipe
from torchdata.datapipes.utils import StreamWrapper
from pyzstd import EndlessZstdDecompressor as ZstdDec
import re

from utils.board import board_to_np
import utils.moves as MoveEncoder
from utils.progressbar import ProgressBar

READ_SIZE = 100_000
NEWLINES_SPLITTER = re.compile("\r?\n+")
MAX_POOLS = None  # Set to None to use all CPU cores, or to a positive integer to limit the number of pools.


def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def fen_to_bitboards(line: str) -> tuple[np.ndarray, int]:
    fen, move = line.split(",")
    board = BaseBoard(fen)
    return board_to_np(board), MoveEncoder.encode(move)


class ZstdDecompressor(IterDataPipe):
    class Decompressor:
        def __init__(self, stream: StreamWrapper, pool: PoolType) -> None:
            self.zstd = ZstdDec()
            self.stream = stream
            self.last_part = ""
            self.__pool = pool

        def __iter__(self):
            for chunk in self.__iter_chunk():
                lines: list[str] = NEWLINES_SPLITTER.split(self.last_part + chunk)
                self.last_part = lines.pop()
                result = self.__pool.map(fen_to_bitboards, lines)
                yield from result

        def __iter_chunk(self):
            while True:
                compressed_chunk: ByteString = self.stream.read(READ_SIZE)
                if not compressed_chunk:
                    self.stream.seek(0)
                    continue
                chunk = self.zstd.decompress(compressed_chunk)
                if not chunk:
                    self.stream.seek(0)
                    continue
                yield chunk.decode("utf-8")

    def __init__(self, file_opener: FileOpener) -> None:
        self.file_opener = file_opener
        self.pool = Pool(MAX_POOLS, initializer=init_worker)

    def __iter__(self):
        for filename, file_stream in self.file_opener:
            decompressor = ZstdDecompressor.Decompressor(file_stream, self.pool)
            yield decompressor


def get_datapipeline_fen(batch_size=512):
    file_lister = FileLister(root="data/", masks="*.fen.zst").open_files("b")
    print(list(file_lister))
    file_decompressor = ZstdDecompressor(file_lister)  # type: ignore
    dataloader = (
        MultiplexerLongest(*file_decompressor)
        .shuffle(buffer_size=batch_size * 10)
        .batch(batch_size=batch_size)
    )
    return dataloader


if __name__ == "__main__":
    dataloader = get_datapipeline_fen()
    bar = ProgressBar(desc="Reading FENs", unit="pos")
    for batch in dataloader:
        bar.update(len(batch))
