from multiprocessing.pool import Pool as PoolType
import signal
from typing import ByteString
from multiprocessing import Pool
from torchdata.datapipes.iter import (
    IterableWrapper,
    FileLister,
    FileOpener,
    MultiplexerLongest,
)
from torch.utils.data.datapipes.datapipe import IterDataPipe
from torchdata.datapipes.utils import StreamWrapper
from pyzstd import EndlessZstdDecompressor
import re

from utils.pgn import str_to_bitboards
from utils.progressbar import ProgressBar

READ_SIZE = 100_000
SCORE_SPLITTER = re.compile(r"(?:(?:1|0|1\/2)-(?:1|0|1\/2)|\*)\n\n")
MAX_POOLS = None  # Set to None to use all CPU cores, or to a positive integer to limit the number of pools.


def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


class ZstdDecompressor(IterDataPipe):
    class Decompressor:
        def __init__(self, stream: StreamWrapper, pool: PoolType) -> None:
            self.zstd = EndlessZstdDecompressor()
            self.stream = stream
            self.last_part = ""
            self.__pool = pool

        def __iter__(self):
            for chunk in self.__iter_chunk():
                games_strs: list[str] = SCORE_SPLITTER.split(self.last_part + chunk)
                self.last_part = games_strs.pop()
                result = self.__pool.map(str_to_bitboards, games_strs)
                for r in result:
                    if r:
                        yield from r

        def __iter_chunk(self):
            while True:
                compressed_chunk: ByteString = self.stream.read(READ_SIZE)
                if not compressed_chunk:
                    self.stream.seek(0)
                    self.zstd._reset_session()
                    continue
                chunk = self.zstd.decompress(compressed_chunk)
                if not chunk:
                    self.stream.seek(0)
                    self.zstd._reset_session()
                    continue
                yield chunk.decode("utf-8")

    def __init__(self, file_opener: FileOpener) -> None:
        self.file_opener = file_opener
        self.pool = Pool(MAX_POOLS, initializer=init_worker)

    def __iter__(self):
        for filename, file_stream in self.file_opener:
            decompressor = ZstdDecompressor.Decompressor(file_stream, self.pool)
            yield decompressor


def get_datapipeline_pgn(batch_size=512):
    file_lister = FileLister(
        root="data/", masks="lichess_db_standard_rated_*.pgn.zst"
    ).open_files("b")
    print("Using files for training:")
    for f, _ in file_lister:
        print("-", f)
    file_decompressor = ZstdDecompressor(file_lister)  # type: ignore
    dataloader = (
        MultiplexerLongest(*file_decompressor)
        .shuffle(buffer_size=batch_size * 10)
        .batch(batch_size=batch_size)
    )
    return dataloader

def get_validation_pgns(batch_size=512, max_num_batches=1_000):
    # Gets only one file for now.
    file_lister = FileLister(
        root="data/", masks="validation_lichess_db_standard_rated.pgn.zst"
    ).open_files("b")
    _, stream = next(iter(file_lister))
    validation_pool = Pool(MAX_POOLS, initializer=init_worker)
    decompressor = ZstdDecompressor.Decompressor(
            stream, validation_pool)  # type: ignore
    dp = IterableWrapper(decompressor).batch(
            batch_size=batch_size, drop_last=True).header(max_num_batches)
    return dp


if __name__ == "__main__":
    dataloader = get_datapipeline_pgn()
    bar = ProgressBar(desc="Reading FENs", unit="pos")
    for batch in dataloader:
        bar.update(len(batch))
