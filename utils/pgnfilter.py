# This file can be used to filter PGN files into smaller files with simpler PGNs
# and only in the correct ELO range. It doesn't make a lot of difference in
# training speed to use simpler ELOs.

import signal
from typing import ByteString
import pyzstd
from torchdata.datapipes.iter import (
    FileLister,
    FileOpener,
)
from torch.utils.data.datapipes.datapipe import IterDataPipe
from torchdata.datapipes.utils import StreamWrapper
from pyzstd import EndlessZstdDecompressor as ZstdDec
import re

from pgn_to_positions import DATE_FROM_ZST
from utils.pgn import str_to_simple_pgn
from utils.progressbar import ProgressBar

READ_SIZE = 100_000
SCORE_SPLITTER = re.compile(r"(?:(?:1|0|1\/2)-(?:1|0|1\/2)|\*)\n\n")
MAX_POOLS = None  # Set to None to use all CPU cores, or to a positive integer to limit the number of pools.


def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


class ZstdDecompressor(IterDataPipe):
    class Decompressor:
        def __init__(self, stream: StreamWrapper) -> None:
            self.zstd = ZstdDec()
            self.stream = stream
            self.last_part = ""

        def __iter__(self):
            for chunk in self.__iter_chunk():
                games_strs: list[str] = SCORE_SPLITTER.split(self.last_part + chunk)
                self.last_part = games_strs.pop()
                result = list(map(str_to_simple_pgn, games_strs))
                for r in result:
                    if r:
                        yield r

        def __iter_chunk(self):
            while True:
                compressed_chunk: ByteString = self.stream.read(READ_SIZE)
                if not compressed_chunk:
                    continue
                chunk = self.zstd.decompress(compressed_chunk)
                if not chunk:
                    continue
                yield chunk.decode("utf-8")

    def __init__(self, file_opener: FileOpener) -> None:
        self.file_opener = file_opener

    def __iter__(self):
        for filename, file_stream in self.file_opener:
            decompressor = ZstdDecompressor.Decompressor(file_stream)
            yield filename, decompressor


def get_datapipeline_pgn_filtered():
    file_lister = FileLister(
        root="data/", masks="lichess_db_standard_rated_*.pgn.zst"
    ).open_files("b")
    for f in file_lister:
        print(f[0])
    print(list(file_lister))
    file_decompressor = ZstdDecompressor(file_lister)  # type: ignore
    return file_decompressor


if __name__ == "__main__":
    dataloader = get_datapipeline_pgn_filtered()
    bar = ProgressBar(desc="Reading PGNs", unit="games")
    for filename, decompressor in dataloader:
        date = DATE_FROM_ZST.findall(filename)[0]
        bar.set_postfix_str(date)
        zstdout = pyzstd.ZstdFile(f"data/lichess_db_1500_{date}.pgn.zst", "w")
        for pgn in decompressor:
            zstdout.write((pgn + "\n").encode("utf-8"))
            bar.update(1)
