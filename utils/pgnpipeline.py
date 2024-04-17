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
import numpy as np
from torch.utils.data.datapipes.datapipe import IterDataPipe
from torchdata.datapipes.utils import StreamWrapper
from pyzstd import EndlessZstdDecompressor
import re

from utils.pgn import str_to_bitboards
from utils.progressbar import ProgressBar
from utils.moves import NUM_OF_PIECE_TYPES, PAD_BOARD, PAD_MOVE

READ_SIZE = 100_000
SCORE_SPLITTER = re.compile(r"(?:(?:1|0|1\/2)-(?:1|0|1\/2)|\*)\n\n")
MAX_POOLS = None  # Set to None to use all CPU cores, or to a positive integer to limit the number of pools.


def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def get_many_positions(game_moves: list[tuple[np.ndarray, int]],
                       cpositions: int) -> tuple[np.array, np.array]:
    """Return consecutive pairs of board positions and moves.

    Consecutive positions start at a random move number in the game to sample
    from openings, middle games and endings. If the cpositions exceeds the
    available moves in the game, pad the arrays with special values.

    Parameters:
    game_moves: A list of board-move pairs.
    cpositions: The number of positions/moves to return as input for
        transformer sequence inputs.

    Returns:
    A pair of position-move numpy arrays. The position array has dimensions
    cpositions x 12 x 8 x 8, where 12 corresponds to the piece types, and
    8 x 8 the number of board squares.
    """
    num_moves = len(game_moves)
    start_move_index = np.random.randint(0, num_moves)
    num_positions = min(cpositions, num_moves - start_move_index)
    result_pos = np.full((cpositions, NUM_OF_PIECE_TYPES, 8, 8), PAD_BOARD,
                         dtype=np.int32)
    result_moves = np.full((cpositions,), PAD_MOVE, dtype=np.int32)
    for i in range(num_positions):
        curr_pos, curr_move = game_moves[start_move_index + i]
        result_pos[i] = curr_pos
        result_moves[i] = curr_move
    return (result_pos, result_moves)


class ZstdDecompressor(IterDataPipe):
    class Decompressor:
        def __init__(self, stream: StreamWrapper,
                     pool: PoolType,
                     consecutive_positions: int) -> None:
            self.zstd = EndlessZstdDecompressor()
            self.stream = stream
            self.last_part = ""
            self.__pool = pool
            self._cpositions = consecutive_positions

        def __iter__(self):
            for chunk in self.__iter_chunk():
                games_strs: list[str] = SCORE_SPLITTER.split(self.last_part + chunk)
                self.last_part = games_strs.pop()
                result = self.__pool.map(str_to_bitboards, games_strs)
                for r in result:
                    # First check if r is None and skip all.
                    if not r:
                        pass
                    elif self._cpositions > 1:
                        # Adjust r for Transformer input with multiple
                        # consecutive positions.
                        yield get_many_positions(r, self._cpositions)
                    else:
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

    def __init__(self, file_opener: FileOpener, consecutive_positions: int) -> None:
        self.file_opener = file_opener
        self.pool = Pool(MAX_POOLS, initializer=init_worker)
        self._cpositions = consecutive_positions

    def __iter__(self):
        for filename, file_stream in self.file_opener:
            decompressor = ZstdDecompressor.Decompressor(file_stream,
                                                         self.pool,
                                                         self._cpositions)
            yield decompressor


def get_datapipeline_pgn(batch_size=512, consecutive_positions=1):
    file_lister = FileLister(
        root="data/", masks="lichess_db_standard_rated_*.pgn.zst"
    ).open_files("b")
    print("Using files for training:")
    for f, _ in file_lister:
        print("-", f)
    file_decompressor = ZstdDecompressor(file_lister, consecutive_positions)
    dataloader = (
        MultiplexerLongest(*file_decompressor)
        .shuffle(buffer_size=batch_size * 10)
        .batch(batch_size=batch_size)
    )
    return dataloader


def get_validation_pgns(batch_size=512, consecutive_positions=1):
    file_lister = FileLister(
        root="data/", masks="validation_lichess_db_standard_rated_*.pgn.zst"
    ).open_files("b")
    assert len(list(file_lister)) > 0, "No validation files found"

    file_decompressor = ZstdDecompressor(file_lister, consecutive_positions)
    dataloader = (
        MultiplexerLongest(*file_decompressor)
        .shuffle(buffer_size=batch_size * 10)
        .batch(batch_size=batch_size)
    )
    return dataloader


if __name__ == "__main__":
    dataloader = get_datapipeline_pgn()
    bar = ProgressBar(desc="Reading FENs", unit="pos")
    for batch in dataloader:
        bar.update(len(batch))
