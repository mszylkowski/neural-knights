from multiprocessing.pool import Pool as PoolType
import signal
from typing import ByteString
from multiprocessing import Pool
import torch
from torchdata.datapipes.iter import (
    FileLister,
    FileOpener,
    MultiplexerLongest,
)
import numpy as np
from torch.utils.data.datapipes.datapipe import IterDataPipe
from torchdata.datapipes.utils import StreamWrapper
from pyzstd import EndlessZstdDecompressor
import re

from models.resnet import ResNet
from train import DEVICE
from utils.model import accuracy
from utils.pgn import str_to_bitboards, str_to_bitboards_and_elo
from utils.progressbar import ProgressBar
from utils.moves import NUM_OF_PIECE_TYPES, PAD_BOARD, PAD_MOVE

READ_SIZE = 100_000
SCORE_SPLITTER = re.compile(r"(?:(?:1|0|1\/2)-(?:1|0|1\/2)|\*)\n\n")
MAX_POOLS = None  # Set to None to use all CPU cores, or to a positive integer to limit the number of pools.


def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def get_many_positions(
    game_moves: list[tuple[np.ndarray, int]], cpositions: int
) -> list[tuple[np.array, np.array]]:
    """Return consecutive pairs of board positions and moves.

    It performs a rolling operation, where the first postions-moves pair
    returned in the list is:
      - (pos1, pos2,..., post-cpositions)
      - (mov1, mov2,..., mov-cpositions)
    And the second positions-moves pair is:
      - (pos2, pos3,..., post-(cpositions+1))
      - (mov2, mov3,..., mov-(cpositions+1))
    and so on.  Some pairs will contain special pad values when cpositions
    exceeds the available moves in the game.

    Parameters:
    game_moves: A list of board-move pairs.
    cpositions: The number of positions/moves to return as input for
        transformer sequence inputs.

    Returns:
    A list of pairs of position-move numpy arrays. The position array has
    dimensions cpositions x 12 x 8 x 8, where 12 corresponds to the piece
    types, and 8 x 8 the number of board squares.
    """
    num_moves = len(game_moves)
    results = []

    for i in range(num_moves):
        # Generate sequence of pos-moves.
        result_pos = np.full(
            (cpositions, NUM_OF_PIECE_TYPES, 8, 8), PAD_BOARD, dtype=np.int32
        )
        result_moves = np.full((cpositions,), PAD_MOVE, dtype=np.int32)
        last_j = min(i + cpositions, num_moves)
        for j in range(i, last_j):
            curr_pos, curr_move = game_moves[j]
            result_pos[j - i] = curr_pos
            result_moves[j - i] = curr_move
        results.append((result_pos, result_moves))
    return results


class ZstdDecompressor(IterDataPipe):
    class Decompressor:
        def __init__(self, stream: StreamWrapper, pool: PoolType):
            self.zstd = EndlessZstdDecompressor()
            self.stream = stream
            self.last_part = ""
            self.__pool = pool

        def __iter__(self):
            for chunk in self.__iter_chunk():
                games_strs: list[str] = SCORE_SPLITTER.split(self.last_part + chunk)
                self.last_part = games_strs.pop()
                result = self.__pool.map(str_to_bitboards_and_elo, games_strs)
                for r in result:
                    # First check if r is None and skip all.
                    if not r:
                        pass
                    else:
                        yield r

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


def get_datapipeline_evaluation_pgn():
    file_lister = FileLister(
        root="data/", masks="validation_lichess_db_standard_rated_*.pgn.zst"
    ).open_files("b")
    print("Using files for training:")
    for f, _ in file_lister:
        print("-", f)
    file_decompressor = ZstdDecompressor(file_lister)
    dataloader = MultiplexerLongest(*file_decompressor)
    return dataloader


if __name__ == "__main__":
    dataloader = get_datapipeline_evaluation_pgn()
    model = ResNet(device=DEVICE, blocks=10)
    model.load_state_dict(torch.load("runs/resnet_10b_64f.pt"))
    model.to(DEVICE)
    # List of moves by elo
    elo_buckets = [[] for _ in range(30)]
    # List of moves by positions
    index_buckets = [[] for _ in range(100)]
    print("Loading positions")
    for i, (moves, elo) in enumerate(dataloader):
        if i >= 10000:
            break
        elo_bucket = elo // 100
        for j, (board, move) in enumerate(moves):
            elo_buckets[elo_bucket].append((board, move))
            if j < 100:
                index_buckets[j].append((board, move))
    print("ELO, Accuracy, Moves found")
    for elo, bucket in enumerate(elo_buckets):
        if len(bucket) == 0:
            continue
        elo = elo * 100
        xs, ys = zip(*bucket)
        xs = torch.tensor(np.array(xs), device=DEVICE)
        ys = torch.tensor(ys, device=DEVICE, dtype=torch.long)
        acc = accuracy(model.forward(xs), ys).item()
        print(f"{elo}, {acc*100:.2f}%, {len(bucket)}")
    print("Position index, Accuracy, Moves found")
    for index, bucket in enumerate(index_buckets):
        if len(bucket) == 0:
            continue
        xs, ys = zip(*bucket)
        xs = torch.tensor(np.array(xs), device=DEVICE)
        ys = torch.tensor(ys, device=DEVICE, dtype=torch.long)
        acc = accuracy(model.forward(xs), ys).item()
        print(f"{index}, {acc*100:.2f}%, {len(bucket)}")
