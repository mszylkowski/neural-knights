from os import fstat
from typing import NamedTuple
from typing import BinaryIO
from chess import Board
from multiprocessing import Pool
import signal

import pyzstd
import re
from experiments.calculate_all_moves import MoveEncoder

from utils.ProgressBar import ProgressBar, format_number
from utils.board import board_to_np

Game = NamedTuple("Game", [("moves", list[str]), ("control", str), ("elo", int)])

REMOVE_BRACKETS = re.compile(r"\s(?:\{[^}]+\})?\s?(?:\d+\.+\s)?")
SCORE_SPLITTER = re.compile(r"(?:(?:1|0|1\/2)-(?:1|0|1\/2)|\*)\n\n")
NOTATION_REMOVE = re.compile(r"[?!]")

move_encoder = MoveEncoder()


def str_to_game(game: str) -> Game | None:
    moves = []
    board = Board()
    white_elo = 0
    black_elo = 0
    control = ""
    for line in game.splitlines():
        if line.startswith("1."):
            line = NOTATION_REMOVE.sub("", line)
            moves = REMOVE_BRACKETS.split(line[3:])[:-1]
            for move in moves:
                uci = board.push_san(move).uci()
                rep = board_to_np(board)
                idx = move_encoder.encode(uci)
                # assert rep.shape == (12, 8, 8)
        elif line.startswith("[BlackElo"):
            black_elo = int(line[11:-2])
        elif line.startswith("[WhiteElo"):
            white_elo = int(line[11:-2])
        elif line.startswith("[TimeControl"):
            control = line[14:-2]
    if not len(moves):
        # Abandoned games without any moves.
        return None
    return Game(moves, control, (white_elo + black_elo) // 2)


def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


class ChessReader(BinaryIO):
    def __init__(self) -> None:
        self.games: list[Game] = []
        self.last_part = ""
        self.pool = Pool(6, initializer=init_worker)
        super().__init__()

    def write(self, n: memoryview) -> int:
        games_str = self.last_part + n.tobytes().decode()
        games = SCORE_SPLITTER.split(games_str)
        results = self.pool.map_async(str_to_game, games[:-1])
        self.games.extend(filter(bool, results.get()))
        self.last_part = games[-1]
        return len(n)

    def size(self) -> int:
        return len(self.games)

    def close(self) -> None:
        self.pool.close()
        self.pool.join()


def read_pgns_from_zstd(file_name: str, show_progress=False) -> list[Game]:
    """
    Reads and returns the games from a zstd compressed file.
    """
    input_stream = open(file_name, "rb")
    output_stream = ChessReader()

    if show_progress:
        bar = ProgressBar(
            total=fstat(input_stream.fileno()).st_size,
            desc="Reading PGNs",
            unit="b",
        )

        def cb(_a, val, _b, _c):
            bar.set(val / 8)
            bar.set_postfix_str(f"{format_number(output_stream.size())} games")

        pyzstd.decompress_stream(
            input_stream,
            output_stream,
            callback=cb,
        )

        bar.close()
    else:
        pyzstd.decompress_stream(
            input_stream,
            output_stream,
        )

    return output_stream.games


if __name__ == "__main__":
    # Read the dataset file for February 2018 (with 17M games), and pass it to the reader decompressed.
    # Download it from https://database.lichess.org/#standard_games
    read_pgns_from_zstd(
        "data/lichess_db_standard_rated_2018-02.pgn.zst", show_progress=True
    )
