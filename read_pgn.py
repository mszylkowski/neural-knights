from os import fstat
from typing import NamedTuple
from typing import BinaryIO

import pyzstd
import re

from utils.ProgressBar import ProgressBar, format_number

Game = NamedTuple("Game", [("moves", list[str]), ("control", str), ("elo", int)])

REMOVE_BRACKETS = re.compile(r"\s(?:\{[^}]+\})?\s?(?:\d+\.+\s)?")
SCORE_SPLITTER = re.compile(r"(?:(?:1|0|1\/2)-(?:1|0|1\/2)|\*)\n\n")


def str_to_game(game: str) -> Game | None:
    moves = []
    white_elo = 0
    black_elo = 0
    control = ""
    for line in game.splitlines():
        if line.startswith("1."):
            moves = REMOVE_BRACKETS.split(line[3:])[:-1]
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


class ChessReader(BinaryIO):
    def __init__(self) -> None:
        self.games: list[Game] = []
        self.last_part = ""
        super().__init__()

    def write(self, n: memoryview) -> int:
        games_str = self.last_part + n.tobytes().decode()
        games = SCORE_SPLITTER.split(games_str)
        self.games.extend(filter(bool, map(str_to_game, games[:-1])))
        self.last_part = games[-1]
        return len(n)

    def size(self) -> int:
        return len(self.games)


def read_pgns_from_zstd(file_name: str, show_progress=False) -> list[Game]:
    """
    Reads and returns the games from a zstd compressed file.
    """
    input_stream = open(file_name, "rb")
    output_stream = ChessReader()

    if show_progress:
        bar = ProgressBar(
            total=fstat(input_stream.fileno()).st_size * 8,
            desc="Reading PGNs",
            unit="b",
        )

        def cb(_a, val, _b, _c):
            bar.set(val)
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
