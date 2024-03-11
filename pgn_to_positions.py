from dataclasses import dataclass
from os import fstat
from typing import BinaryIO, TextIO
from chess import Board
from multiprocessing import Pool
import signal

import pyzstd
import re

from utils.ProgressBar import ProgressBar, format_number
from utils.board import board_to_np

LOWEST_ELO = 1400
HIGHEST_ELO = 1600


@dataclass
class Game:
    moves: list[tuple[str, str]]
    """Stores the board fen and the move in UCI notation"""

    control: str
    """Time control. Eg: '300+3'"""

    elo: int
    """Average ELO rating of the game"""


REMOVE_BRACKETS = re.compile(r"\s(?:\{[^}]+\})?\s?(?:\d+\.+\s)?")
SCORE_SPLITTER = re.compile(r"(?:(?:1|0|1\/2)-(?:1|0|1\/2)|\*)\n\n")
NOTATION_REMOVE = re.compile(r"[?!]")


def str_to_game(game: str) -> Game | None:
    moves: list[tuple[str, str]] = []
    board = Board()
    elo = 0
    control = ""
    for line in game.splitlines():
        if line.startswith("1."):
            if not HIGHEST_ELO > elo > LOWEST_ELO:
                return None
            line = NOTATION_REMOVE.sub("", line)
            moves_str = REMOVE_BRACKETS.split(line[3:])[:-1]
            if len(moves_str) < 10:
                return None
            for move in moves_str:
                uci = board.push_san(move).uci()
                rep = board.board_fen()
                moves.append((rep, uci))
        elif line.startswith("[BlackElo"):
            elo += int(line[11:-2]) // 2
        elif line.startswith("[WhiteElo"):
            elo += int(line[11:-2]) // 2
        elif line.startswith("[TimeControl"):
            control = line[14:-2]
            base = control.split("+")[0]
            if base != "-" and int(base) < 150:
                return None
    if not len(moves):
        # Abandoned games without any moves.
        return None
    return Game(moves, control, elo)


def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


class ChessReader(BinaryIO):
    def __init__(self, storage: TextIO) -> None:
        super().__init__()

        self.training_size = 0
        self.games = 0
        self.last_part = ""
        self.pool = Pool(None, initializer=init_worker)
        self.storage = storage

    def write(self, n: memoryview) -> int:
        games_str = self.last_part + n.tobytes().decode()
        games = SCORE_SPLITTER.split(games_str)
        self.last_part = games[-1]
        self.pool.map_async(str_to_game, games[:-1], callback=self.results)
        return len(n)

    def results(self, games):
        for game in games:
            if game is not None:
                self.storage.writelines([",".join(x) + "\n" for x in game.moves])
                self.training_size += len(game.moves)
                self.games += 1

    def size(self) -> int:
        return self.games

    def close(self) -> None:
        self.pool.close()
        self.pool.join()
        self.storage.close()


def read_pgns_from_zstd(input_stream: BinaryIO, output_stream: TextIO):
    """
    Reads and returns the games from a zstd compressed file.
    """
    processor = ChessReader(output_stream)

    bar = ProgressBar(
        total=fstat(input_stream.fileno()).st_size,
        desc="Reading PGNs",
        unit="b",
    )

    def cb(total_input, total_output, read_data, write_data):
        bar.set(total_input)
        bar.set_postfix_str(f"{format_number(processor.size())} positions")

    pyzstd.decompress_stream(
        input_stream,
        processor,
        callback=cb,
    )

    bar.close()


if __name__ == "__main__":
    # Read the dataset file for February 2018 (with 17M games), and pass it to the reader decompressed.
    # Download it from https://database.lichess.org/#standard_games
    input_stream = open("data/lichess_db_standard_rated_2018-02.pgn.zst", "rb")
    output_stream = open("data/fen_1500_2018-02.fen.txt", "w")

    read_pgns_from_zstd(input_stream, output_stream)

    input_stream.close()
    output_stream.close()
