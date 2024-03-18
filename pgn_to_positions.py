import argparse
from io import BufferedIOBase
from os import fstat
from time import sleep
from typing import BinaryIO
from multiprocessing import Pool
import signal

import pyzstd
import re

from utils.pgn import str_to_game
from utils.progressbar import ProgressBar, format_number


SCORE_SPLITTER = re.compile(r"(?:(?:1|0|1\/2)-(?:1|0|1\/2)|\*)\n\n")
DATE_FROM_ZST = re.compile(r"\d{4}-\d{2}")


def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def strs_to_games(game_strs: list[str]):
    return [str_to_game(game) for game in game_strs]


class ChessReader(BinaryIO):
    def __init__(self, storage: BufferedIOBase) -> None:
        super().__init__()

        self.training_size = 0
        self.to_process_games = 0
        self.processed_games = 0
        self.games = 0
        self.last_part = ""
        self.pool = Pool(None, initializer=init_worker)
        self.storage = storage

    def write(self, n: memoryview) -> int:
        games_str = self.last_part + n.tobytes().decode(encoding="ISO-8859-1")
        games = SCORE_SPLITTER.split(games_str)
        self.last_part = games.pop()
        self.to_process_games += len(games)
        self.last_res = self.pool.apply_async(
            strs_to_games, [games], callback=self.results
        )
        return len(n)

    def results(self, games):
        for game in games:
            if game is not None:
                self.storage.writelines(
                    [(",".join(x) + "\n").encode("utf-8") for x in game.moves]
                )
                self.training_size += len(game.moves)
                self.games += 1
        self.processed_games += len(games)

    def size(self) -> int:
        return self.games

    def close(self) -> None:
        self.last_res.get()
        self.pool.close()
        self.pool.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="PGN to FEN", description="Converts PGN files to FEN format."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=argparse.FileType("rb"),
        help="Path to the PGN file. Should be `data/*.pgn.zst`",
        required=True,
    )
    parser.add_argument("--elo", "-e", type=int, default=1500)
    args = parser.parse_args()

    date = DATE_FROM_ZST.findall(args.input.name)[0]

    # Read the dataset file for February 2018 (with 17M games), and pass it to the reader decompressed.
    # Download it from https://database.lichess.org/#standard_games
    output_stream = pyzstd.ZstdFile(f"data/fen_{args.elo}_{date}.fen.zst", "w")

    processor = ChessReader(output_stream)

    bar = ProgressBar(
        total=fstat(args.input.fileno()).st_size,
        desc="Reading PGNs",
        unit="b",
    )

    def cb(total_input, _to, _rd, _wd):
        bar.set(total_input)
        bar.set_postfix_str(
            f"{format_number(processor.size())}/{format_number(processor.processed_games)} games, {format_number(processor.training_size)} pos"
        )

    pyzstd.decompress_stream(
        args.input,
        processor,
        callback=cb,
    )

    bar = ProgressBar(total=processor.to_process_games, unit="games", colour="yellow")
    while processor.to_process_games > processor.processed_games:
        sleep(0.01)
        bar.set(processor.processed_games)
        bar.set_postfix_str(
            f"{format_number(processor.size())}/{format_number(processor.processed_games)} games, {format_number(processor.training_size)} pos"
        )

    print("Cleaning")

    processor.close()
    bar.close()

    output_stream.close()
