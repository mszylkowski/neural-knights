from dataclasses import dataclass
import re
from chess import BLACK, Board
import numpy as np
from itertools import zip_longest

from utils.board import board_to_np
from utils.moves import encode, mirror_move


@dataclass
class Game:
    moves: list[tuple[str, str]]
    """Stores the board fen and the move in UCI notation"""

    control: str
    """Time control. Eg: '300+3'"""

    elo: int
    """Average ELO rating of the game"""


REMOVE_BRACKETS = re.compile(r"\s(?:\{[^}]+\})?\s?(?:\d+\.+\s)?")
NOTATION_REMOVE = re.compile(r"[?!]")
CLOCK_VALUES = re.compile(r"\d+:\d+:\d+")


def str_to_game(game: str, elo=1500, elo_range=100, low_time=20) -> Game | None:
    moves: list[tuple[str, str]] = []
    current_elo = 0
    control = ""
    for line in game.splitlines():
        if line.startswith("1."):
            if not (elo + elo_range) > current_elo > (elo - elo_range):
                return None
            board = Board()
            line = NOTATION_REMOVE.sub("", line)
            moves_str = REMOVE_BRACKETS.split(line[3:])[:-1]
            clock_times = CLOCK_VALUES.findall(line)
            if len(moves_str) < 10:
                return None
            for move, remaining in zip_longest(moves_str, clock_times):
                if move is None:
                    break
                turn = board.turn
                rep = (board.mirror() if turn == BLACK else board).board_fen()
                uci = board.push_san(move).uci()
                if turn == BLACK:
                    uci = mirror_move(uci)
                use_move = remaining is None or not (
                    remaining.startswith("0:00") and int(remaining[-2:]) < low_time
                )
                if use_move:
                    moves.append((rep, uci))
            del board
        elif line.startswith("[BlackElo") or line.startswith("[WhiteElo"):
            elo_str = line[11:-2]
            if elo_str == "?":
                return None
            current_elo += int(elo_str) // 2
        elif line.startswith("[TimeControl"):
            control = line[14:-2]
            base = control.split("+")[0]
            if base != "-" and int(base) < 150:
                return None
    if not len(moves):
        # Abandoned games without any moves.
        return None
    return Game(moves, control, elo)


def str_to_bitboards(
    game: str, elo=1500, elo_range=100, low_time=20
) -> list[tuple[np.ndarray, int]] | None:
    moves: list[tuple[np.ndarray, int]] = []
    current_elo = 0
    control = ""
    for line in game.splitlines():
        if line.startswith("1."):
            if not (elo + elo_range) > current_elo > (elo - elo_range):
                return None
            board = Board()
            line = NOTATION_REMOVE.sub("", line)
            moves_str = REMOVE_BRACKETS.split(line[3:])[:-1]
            clock_times = CLOCK_VALUES.findall(line)
            if len(moves_str) < 10:
                return None
            for move, remaining in zip_longest(moves_str, clock_times):
                if not move:
                    break
                turn = board.turn
                rep = board_to_np((board.mirror() if turn == BLACK else board))
                uci = board.push_san(move).uci()
                if turn == BLACK:
                    uci = mirror_move(uci)
                use_move = remaining is None or not (
                    remaining.startswith("0:00") and int(remaining[-2:]) < low_time
                )
                if use_move:
                    moves.append((rep, encode(uci)))
            del board
        elif line.startswith("[BlackElo") or line.startswith("[WhiteElo"):
            elo_str = line[11:-2]
            if elo_str == "?":
                return None
            current_elo += int(elo_str) // 2
        elif line.startswith("[TimeControl"):
            control = line[14:-2]
            base = control.split("+")[0]
            if base != "-" and int(base) < 150:
                return None
    if not len(moves):
        # Abandoned games without any moves.
        return None
    return moves


def str_to_bitboards_and_elo(
    game: str,
) -> tuple[list[tuple[np.ndarray, int]], int] | None:
    moves: list[tuple[np.ndarray, int]] = []
    board = Board()
    current_elo = 0
    control = ""
    for line in game.splitlines():
        if line.startswith("1."):
            line = NOTATION_REMOVE.sub("", line)
            moves_str = REMOVE_BRACKETS.split(line[3:])[:-1]
            if len(moves_str) < 10:
                return None
            for move in moves_str:
                turn = board.turn
                rep = board_to_np((board.mirror() if turn == BLACK else board))
                uci = board.push_san(move).uci()
                if turn == BLACK:
                    uci = mirror_move(uci)
                moves.append((rep, encode(uci)))
            del board
        elif line.startswith("[BlackElo") or line.startswith("[WhiteElo"):
            elo_str = line[11:-2]
            if elo_str == "?":
                return None
            current_elo += int(elo_str) // 2
        elif line.startswith("[TimeControl"):
            control = line[14:-2]
            base = control.split("+")[0]
            if base != "-" and int(base) < 150:
                return None
    if not len(moves):
        # Abandoned games without any moves.
        return None
    return moves, current_elo


def str_to_simple_pgn(game: str, elo=1500, elo_range=100) -> str | None:
    content = []
    current_elo = 0
    result = "1/2-1/2"
    for line in game.splitlines():
        if line.startswith("1."):
            if not (elo + elo_range) > current_elo > (elo - elo_range):
                return None
            line = NOTATION_REMOVE.sub("", line)
            moves = REMOVE_BRACKETS.split(line[3:])[:-1]
            if len(moves) < 10:
                return None
            moves_str = [
                f"{str(i//2 + 1) + ". " if i % 2 == 0 else ''}{m}"
                for i, m in enumerate(moves)
            ]
            content.append("\n" + " ".join(moves_str) + " " + result + "\n")
            return "\n".join(content)
        elif line.startswith("[BlackElo"):
            current_elo += int(line[11:-2]) // 2
            content.append(line)
        elif line.startswith("[Result"):
            result = line[9:-2]
        elif line.startswith("[BlackElo") or line.startswith("[WhiteElo"):
            elo_str = line[11:-2]
            if elo_str == "?":
                return None
            current_elo += int(elo_str) // 2
            content.append(line)
        elif line.startswith("[TimeControl"):
            control = line[14:-2]
            base = control.split("+")[0]
            content.append(line)
            if base != "-" and int(base) < 150:
                return None


if __name__ == "__main__":
    input_stream = open("data/sample-pgn.txt", "r").read()
    print(input_stream)
    print(str_to_bitboards(input_stream))
