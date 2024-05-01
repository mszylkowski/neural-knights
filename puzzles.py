import argparse
from chess import Board
from play import DEVICE, get_model_move
from pyzstd import decompress
from collections import defaultdict

from utils.load_model import load_model_from_saved_run

MAX_PUZZLES = 10_000


def get_args():
    parser = argparse.ArgumentParser(description="Runs puzzles for the model provided.")
    parser.add_argument(
        "--run",
        "-r",
        type=str,
        default="./runs/resnet_6b_256f.pt",
        help="Model to evaluate. Has to be .pt",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    model = load_model_from_saved_run(args.run, device=DEVICE)
    scores = defaultdict(list)
    with open("data/lichess_db_puzzle.csv.zst", "rb") as f:
        puzzles = decompress(f.read()).decode("utf-8")
    puzzles = puzzles.splitlines()[1:]
    print(f"running {len(puzzles)} puzzles")
    count = 0
    for puzzle in puzzles:
        (
            puzzle_id,
            fen,
            moves,
            elo,
            elo_std,
            popularity,
            plays,
            themes,
            url,
            openings,
        ) = puzzle.split(",")
        # if abs(1500 - int(elo)) > 100:
        #     continue
        count += 1
        if count > MAX_PUZZLES:
            break
        moves = moves.split(" ")
        themes = themes.split(" ")
        board = Board(fen)
        board.push_san(moves[0])
        prediction = get_model_move(board, board.turn, model)
        correct = moves[1] == prediction.uci()
        for theme in themes:
            scores[theme].append(correct)
    for theme, results in sorted(scores.items()):
        print(
            f"{theme}, {sum([1 if res else 0 for res in results]) / len(results):.2f}, {len(results)}"
        )
