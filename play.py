import argparse
import yaml

import torch
from torch import nn
from chess import BLACK, WHITE, Board, Move
from termcolor import colored

import utils.moves as MoveEncoder
from utils.args import save_config_to_args
from utils.board import board_to_np
from utils.model import model_summary
from utils.load_model import load_model_from_saved_run


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_args():
    parser = argparse.ArgumentParser(
        prog="Play", description="Plays a game of Chess against Neural Knight"
    )
    parser.add_argument(
        "--color",
        "-c",
        type=str,
        default="w",
        choices=["w", "b"],
        help="Color to play for the human. Defaults to white.",
    )
    parser.add_argument(
        "--run",
        "-r",
        type=str,
        default="runs/Mar16_1949.pt",
        help="File path of the model. Should be `runs/*.pt`.",
    )
    return parser.parse_args()


def pprint_board(board: Board):
    unicode = board.unicode(empty_square=" ")
    for i, line in enumerate(unicode.split("\n")):
        for j, char in enumerate(line.split(" ")):
            print(
                colored(
                    char + " ",
                    "black",
                    (
                        "on_white"
                        if (i + (j if len(char) else j // 2)) % 2
                        else "on_light_grey"
                    ),
                ),
                end="",
            )
        print()


def get_player_move(board: Board, player_color: bool):
    pprint_board(board)
    while board.turn == WHITE:
        player_move = input("Enter your move: ")
        if len(player_move) > 2:
            player_move = player_move.upper() if player_color == WHITE else player_move
            try:
                move = board.push_san(player_move)
                board.pop()
                return move
            except ValueError as e:
                print(e)
                continue
        try:
            move = board.push_san(player_move)
            board.pop()
            return move
        except ValueError as e:
            print(e)


def get_model_move(
    board: Board, model_color: bool, model: nn.Module, greedy=True
) -> Move:
    board_correct_view = board.mirror() if model_color == BLACK else board
    model_input = torch.tensor(
        board_to_np(board_correct_view).reshape(1, 12, 8, 8), device=DEVICE
    )
    model_move: torch.Tensor = model.forward(model_input)
    generator = (
        model_move.argsort(descending=True).cpu().numpy()[0]
        if greedy
        else torch.pow(model_move, 10).multinomial(model_move.shape.numel())[0]
    )
    for move_idx in generator:
        corrected_move = MoveEncoder.decode(int(move_idx.item()))
        if corrected_move.startswith("<"):
            continue
        move_uci = (
            MoveEncoder.mirror_move(corrected_move)
            if model_color == BLACK
            else corrected_move
        )
        if move_uci.startswith("O"):
            try:
                board.push_san(move_uci)
                return board.pop()
            except ValueError:
                continue
        else:
            corrected_move = Move.from_uci(corrected_move)
            move = Move.from_uci(move_uci)
            # Try both normal move, mirrored move and promotion.
            moves_to_try = [Move.from_uci(move_uci + "q"), move, corrected_move]
            for m in moves_to_try:
                if board.is_legal(m):
                    return move
            else:
                continue
    raise Exception("No move found")


if __name__ == "__main__":
    args = get_args()
    config = yaml.safe_load(args.config)
    save_config_to_args(config, args)

    path_to_run = args.run
    model = load_model_from_saved_run(path_to_run, device=DEVICE)
    model.eval()
    model_name = model.__class__.__name__
    print(f"Loaded model: {model_name}")
    print(model_summary(model, batchsize=1))

    board = Board()
    player_color = WHITE if args.color == "w" else BLACK
    while True:
        move = get_player_move(board, player_color)
        board.push(move)
        move = get_model_move(board, not player_color, model)
        board.push(move)
        if board.is_game_over():
            print(board.result())
            break
