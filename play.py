import argparse
from chess import BLACK, WHITE, Board, Move
import torch

from model import NeuralKnight
from utils.board import board_to_np
from utils.model import model_summary
import utils.moves as MoveEncoder

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
        "--model",
        "-m",
        type=str,
        default="runs/Mar16_1949.pt",
        help="File path of the model. Should be `runs/*.pt`.",
    )
    return parser.parse_args()


def get_player_move(board: Board, player_color: bool):
    print(board.unicode(empty_square=" "))
    while board.turn == WHITE:
        player_move = input("Enter your move: ")
        if len(player_move) > 2:
            player_move = player_move.upper() if player_color == WHITE else player_move
            try:
                board.push_san(player_move)
            except ValueError as e:
                print(e)
                continue
            return
        try:
            board.push_san(player_move)
        except ValueError as e:
            print(e)


def get_model_move(board: Board, model_color: bool, model: NeuralKnight) -> Move:
    board_correct_view = board.mirror() if model_color == BLACK else board
    model_input = torch.tensor(
        board_to_np(board_correct_view).reshape(1, 12, 8, 8), device=DEVICE
    )
    model_move = model.forward(model_input)
    for move_idx in model_move.argsort(descending=True).cpu().numpy()[0]:
        corrected_move = MoveEncoder.decode(int(move_idx.item()))
        move = (
            MoveEncoder.mirror_move(corrected_move)
            if model_color == BLACK
            else corrected_move
        )
        corrected_move = Move.from_uci(corrected_move)
        if move.startswith("O"):
            try:
                return board.push_san(move)
            except ValueError:
                continue
        else:
            move = Move.from_uci(move)
            # Try both normal move and mirrored move, in case the model was not trained properly.
            if board.is_legal(move):
                print("Used model move:", move.uci())
                board.push(move)
            elif board.is_legal(corrected_move):
                print("Used mirrored model move:", move.uci())
                board.push(corrected_move)
            else:
                print("Move was illegal", move.uci(), corrected_move.uci())
                continue
        return move
    raise Exception("No move found")


if __name__ == "__main__":
    args = get_args()
    model = NeuralKnight(device=DEVICE)
    model.load_state_dict(torch.load(args.model, map_location=DEVICE))
    print("Loaded model Neural Knight:")
    print(model_summary(model, batchsize=1))

    board = Board()
    player_color = WHITE if args.color == "w" else BLACK
    while True:
        get_player_move(board, player_color)
        get_model_move(board, not player_color, model)

        if board.is_game_over():
            print(board.result())
            break
