from play import DEVICE, get_model_move
from utils.load_model import load_model_from_saved_run
from chess import BLACK, WHITE, Board

from utils.progressbar import ProgressBar

FENS = [
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 1",
    "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/2N5/PPPP1PPP/R1BQKBNR w KQkq - 2 3",
    "rnbqkb1r/pppp1ppp/5n2/4p3/2B1P3/8/PPPP1PPP/RNBQK1NR w KQkq - 2 3",
    "rnbqkb1r/ppp1pppp/5n2/3p4/3P1B2/8/PPP1PPPP/RN1QKBNR w KQkq - 2 3",
    "rnbqkbnr/ppp2ppp/4p3/3p4/2PP4/8/PP2PPPP/RNBQKBNR w KQkq - 0 3",
    "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/8/PPPP1PPP/RNBQK1NR w KQkq - 2 3",
]

MODELS_TO_TEST = [
    "runs/resnet_6b_256f_20t.pt",
    "runs/resnet_6b_256f.pt",
    "runs/largecnn_10b_64f.pt",
    "runs/linear_512h_6l.pt",
    "runs/linear_512h_4l.pt",
    "runs/linear_512h_2l(1).pt",
]

if __name__ == "__main__":
    models = {m: load_model_from_saved_run(m, device=DEVICE) for m in MODELS_TO_TEST}
    scores = [[0.0] * len(models) for _ in models]
    for i in range(len(scores)):
        scores[i][i] = "-"
    print(models.keys())
    bar = ProgressBar(
        unit="games",
        total=(len(FENS) * len(MODELS_TO_TEST) * (len(MODELS_TO_TEST) - 1)),
    )
    for i, model_white in enumerate(models.items()):
        for j, model_black in list(enumerate(models.items())):
            if i == j:
                continue
            for game, fen in enumerate(FENS):
                bar.update(1)
                board = Board(fen)
                while board.outcome() is None:
                    try:
                        board.push(get_model_move(board, WHITE, model_white[1]))
                        board.push(get_model_move(board, BLACK, model_black[1]))
                    except Exception:
                        break
                outcome = board.outcome()
                if not outcome:
                    pass
                elif outcome.winner == WHITE:
                    scores[i][j] += 1
                elif outcome.winner == BLACK:
                    scores[j][i] += 1
                else:
                    scores[i][j] += 0.5
                    scores[j][i] += 0.5
    bar.close()

    for model_name, results in zip(MODELS_TO_TEST, scores):
        print(
            model_name.ljust(30),
            results,
            "=",
            sum([x for x in results if isinstance(x, float)]),
        )
