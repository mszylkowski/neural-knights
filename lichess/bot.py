import argparse
from queue import Queue
import chess
import chess.polyglot

from play import DEVICE, get_model_move
from utils.load_model import load_model_from_saved_run
from . import model as li_model
import json
from . import api as lichess
import logging
import multiprocessing
from multiprocessing import Process
import signal
import backoff
from requests.exceptions import HTTPError, ReadTimeout
import yaml

logger = logging.getLogger(__name__)


terminated = False
force_quit = False


def get_token():
    with open("lichess/TOKEN.txt", "rt") as token_file:
        return token_file.readline().strip("\n")


def get_config():
    with open("lichess/config.yaml") as config_file:
        return yaml.safe_load(config_file)


def get_args():
    parser = argparse.ArgumentParser(description="Play on Lichess with a bot")
    parser.add_argument(
        "-l", "--logfile", help="Log file to append logs to.", default=None
    )
    parser.add_argument(
        "-v",
        help="Set logging level to verbose",
        action="store_true",
    )
    parser.add_argument(
        "--run",
        "-r",
        type=str,
        default="./runs/resnet_6b_256f.pt",
        help="Model to evaluate. Has to be .pt",
    )
    return parser.parse_args()


def signal_handler(signal: int, frame):
    global terminated
    global force_quit
    in_starting_thread = __name__ == "__main__"
    if not terminated:
        if in_starting_thread:
            logger.debug("Recieved SIGINT. Terminating client.")
        terminated = True
    else:
        if in_starting_thread:
            logger.debug("Received second SIGINT. Quitting now.")
        force_quit = True


signal.signal(signal.SIGINT, signal_handler)


def is_final(exception: Exception):
    return isinstance(exception, HTTPError) and exception.response.status_code < 500


def upgrade_account(li):
    if li.upgrade_to_bot_account() is None:
        return False

    logger.info("Succesfully upgraded to Bot Account!")
    return True


def watch_control_stream(control_queue: Queue, li: lichess.Lichess) -> None:
    """Put the events in a queue."""
    error = None
    while not terminated:
        try:
            response = li.get_event_stream()
            lines = response.iter_lines()
            for line in lines:
                if line:
                    event = json.loads(line.decode("utf-8"))
                    control_queue.put_nowait(event)
                else:
                    control_queue.put_nowait({"type": "ping"})
        except Exception:
            break

    control_queue.put_nowait({"type": "terminated", "error": error})


def start(li, user_profile, config):
    challenge_config = config["challenge"]
    logger.info(
        "You're now connected to {} and awaiting challenges.".format(config["url"])
    )
    control_queue = multiprocessing.Manager().Queue()
    control_stream = Process(target=watch_control_stream, args=[control_queue, li])
    control_stream.start()
    while not terminated:
        event = control_queue.get()
        if event["type"] == "terminated":
            break
        elif event["type"] == "challenge":
            chlng = li_model.Challenge(event["challenge"])
            if chlng.is_supported(challenge_config):
                try:
                    logger.info("Accept {}".format(chlng))
                    response = li.accept_challenge(chlng.id)
                    logger.info("Challenge Accept Response  :{}".format(response))
                except (HTTPError, ReadTimeout) as exception:
                    if (
                        isinstance(exception, HTTPError)
                        and exception.response.status_code == 404
                    ):  # ignore missing challenge
                        logger.info("    Skip missing  :{}".format(chlng))
            else:
                try:
                    li.decline_challenge(chlng.id)
                    logger.info("    Decline  :{}".format(chlng))
                except:
                    pass
        elif event["type"] == "gameStart":
            game_process = Process(
                target=play_game,
                args=(li, event["game"]["id"], user_profile, config),
            )
            game_process.start()

    logger.info("Terminated")
    control_stream.terminate()
    control_stream.join()


ponder_results = {}


@backoff.on_exception(backoff.expo, BaseException, max_time=600, giveup=is_final)
def play_game(li, game_id, user_profile, config):
    response = li.get_game_stream(game_id)
    run = get_args().run
    model = load_model_from_saved_run(run, device=DEVICE)
    lines = response.iter_lines()
    # Initial response of stream will be the full game info. Store it
    initial_state = json.loads(next(lines).decode("utf-8"))
    game = li_model.Game(
        initial_state,
        user_profile["username"],
        li.baseUrl,
        config.get("abort_time", 20),
    )
    timelim = game.state["btime"] / 1000
    timelim = timelim / 60
    timep = round(timelim / 85 * 60, 1)
    if timep > 10:
        timep = 10
    elif timep < 0.3:
        timep = 0.3
    board = setup_board(game)

    logger.info("Game Details  :{}".format(game))
    li.chat(
        game_id,
        "player",
        "Thanks for helping us test our bot! Have fun and let us know how it went.",
    )

    if is_engine_move(game, board.move_stack) and not is_game_over(game):
        try:
            move = get_model_move(board, board.turn, model)
            board.push(move)
            li.make_move(game.id, move.uci())
        except Exception as e:
            raise e

    while not terminated:
        try:
            binary_chunk = next(lines)
        except StopIteration:
            break
        upd: dict | None = (
            json.loads(binary_chunk.decode("utf-8")) if binary_chunk else None
        )
        u_type = upd["type"] if upd else "ping"
        if not board.is_game_over():
            if u_type == "gameState":
                game.state = upd
                moves = upd["moves"].split()
                board = update_board(board, moves[-1])
                if not is_game_over(game) and is_engine_move(game, moves):
                    move = get_model_move(board, board.turn, model)
                    board.push(move)
                    li.make_move(game.id, move.uci())

                if board.turn == chess.WHITE:
                    game.ping(
                        config.get("abort_time", 20),
                        (upd["wtime"] + upd["winc"]) / 1000 + 60,
                    )
                else:
                    game.ping(
                        config.get("abort_time", 20),
                        (upd["btime"] + upd["binc"]) / 1000 + 60,
                    )

            elif u_type == "ping":
                if game.should_abort_now():
                    logger.info(
                        "    Aborting {} by lack of activity".format(game.url())
                    )
                    li.abort(game.id)
                    break
                elif game.should_terminate_now():
                    logger.info(
                        "    Terminating {} by lack of activity".format(game.url())
                    )
                    if game.is_abortable():
                        li.abort(game.id)
                    break
        else:
            break
    if board.is_game_over():
        li.chat(game_id, "player", "Fun game! Wanna play again?")
    logger.info("game over")


def is_white_to_move(game, moves):
    return len(moves) % 2 == (0 if game.white_starts else 1)


def setup_board(game):
    if game.initial_fen == "startpos":
        board = chess.Board()
    else:
        board = chess.Board(game.initial_fen)
    moves = game.state["moves"].split()
    for move in moves:
        board = update_board(board, move)
    return board


def is_engine_move(game, moves):
    return game.is_white == is_white_to_move(game, moves)


def is_game_over(game):
    return game.state["status"] != "started"


def update_board(board, move):
    uci_move = chess.Move.from_uci(move)
    if board.is_legal(uci_move):
        board.push(uci_move)
    else:
        logger.debug("Ignoring illegal move {} on board {}".format(move, board.fen()))
    return board


def intro():
    return r"""
    .   _/|
    .  // o\
    .  ||  _)  lichess-bot
    .  //__\
    .  )___(   Play on Lichess with a bot
    """


if __name__ == "__main__":
    args = get_args()
    config = get_config()

    logging.basicConfig(
        level=logging.DEBUG if args.v else logging.INFO,
        filename=args.logfile,
        format="%(asctime)-15s: %(message)s",
    )
    logger.info(intro())
    li = lichess.Lichess(get_token(), config["url"], "1.2.0")

    user_profile = li.get_profile()
    username = user_profile["username"]
    is_bot = user_profile.get("title") == "BOT"
    logger.info("Welcome {}!".format(username))

    if not is_bot:
        is_bot = upgrade_account(li)

    if is_bot:
        start(li, user_profile, config)
    else:
        logger.error(
            "{} is not a bot account. Please upgrade it to a bot account!".format(
                user_profile["username"]
            )
        )
