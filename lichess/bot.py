import argparse
import chess
from chess import engine
from chess import variant
import chess.polyglot
import model
import json
import lichess
import logging
import multiprocessing
from multiprocessing import Process
import signal
import backoff
from config import load_config
from requests.exceptions import HTTPError, ReadTimeout
import os
import time

logger = logging.getLogger(__name__)

from http.client import RemoteDisconnected


terminated = False


def signal_handler(signal, frame):
    global terminated
    logger.debug("Recieved SIGINT. Terminating client.")
    terminated = True


signal.signal(signal.SIGINT, signal_handler)


def is_final(exception):
    return isinstance(exception, HTTPError) and exception.response.status_code < 500


def upgrade_account(li):
    if li.upgrade_to_bot_account() is None:
        return False

    logger.info("Succesfully upgraded to Bot Account!")
    return True


def watch_control_stream(control_queue, li):
    while not terminated:
        try:
            response = li.get_event_stream()
            lines = response.iter_lines()
            for line in lines:
                if line:
                    event = json.loads(line.decode("utf-8"))
                    control_queue.put_nowait(event)
                    logger.info(event)
        except:
            logger.info(
                "Network error:Cannot get data from lichess! Check your network connection or try again in a few minutes."
            )
            pass


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
            chlng = model.Challenge(event["challenge"])
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
            play_game(li, event["game"]["id"], user_profile, config)

    logger.info("Terminated")
    control_stream.terminate()
    control_stream.join()


ponder_results = {}


@backoff.on_exception(backoff.expo, BaseException, max_time=600, giveup=is_final)
def play_game(li, game_id, user_profile, config):
    response = li.get_game_stream(game_id)
    lines = response.iter_lines()
    # Initial response of stream will be the full game info. Store it
    initial_state = json.loads(next(lines).decode("utf-8"))
    game = model.Game(
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
    cfg = config["engine"]

    if type(board).uci_variant == "chess":
        engine_path = os.path.join(cfg["dir"], cfg["name"])
    else:
        engine_path = os.path.join(cfg["dir"], cfg["variantname"])
    engineeng = engine.SimpleEngine.popen_uci(engine_path)
    engineeng.configure({"Threads": 5})
    engineeng.configure({"Hash": 120})
    try:
        engineeng.configure({"EvalFile": "nn-4f56ecfca5b7.nnue"})
    except:
        pass
    engineeng.configure({"Use NNUE": True})

    logger.info("Game Details  :{}".format(game))

    delay_seconds = config.get("rate_limiting_delay", 0) / 1000

    if is_engine_move(game, board.move_stack) and not is_game_over(game):
        with chess.polyglot.open_reader("book.bin") as reader:
            movesob = []
            weight = []
            for entry in reader.find_all(board):
                movesob.append(entry.move)
                weight.append(entry.weight)
        if len(weight) == 0 or max(weight) < 9:
            move = engineeng.play(board, engine.Limit(time=timep))
            board.push(move.move)
            li.make_move(game.id, move.move)
            time.sleep(delay_seconds)
        else:
            move = movesob[weight.index(max(weight))]
            board.push(move)
            li.make_move(game.id, move)

    with chess.polyglot.open_reader("book.bin") as reader:
        while not terminated:
            try:
                binary_chunk = next(lines)
            except StopIteration:
                break
            upd = json.loads(binary_chunk.decode("utf-8")) if binary_chunk else None
            u_type = upd["type"] if upd else "ping"
            if not board.is_game_over():
                if u_type == "gameState":
                    game.state = upd
                    moves = upd["moves"].split()
                    board = update_board(board, moves[-1])
                    if not is_game_over(game) and is_engine_move(game, moves):
                        if chess.popcount(board.occupied) <= 7:
                            move = egtb_move(li, board, game)
                            if move != None:
                                board.push(move)
                                li.make_move(game.id, move)
                            else:
                                moves = []
                            weight = []
                            for entry in reader.find_all(board):
                                moves.append(entry.move)
                                weight.append(entry.weight)
                            if len(weight) == 0 or max(weight) < 9:
                                timelim = (
                                    game.state["wtime"] / 1000
                                    if game.is_white
                                    else game.state["btime"] / 1000
                                )
                                divtime = 85 - int(len(board.move_stack) / 2)
                                if divtime < 1:
                                    timep = 1
                                else:
                                    timep = round(timelim / divtime, 1)
                                    if timep > 10:
                                        timep = 10
                                    elif timep < 0.3:
                                        timep = 0.3
                                move = engineeng.play(board, engine.Limit(time=timep))
                                board.push(move.move)
                                li.make_move(game.id, move.move)
                                time.sleep(delay_seconds)
                            else:
                                move = moves[weight.index(max(weight))]
                                board.push(move)
                                li.make_move(game.id, move)
                        else:
                            moves = []
                            weight = []
                            for entry in reader.find_all(board):
                                moves.append(entry.move)
                                weight.append(entry.weight)
                            if len(weight) == 0 or max(weight) < 9:
                                timelim = (
                                    game.state["wtime"] / 1000
                                    if game.is_white
                                    else game.state["btime"] / 1000
                                )
                                divtime = 85 - int(len(board.move_stack) / 2)
                                if divtime < 1:
                                    timep = 1
                                else:
                                    timep = round(timelim / divtime, 1)
                                    if timep > 10:
                                        timep = 10
                                    elif timep < 0.3:
                                        timep = 0.3
                                move = engineeng.play(board, engine.Limit(time=timep))
                                board.push(move.move)
                                li.make_move(game.id, move.move)
                                time.sleep(delay_seconds)
                            else:
                                move = moves[weight.index(max(weight))]
                                board.push(move)
                                li.make_move(game.id, move)

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
    logger.info("game over")
    engineeng.quit()


def is_white_to_move(game, moves):
    return len(moves) % 2 == (0 if game.white_starts else 1)


def setup_board(game):
    if game.variant_name.lower() == "chess960":
        board = chess.Board(game.initial_fen, chess960=True)
    elif game.variant_name == "From Position":
        board = chess.Board(game.initial_fen)
    else:
        VariantBoard = variant.find_variant(game.variant_name)
        board = VariantBoard()
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


def egtb_move(li, board, game):
    try:
        if board.uci_variant not in ["chess", "antichess", "atomic"]:
            return None
        name_to_wld = {
            "loss": -2,
            "maybe-loss": -1,
            "blessed-loss": -1,
            "draw": 0,
            "cursed-win": 1,
            "maybe-win": 1,
            "win": 2,
        }
        max_pieces = 7 if board.uci_variant == "chess" else 6
        variant = "standard" if board.uci_variant == "chess" else board.uci_variant
        if chess.popcount <= max_pieces:
            data = li.api_get(
                f"http://tablebase.lichess.ovh/{variant}?fen={board.fen()}"
            )
            move = data["moves"][0]["uci"]
            wdl = name_to_wld[data["moves"][0]["category"]] * -1
            dtz = data["moves"][0]["dtz"] * -1
            dtm = data["moves"][0]["dtm"]
            if dtm:
                dtm *= -1
            if wdl != None:
                return move
            else:
                return None
    except:
        return None


def intro():
    return r"""
    .   _/|
    .  // o\
    .  ||  _)  lichess-bot
    .  //__\
    .  )___(   Play on Lichess with a bot
    """


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play on Lichess with a bot")
    parser.add_argument(
        "-u",
        action="store_true",
        help="Add this flag to upgrade your account to a bot account.",
    )
    parser.add_argument(
        "-v",
        action="store_true",
        help="Verbose output. Changes log level from INFO to DEBUG.",
    )
    parser.add_argument(
        "--config", help="Specify a configuration file (defaults to ./config.yml)"
    )
    parser.add_argument(
        "-l", "--logfile", help="Log file to append logs to.", default=None
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.v else logging.INFO,
        filename=args.logfile,
        format="%(asctime)-15s: %(message)s",
    )
    logger.info(intro())
    CONFIG = load_config(args.config or "./config.yml")
    li = lichess.Lichess(CONFIG["token"], CONFIG["url"], "1.2.0")

    user_profile = li.get_profile()
    username = user_profile["username"]
    is_bot = user_profile.get("title") == "BOT"
    logger.info("Welcome {}!".format(username))

    if not is_bot:
        is_bot = upgrade_account(li)

    if is_bot:
        start(li, user_profile, CONFIG)
    else:
        logger.error(
            "{} is not a bot account. Please upgrade it to a bot account!".format(
                user_profile["username"]
            )
        )
