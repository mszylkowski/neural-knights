import argparse

from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs
from chess import Board
import torch

from play import get_model_move
from utils.load_model import load_model_from_saved_run

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESET_ANCHOR = "<a href='/'>Start again</a>"

hostName = "localhost"
serverPort = 8080


def get_args():
    parser = argparse.ArgumentParser(
        prog="Play", description="Plays a game of Chess against Neural Knight"
    )
    parser.add_argument(
        "--run",
        "-r",
        type=str,
        default="runs/resnet_6b_64f.pt",
        help="File path of the model. Should be `runs/*.pt`.",
    )
    return parser.parse_args()


args = get_args()
path_to_run = args.run
model = load_model_from_saved_run(path_to_run, device=DEVICE)
model.eval()


class MyServer(BaseHTTPRequestHandler):
    def do_GET(self):
        queryparams = parse_qs(self.path.split("?").pop())
        print(queryparams)
        if self.path == "/style.css":
            self.send_response(200)
            self.wfile.write(open("web/style.css", "rb").read())
            return

        fen = queryparams["fen"][0] if "fen" in queryparams else None
        board = Board(fen=fen)
        move = ""
        if fen and not board.is_game_over():
            move = get_model_move(
                board,
                board.turn,
                model,
            ).uci()
        elif board.is_game_over():
            move = "game_over"

        if self.path.startswith("/getmove?"):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(bytes(move, "utf-8"))
            return
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()

        with open("web/index.html", "r") as f:
            content = f.read()
        self.wfile.write(
            bytes(
                content,
                "utf-8",
            )
        )


if __name__ == "__main__":
    webServer = HTTPServer((hostName, serverPort), MyServer)
    print("Server started http://%s:%s" % (hostName, serverPort))

    try:
        webServer.serve_forever()
    except KeyboardInterrupt:
        pass

    webServer.server_close()
    print("Server stopped.")
