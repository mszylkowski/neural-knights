import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs
from chess import BLACK, STARTING_FEN, Board
import torch

from model import NeuralKnight
from play import get_model_move

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESET_ANCHOR = "<a href='/'>Start again</a>"

hostName = "localhost"
serverPort = 8080


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


args = get_args()
model = NeuralKnight()
model.to(DEVICE)
model.load_state_dict(torch.load(args.model, map_location=DEVICE))


class MyServer(BaseHTTPRequestHandler):
    def do_GET(self):
        queryparams = parse_qs(self.path[2:])
        fen = queryparams["fen"][0] if "fen" in queryparams else STARTING_FEN
        move = queryparams["move"][0] if "move" in queryparams else None
        board = Board(fen=fen)
        notes = []
        if move:
            try:
                board.push_san(move)
                if board.is_game_over():
                    notes.append("Game over! " + RESET_ANCHOR + board.result())
                else:
                    move = get_model_move(board, BLACK, model)
                    notes.append(f"Neural Knight responded with <b>{move}</b>")
                    if board.is_game_over():
                        notes.append("Game over! " + RESET_ANCHOR + board.result())
            except ValueError as e:
                notes.append(e)
        else:
            notes.append("No move provided")
        print(notes)
        self.send_response(200)
        self.send_header("Content-type", "text/html")
        self.end_headers()
        self.wfile.write(
            bytes(
                "<html><head>"
                "<title>Neural Knights</title>"
                "<link rel='stylesheet' "
                "href='https://cdn.jsdelivr.net/npm/@picocss/pico@2/css/pico.classless.blue.min.css' />"
                "<style>svg{width:100%;height:auto;max-height:60vmin}</style>"
                "</head>",
                "utf-8",
            )
        )
        self.wfile.write(b"<body><header><h1>Neural Knights</h1></header><main>")
        self.wfile.write(board._repr_svg_().encode("utf-8"))
        self.wfile.write(b"<br><br>")
        if board.is_game_over():
            self.wfile.write(
                bytes(
                    "<p>Game over! "
                    + RESET_ANCHOR
                    + board.result()
                    + "</p><form><input type='submit' value='Reset game'></form>",
                    "utf-8",
                )
            )
        else:
            self.wfile.write(
                bytes(
                    "<form><fieldset>"
                    f"<label for='move'>Move</label>"
                    f"<input type='text' name='move' id='move' autofocus autocomplete>"
                    "<small>Use SAN notation. Eg: <code>d4</code>, <code>Nf3</code>, <code>O-O-O</code></small>"
                    f"<input type='hidden' name='fen' value='{board.board_fen()}'>"
                    "</fieldset></form>",
                    "utf-8",
                )
            )
        self.wfile.write(
            bytes(
                f"<p>Notes: </p><ul>{"".join(map(lambda x: f"<li>{x}</li>", notes))}</ul>",
                "utf-8",
            )
        )
        self.wfile.write(b"</body></body></html>")


if __name__ == "__main__":
    webServer = HTTPServer((hostName, serverPort), MyServer)
    print("Server started http://%s:%s" % (hostName, serverPort))

    try:
        webServer.serve_forever()
    except KeyboardInterrupt:
        pass

    webServer.server_close()
    print("Server stopped.")
