# Run with python -m test.run_pgn

from pgn_to_positions import str_to_game


if __name__ == "__main__":
    input_stream = open("data/sample-pgn.txt", "r").read()
    print(input_stream)
    print(str_to_game(input_stream))
