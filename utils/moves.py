import re

NUMBER_RE = re.compile(r"\d")
NUM_OF_SQUARES = 64
# Pawns, Rooks, Knights, Bishops, Queens, and Kings X 2 players.
NUM_OF_PIECE_TYPES = 12

# Used to pad sequence of moves after the game ended for transformer training.
PAD_BOARD = np.zeros((NUM_OF_PIECE_TYPES, 8, 8))
PAD_MOVE = -1


def __flatten(xss):
    return [x for xs in xss for x in xs]


def get_all_moves():
    positive_deltas = __flatten(
        (
            # King movements (distance 1)
            [(0, 1), (1, 1), (1, 0)],
            # Rook movements
            [(0, 2), (0, 3), (0, 4), (0, 5), (0, 6), (0, 7)],
            [(2, 0), (3, 0), (4, 0), (5, 0), (6, 0), (7, 0)],
            # Bishop movements
            [(2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7)],
            # Knight movements
            [(1, 2), (2, 1)],
            # Pawns moves are covered by the king (forward and en-passant) and rook (2-forward)
        )
    )
    all_deltas = __flatten(
        [((a, b), (-a, b), (a, -b), (-a, -b)) for a, b in positive_deltas]
    )

    piece_movements = set(["O-O", "O-O-O"])
    board_locations = [
        [letter + number for number in "12345678"] for letter in "abcdefgh"
    ]
    for i, row in enumerate(board_locations):
        for j, square in enumerate(row):
            for a, b in all_deltas:
                if 0 <= i + a < 8 and 0 <= j + b < 8:
                    piece_movements.add(square + board_locations[i + a][j + b])
    return sorted(piece_movements)


__moves = get_all_moves()
__moves_to_idx = {move: idx for idx, move in enumerate(__moves)}


def encode(move: str) -> int:
    """Converts a UCI move to an index."""
    if move.startswith("O"):
        return __moves_to_idx[move]
    return __moves_to_idx[move[:4]]


def decode(move_idx: int) -> str:
    """Converts an index to a UCI move."""
    return __moves[move_idx]


def mirror_move(uci_move: str) -> str:
    """Mirrors a UCI move."""
    if uci_move.startswith("O"):
        return uci_move
    return NUMBER_RE.sub(lambda x: str(9 - int(x.group())), uci_move)


if __name__ == "__main__":
    print(sorted(__moves))
    print(len(__moves))
