# Chess Games

The chess games are downloaded from [Lichess Database](https://database.lichess.org/#standard_games) in PGN compressed with [zstd](http://facebook.github.io/zstd/) format (a fast lossless compression algorithm). The `zstd` files can be decoded quickly by blocks, so the algorithm can stream and encode PGN games from these files without having larger files in memory (or in disk) at once.

_Note: If we were to use the PGN files uncompressed, they would be 7.1 times larger each._

## Format

PGN games are encoded in zstd format, directly downloaded from Lichess Open Games Database. Go to the [Lichess Database](https://database.lichess.org/#standard_games) and download any `.pgn.zst` into this folder.

PGN games contain tags with metadata and the moves (sometimes with time and advantage). Example:

```
[Event "Rated Bullet tournament https://lichess.org/tournament/yc1WW2Ox"]
[Site "https://lichess.org/PpwPOZMq"]
[Date "2017.04.01"]
[White "Abbot"]
[Black "Costello"]
[Result "0-1"]
[WhiteElo "2100"]
[BlackElo "2000"]
[ECO "B30"]
[Opening "Sicilian Defense: Old Sicilian"]
[TimeControl "300+0"]
[Termination "Time forfeit"]

1. e4 c5 2. Nf3 Nc6 .3 Bc4 e6 4. c3 4... b5? 5. Bb3?! c4 1-0
```

## Files in this directory

Developers that want to run many of the scripts will need to download into this folder:

- `lichess_db_standard_rated_<date>.pgn.zst`: Files used for training the models. This is the default name from [Lichess Database](https://database.lichess.org).
- `validation_lichess_db_standard_rated_<date>.pgn.zst`: By prepending "validation_" to the file name, the file will be used for testing and not for training.
- `lichess_db_puzzle.csv.zst`: Files used for `puzzle.py`, downloaded from [Lichess Database of Puzzles](https://database.lichess.org/#puzzles) directly.
