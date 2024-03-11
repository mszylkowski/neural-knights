# Chess Games

The chess games are downloaded from [Lichess Database](https://database.lichess.org/#standard_games) in PGN compressed with [zstd](http://facebook.github.io/zstd/) format (a fast lossless compression algorithm). The `zstd` files can be decoded quickly by blocks, so the algorithm can stream and encode PGN games from these files without having larger files in memory (or in disk) at once.

_Note: If we were to use the PGN files uncompressed, they would be 7.1 times larger each._

## Files in this directory

### `lichess_db_standard_rated_<date>.pgn.zst`

PGN games encoded in zstd format. Directly downloaded from Lichess Open Games Database. Go to the [Lichess Database](https://database.lichess.org/#standard_games) and download any `.pgn.zst` into this folder.

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

### `fen_<elo>_<date>.fen.txt`

All positions of the PGN file for that date that are within 100 elo of the specified elo. It's a CSV file, where the first column is the board position, and the second column was the move made.

```
2k1r2r/2pb1pp1/1p5p/pP2nq2/Pb1B4/1BN5/5PPP/R1Q1R1K1,d1c1
2k1r2r/2pb1pp1/1p5p/pP3q2/Pb1B4/1BNn4/5PPP/R1Q1R1K1,e5d3
2k1r2r/2pb1pp1/1p5p/pP3q2/Pb1B4/2Nn4/2B2PPP/R1Q1R1K1,b3c2
```

### `.fen.zst`

Compressed version of the file `fen_<elo>_<date>.fen.zst`. Use [zstd.py as a CLI](../utils/zstd.py) to decompress the files.
