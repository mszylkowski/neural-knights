# Chess Games

The chess games are downloaded from [Lichess Database](https://database.lichess.org/#standard_games) in PGN compressed with [zstd](http://facebook.github.io/zstd/) format (a fast lossless compression algorithm). The `zstd` files can be decoded quickly by blocks, so the algorithm can stream and encode PGN games from these files without having larger files in memory (or in disk) at once.

_Note: If we were to use the PGN files uncompressed, they would be 7.1 times larger each._

## Download instructions

Go to the [Lichess Database](https://database.lichess.org/#standard_games) and download any `.pgn.zst` into this folder.