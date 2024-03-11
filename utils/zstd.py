import pyzstd
import argparse

parser = argparse.ArgumentParser(
    prog="Zstd utils", description="Compresses and decompresses zstd files"
)
parser.add_argument(
    "--input",
    "-i",
    type=argparse.FileType("rb"),
    required=True,
)
parser.add_argument("--method", "-m", choices=["compress", "decompress"])
parser.add_argument("--output", "-o", type=argparse.FileType("wb"))

args = parser.parse_args()

print(args)

if args.method == "compress":
    pyzstd.compress_stream(args.input, args.output)
elif args.method == "decompress":
    pyzstd.decompress_stream(args.input, args.output)
