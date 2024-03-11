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

if args.method == "compress" or (args.method is None and args.input.endswith(".txt")):
    if args.output is None:
        args.output = open(args.input.name.replace(".txt", ".zst"), "wb")
    print("Compressing", args.input.name, "to", args.output.name)
    pyzstd.compress_stream(args.input, args.output)
elif args.method == "decompress" or (
    args.method is None and args.input.endswith(".zst")
):
    if args.output is None:
        args.output = open(
            args.input.name.replace(".zstd", ".txt").replace(".zst", ".txt"), "wb"
        )
    print("Decompressing", args.input.name, "to", args.output.name)
    pyzstd.decompress_stream(args.input, args.output)
