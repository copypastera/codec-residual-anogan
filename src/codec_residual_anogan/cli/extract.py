"""CLI for manifest-driven codec-residual extraction."""
import argparse

from ..features import extract_manifest


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract codec-residual features from a CSV/TSV manifest")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-split", required=True)
    parser.add_argument(
        "--manifest-split",
        help="optional value from the manifest split column")
    parser.add_argument("--bitrate", default="16k")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    return parser.parse_args()


def main():
    args = parse_args()
    extract_manifest(
        manifest=args.manifest,
        output_dir=args.output_dir,
        output_split=args.output_split,
        manifest_split=args.manifest_split,
        bitrate=args.bitrate,
        workers=args.workers,
        checkpoint_every=args.checkpoint_every)


if __name__ == "__main__":
    main()
