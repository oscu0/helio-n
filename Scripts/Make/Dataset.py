#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

from Library.Config import paths
from Library.IO import prepare_dataset


def main(argv):
    parser = argparse.ArgumentParser(description="Build the FITS/mask dataset index.")
    parser.add_argument(
        "hourly",
        nargs="?",
        choices=("hourly",),
        help="Keep only one observation per hour.",
    )
    parser.add_argument("--start", help="inclusive YYYYMMDD date")
    parser.add_argument("--end", help="inclusive YYYYMMDD date")
    args = parser.parse_args(argv[1:])
    assert (args.start is None) == (args.end is None), (
        "Pass --start and --end together."
    )

    prepare_dataset(
        paths["fits_root"],
        paths["masks_root"],
        hmi_root=paths["hmi_root"],
        aia304_root=paths["aia304_root"],
        hourly=args.hourly == "hourly",
        start=args.start,
        end=args.end,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
