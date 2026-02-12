#!/usr/bin/env python3
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

from Library.Config import paths
from Library.IO import prepare_dataset


def main(argv):
    if len(argv) == 1:
        print("Usage: python Scripts/Make.py Dataset [hourly]")

    hourly = False
    if len(argv) > 1:
        if argv[1].lower() == "hourly":
            hourly = True
        else:
            print(f"Unknown argument: {argv[1]}")
            return 1

    prepare_dataset(
        paths["fits_root"],
        paths["masks_root"],
        hmi_root=paths["hmi_root"],
        aia304_root=paths["aia304_root"],
        hourly=hourly,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
