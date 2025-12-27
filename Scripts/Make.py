#!/usr/bin/env python3
from Library.Config import paths

import sys

architecture = sys.argv[1]
date_range = sys.argv[2]

if not architecture or not date_range:
    # will produce empty pmap column, useful for first runs
    architecture = date_range = ""

sys.path.append("..")
from Library.IO import prepare_dataset


prepare_dataset(
    paths["fits_root"],
    paths["masks_root"],
    pmaps_root=paths["masks_root"],
    hmi_root=paths["hmi_root"],
    architecture=architecture,
    date_range=date_range,
)
