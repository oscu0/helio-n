#!/usr/bin/env python3
from Library.Config import paths

import sys

sys.path.append("..")
from Library.IO import prepare_dataset

try:
    hourly = sys.argv[1]
except IndexError:
    hourly = True

prepare_dataset(
    paths["fits_root"],
    paths["masks_root"],
    hmi_root=paths["hmi_root"],
    hourly=hourly
)
