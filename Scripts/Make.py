#!/usr/bin/env python3


from Library.Config import paths


import sys

sys.path.append("..")
from Library.IO import prepare_dataset


prepare_dataset(
    paths["fits_root"],
    paths["masks_root"],
    pmaps_root=paths["masks_root"],
    hmi_root=paths["hmi_root"],
)
