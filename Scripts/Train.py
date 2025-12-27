#!/usr/bin/env python3

from pathlib import Path

from Library.Config import paths

SCRIPT_DIR = Path(__file__).resolve().parent

import sys

sys.path.append("..")
from Library import Model

import pandas as pd

train_df = pd.read_parquet(paths["artifact_root"] + "train_df.parquet")

Model.train_model(train_df)
