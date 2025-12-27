#!/usr/bin/env python3
import sys
from pathlib import Path

BASE_DIR = str(Path(__file__).resolve().parent.parent) + "/"

architecture = sys.argv[1]
date_range = sys.argv[2]

if date_range is None or architecture is None:
    print(
        "Usage: ./Train.py <experiment>, where <experiment> is the name of a config in Config/Model/Date Range/"
    )
    sys.exit(1)

from Library.Config import paths

import sys

sys.path.append("..")
from Library import Model

import pandas as pd
import json

date_range_json = json.load(
    open(BASE_DIR + "Config/Model/Date Range/" + (date_range + ".json"))
)

architecture_json = json.load(
    open(BASE_DIR + "Config/Model/Architecture/" + (architecture + ".json"))
)

df = pd.read_parquet(paths["artifact_root"] + "Paths.parquet")
train_df = df[date_range_json["start"] : date_range_json["end"]]

Model.train_model(
    train_df,
    keep_every=date_range_json["keep_every"],
    model_params=architecture_json,
    path=BASE_DIR + "Outputs/Models" + (architecture + date_range) + ".keras",
)
