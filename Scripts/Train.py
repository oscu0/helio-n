#!/usr/bin/env python3
import sys
from pathlib import Path

BASE_DIR = str(Path(__file__).resolve().parent.parent) + "/"

if len(sys.argv) < 3:
    print("Usage: python Scripts/Train.py <architecture_id> <date_range_id>")
    print("Example: python Scripts/Train.py A2 D1")
    sys.exit(1)

architecture_id = sys.argv[1]
date_range_id = sys.argv[2]

from Library.Config import paths, train_batch_size

import sys

sys.path.append("..")
from Library import Model

import pandas as pd
import json

date_range = json.load(
    open(BASE_DIR + "Config/Model/Date Range/" + (date_range_id + ".json"))
)

architecture = json.load(
    open(BASE_DIR + "Config/Model/Architecture/" + (architecture_id + ".json"))
)
architecture["batch_size"] = train_batch_size

df = pd.read_parquet(paths["artifact_root"] + "Paths.parquet")
train_df = df[date_range["start"] : date_range["end"]]

Model.train_model(
    train_df,
    keep_every=date_range["keep_every"],
    model_params=architecture,
    path=BASE_DIR + "Outputs/Models/" + (architecture_id + date_range_id) + ".keras",
)
