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

sys.path.append("..")
from Library import Model
from Models import load_architecture, load_date_range

import pandas as pd

date_range = load_date_range(architecture_id, date_range_id)
architecture = load_architecture(architecture_id)
architecture.setdefault("batch_size", train_batch_size)

df = pd.read_parquet(paths["artifact_root"] + "Paths.parquet")
train_df, val_df = date_range.select_pairs(df)

Model.train_model(
    train_df,
    val_df=val_df,
    keep_every=date_range.keep_every,
    model_params=architecture,
    path=BASE_DIR + "Outputs/Models/" + (architecture_id + date_range_id) + ".keras",
)
