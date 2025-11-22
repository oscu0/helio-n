#!/usr/bin/env python3
import sys

if len(sys.argv) != 3:
    print("Usage: ./Apply.py <start> <end>")
    sys.exit(1)

start = sys.argv[1]
end = sys.argv[2]

import os
os.path.append("..")

import json
import pandas as pd

from Library.Model import load_trained_model
from Library.Processing import *

with open("./Config/Paths.json", "r") as f:
    paths = json.load(f)

model = load_trained_model(paths["model_path"])

df = pd.read_parquet("../Data/df.parquet")

df = df[start:end]

if len(df) == 0:
    print("Desired time range is empty, likely due to incorrect date range specification. Use YYYYMMHH.")
    sys.exit(1)
else:
    df.apply(save_pmap)
