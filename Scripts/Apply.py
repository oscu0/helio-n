#!/usr/bin/env python3
import sys

if len(sys.argv) != 3:
    print("Usage: ./Apply.py <start> <end>")
    sys.exit(1)

start = sys.argv[1]
end = sys.argv[2]

import sys
sys.path.append("..")

import pandas as pd

from pathlib import Path


from Library.Model import load_trained_model
from Library.Processing import *
from Library.Config import *


model = load_trained_model(paths["model_path"])

df = pd.read_parquet("./Data/df.parquet")

df = df[start:end]

if len(df) == 0:
    print("Desired time range is empty, likely due to incorrect date range specification. Use YYYYMMHH.")
    sys.exit(1)
else:
    df.apply(lambda x: save_pmap(model, x), axis=1)
