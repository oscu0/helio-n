#!/usr/bin/env python3
import sys
from tqdm import tqdm

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
from Library.Config import paths


model = load_trained_model(paths["model_path"])

df = pd.read_parquet(paths["artifact_root"] + "df.parquet")

df = df[start:end]

if len(df) == 0:
    print("Desired time range is empty, likely due to incorrect date range specification. Use YYYYMMHH.")
    sys.exit(1)
else:
    pmap_paths = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating pmaps"):
        try:
            path = save_pmap(model, row)   # your existing function
        except Exception as e:
            print(f"Error processing {row.name}: {e}")
            path = None
        pmap_paths.append(path)

    df["pmap_path"] = pmap_paths
