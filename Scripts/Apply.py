#!/usr/bin/env python3
import sys
from tqdm import tqdm

architecture = sys.argv[1]
date_range = sys.argv[2]
postprocessing = sys.argv[3]
start = sys.argv[4]
end = sys.argv[5]

import sys

sys.path.append("..")

import pandas as pd


from Library.Model import load_trained_model
from Library.Processing import *
from Library.Plot import save_ch_map_unet
from Library.Config import *
from Library.Config import paths


model = load_trained_model(architecture, date_range)

df = pd.read_parquet(paths["artifact_root"] + "Paths.parquet")

df = df[start:end]

if len(df) == 0:
    print(
        "Desired time range is empty, likely due to incorrect date range specification. Use YYYYMMHH."
    )
    sys.exit(1)
else:
    pmap_paths = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating pmaps"):
        try:
            path, pmap = save_pmap(model, row)  # your existing function
            save_ch_map_unet(row, model, pmap=pmap, postprocessing=postprocessing)
            save_ch_map_unet(row, model, pmap=pmap, postprocessing="P0")

        except Exception as e:
            print(f"Error processing {row.name}: {e}")
            path = None
        pmap_paths.append(path)

    df["pmap_path"] = pmap_paths
