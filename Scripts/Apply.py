#!/usr/bin/env python3
import os
import sys
from collections import deque
from concurrent.futures import ThreadPoolExecutor, Future

import tensorflow as tf
import numpy as np
import pandas as pd
import sunpy.map
from tqdm import tqdm

sys.path.append("..")

import matplotlib

matplotlib.use("Agg")

from Library.Model import load_trained_model
from Library.Processing import prepare_fits, save_pmap
from Library.Plot import save_ch_map_unet, save_ch_mask_only_unet
from Library.CH import generate_omask
from Library.IO import pmap_path
from Library.Config import paths, apply_config


def main():
    architecture = sys.argv[1]
    date_range = sys.argv[2]
    postprocessing = sys.argv[3]
    start = sys.argv[4]
    end = sys.argv[5]

    model = load_trained_model(architecture, date_range)

    df = pd.read_parquet(paths["artifact_root"] + "Paths.parquet")
    df = df[start:end]

    if len(df) == 0:
        print(
            "Desired time range is empty, likely due to incorrect date range specification. Use YYYYMMHH."
        )
        sys.exit(1)

    batch_size = apply_config["batch_size"]
    plot_workers = apply_config["plot_threads"]
    max_inflight_plots = apply_config["max_inflight_plots"]
    target_size = model.architecture["img_size"]

    rows = list(df.itertuples())
    pmap_paths = []
    inflight: deque[Future] = deque()

    def plot_row(row, pmap, base_map):
        path = pmap_path(row, model.architecture_id, model.date_range_id)
        oval = generate_omask(row)
        save_ch_map_unet(
            row,
            model,
            postprocessing,
            pmap,
            oval,
            base_map,
        )
        save_ch_map_unet(
            row,
            model,
            "P0",
            pmap,
            oval,
            base_map,
        )
        save_ch_mask_only_unet(
            row,
            model,
            postprocessing,
            pmap,
            True,
        )
        return path

    with ThreadPoolExecutor(max_workers=plot_workers) as executor:
        with tqdm(total=len(rows), desc="Generating pmaps") as pbar:
            for offset in range(0, len(rows), batch_size):
                batch_rows = rows[offset : offset + batch_size]

                imgs = []
                valid_rows = []
                base_maps = []
                for row in batch_rows:
                    try:
                        base_map, img = prepare_fits(row.fits_path)
                        imgs.append(img)
                        valid_rows.append(row)
                        base_maps.append(base_map)
                    except Exception as e:
                        print(f"Error loading {row.Index}: {e}")

                if not imgs:
                    pbar.update(len(batch_rows))
                    continue

                x = np.stack(imgs)[..., np.newaxis].astype(np.float32)
                if x.shape[1] != target_size:
                    x = tf.image.resize(
                        x, [target_size, target_size], method="bilinear"
                    ).numpy()

                try:
                    probs = model.compiled_infer(x)
                except Exception as e:
                    print(
                        f"Error batch predicting starting at {batch_rows[0].Index}: {e}"
                    )
                    pbar.update(len(batch_rows))
                    continue

                for row, prob, base_map in zip(valid_rows, probs, base_maps):
                    pmap = prob[..., 0]
                    save_pmap(model, row, pmap)
                    fut = executor.submit(plot_row, row, pmap, base_map)
                    inflight.append(fut)

                    while len(inflight) >= max_inflight_plots:
                        oldest = inflight.popleft()
                        try:
                            res = oldest.result()
                            pmap_paths.append(res)
                        except Exception as e:
                            print(f"Error in plotting task: {e}")

                pbar.update(len(batch_rows))

            while inflight:
                fut = inflight.popleft()
                try:
                    res = fut.result()
                    pmap_paths.append(res)
                except Exception as e:
                    print(f"Error in plotting task: {e}")


if __name__ == "__main__":
    main()
