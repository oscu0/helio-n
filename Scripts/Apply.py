#!/usr/bin/env python3
import os
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
from Library.Plot import save_ch_map_unet, save_ch_mask_only_unet
from Library.Config import *
from Library.Config import paths
from concurrent.futures import ThreadPoolExecutor, wait


model = load_trained_model(architecture, date_range)

df = pd.read_parquet(paths["artifact_root"] + "Paths.parquet")

df = df[start:end]

if len(df) == 0:
    print(
        "Desired time range is empty, likely due to incorrect date range specification. Use YYYYMMHH."
    )
    sys.exit(1)
else:
    batch_size = model.architecture["apply_batch_size"]
    env_workers = os.environ.get("APPLY_PLOT_THREADS")
    if env_workers:
        plot_workers = int(env_workers)
    else:
        cpu_workers = os.cpu_count() or 4
        plot_workers = max(1, min(batch_size * 3, cpu_workers))
    rows = list(df.itertuples())
    pmap_paths = []

    def process_batch(batch_rows, executor):
        results = [None] * len(batch_rows)
        imgs = []
        metas = []
        for i, row in enumerate(batch_rows):
            try:
                imgs.append(prepare_fits(row.fits_path))
                metas.append((i, row))
            except Exception as e:
                print(f"Error loading {row.Index}: {e}")
        if not imgs:
            return results

        x = np.stack(imgs)[..., np.newaxis].astype(np.float32)
        try:
            probs = model.predict(x)
        except Exception as e:
            print(f"Error batch predicting starting at {batch_rows[0].Index}: {e}")
            return results

        for (i, row), prob in zip(metas, probs):
            try:
                pmap = prob[..., 0]
                path = pmap_path(row, model.architecture_id, model.date_range_id)
                np.save(path, pmap)
                futures = [
                    executor.submit(
                        save_ch_map_unet,
                        row,
                        model,
                        pmap,
                        postprocessing,
                    ),
                    executor.submit(
                        save_ch_map_unet,
                        row,
                        model,
                        pmap,
                        "P0",
                    ),
                    executor.submit(
                        save_ch_mask_only_unet,
                        row,
                        model,
                        pmap,
                        postprocessing,
                    ),
                ]
                done, _ = wait(futures)
                for f in done:
                    try:
                        f.result()
                    except Exception as e:
                        print(f"Error during plotting {row.Index}: {e}")
                results[i] = path
            except Exception as e:
                print(f"Error processing {row.Index}: {e}")
        return results

    with ThreadPoolExecutor(max_workers=plot_workers) as executor:
        with tqdm(total=len(rows), desc="Generating pmaps") as pbar:
            for offset in range(0, len(rows), batch_size):
                batch_rows = rows[offset : offset + batch_size]
                batch_results = process_batch(batch_rows, executor)
                pmap_paths.extend(batch_results)
                pbar.update(len(batch_rows))
