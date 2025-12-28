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
import sunpy.map
from concurrent.futures import ThreadPoolExecutor, wait, Future
from collections import deque


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
    cpu_workers = os.cpu_count() or 12
    # plot_workers = max(1, min(batch_size * 3, cpu_workers))
    plot_workers = model.architecture["plot_threads"]
    cpu_workers = os.cpu_count() or 12
    # plot_workers = max(1, min(batch_size * 3, cpu_workers))
    plot_workers = model.architecture["plot_threads"]

    # Limit how many batches we keep in-flight for plotting to avoid memory bloat
    max_inflight_batches = model.architecture["max_inflight_batches"] 
    rows = list(df.itertuples())
    pmap_paths = []

    def plot_batch(batch_rows, probs):
        results = [None] * len(batch_rows)

        def worker(row, pmap, base_map):
            try:
                path = pmap_path(row, model.architecture_id, model.date_range_id)
                np.save(path, pmap)
                save_ch_map_unet(
                    row,
                    model,
                    postprocessing,
                    pmap,
                    None,
                    base_map,
                )
                save_ch_map_unet(
                    row,
                    model,
                    "P0",
                    pmap,
                    None,
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
            except Exception as e:
                print(f"Error processing {row.Index}: {e}")
                return None

        futures = []
        for (i, row), prob in zip(enumerate(batch_rows), probs):
            try:
                pmap = prob[..., 0]
                try:
                    base_map = sunpy.map.Map(row.fits_path)
                except Exception as e:
                    print(f"Error loading base map {row.Index}: {e}")
                    base_map = None
                futures.append(executor.submit(worker, row, pmap, base_map))
            except Exception as e:
                print(f"Error preparing plotting for {row.Index}: {e}")
                futures.append(None)

        for i, fut in enumerate(futures):
            if fut is None:
                continue
            try:
                results[i] = fut.result()
            except Exception as e:
                print(f"Error during plotting {batch_rows[i].Index}: {e}")
        return results

    inflight: deque[Future] = deque()

    with ThreadPoolExecutor(max_workers=plot_workers) as executor:
        with tqdm(total=len(rows), desc="Generating pmaps") as pbar:
            for offset in range(0, len(rows), batch_size):
                batch_rows = rows[offset : offset + batch_size]

                # Predict (GPU) in main thread
                imgs = []
                for row in batch_rows:
                    try:
                        imgs.append(prepare_fits(row.fits_path))
                    except Exception as e:
                        print(f"Error loading {row.Index}: {e}")
                        imgs.append(None)
                # filter out failed loads
                valid = [(r, img) for r, img in zip(batch_rows, imgs) if img is not None]
                if not valid:
                    pbar.update(len(batch_rows))
                    continue
                batch_rows_valid, imgs_valid = zip(*valid)
                x = np.stack(imgs_valid)[..., np.newaxis].astype(np.float32)
                try:
                    probs = model.predict(x)
                except Exception as e:
                    print(f"Error batch predicting starting at {batch_rows[0].Index}: {e}")
                    pbar.update(len(batch_rows))
                    continue

                # Submit plotting to thread pool
                fut = executor.submit(plot_batch, batch_rows_valid, probs)
                inflight.append(fut)

                # If too many in-flight batches, wait for the oldest
                while len(inflight) >= max_inflight_batches:
                    oldest = inflight.popleft()
                    try:
                        res = oldest.result()
                        pmap_paths.extend(res)
                    except Exception as e:
                        print(f"Error in plotting batch: {e}")

                pbar.update(len(batch_rows))

            # drain remaining
            while inflight:
                fut = inflight.popleft()
                try:
                    res = fut.result()
                    pmap_paths.extend(res)
                except Exception as e:
                    print(f"Error in plotting batch: {e}")
