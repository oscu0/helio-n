#!/usr/bin/env python3
import os
import sys
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, Future
from collections import deque
from types import SimpleNamespace

import numpy as np
import pandas as pd
import sunpy.map
from tqdm import tqdm

sys.path.append("..")

from Library.Model import load_trained_model
from Library.Processing import prepare_fits
from Library.Plot import save_ch_map_unet, save_ch_mask_only_unet
from Library.CH import generate_omask
from Library.IO import pmap_path
from Library.Config import paths


def render_batch(task):
    """
    Worker process: load saved pmaps and generate outputs.
    Expects task dict with keys: rows (list of dicts), postprocessing, arch_id, date_id, fast_mask.
    """
    postprocessing = task["postprocessing"]
    arch_id = task["arch_id"]
    date_id = task["date_id"]
    fast_mask = task.get("fast_mask", True)
    rows = task["rows"]
    results = []

    for info in rows:
        try:
            pmap = np.load(info["pmap_path"])
            base_map = sunpy.map.Map(info["fits_path"])
            row_obj = SimpleNamespace(
                fits_path=info["fits_path"], mask_path=info["mask_path"]
            )
            oval = generate_omask(row_obj)

            save_ch_map_unet(
                row_obj,
                None,
                postprocessing,
                pmap,
                oval,
                base_map,
                arch_id,
                date_id,
            )
            save_ch_map_unet(
                row_obj,
                None,
                "P0",
                pmap,
                oval,
                base_map,
                arch_id,
                date_id,
            )
            save_ch_mask_only_unet(
                row_obj,
                None,
                postprocessing,
                pmap,
                fast_mask,
                arch_id,
                date_id,
            )
            results.append(info["pmap_path"])
        except Exception as e:
            print(f"Error plotting {info.get('fits_path', '?')}: {e}")
            results.append(None)
    return results


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

    batch_size = model.architecture["apply_batch_size"]
    plot_workers = model.architecture.get("plot_threads", os.cpu_count() or 4)
    max_inflight_batches = model.architecture.get("max_inflight_batches", 2)
    rows = list(df.itertuples())
    pmap_paths = []
    inflight: deque[Future] = deque()

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=plot_workers, mp_context=ctx) as executor:
        with tqdm(total=len(rows), desc="Generating pmaps") as pbar:
            for offset in range(0, len(rows), batch_size):
                batch_rows = rows[offset : offset + batch_size]

                imgs = []
                valid_rows = []
                for row in batch_rows:
                    try:
                        imgs.append(prepare_fits(row.fits_path))
                        valid_rows.append(row)
                    except Exception as e:
                        print(f"Error loading {row.Index}: {e}")

                if not imgs:
                    pbar.update(len(batch_rows))
                    continue

                x = np.stack(imgs)[..., np.newaxis].astype(np.float32)
                try:
                    probs = model.predict(x)
                except Exception as e:
                    print(
                        f"Error batch predicting starting at {batch_rows[0].Index}: {e}"
                    )
                    pbar.update(len(batch_rows))
                    continue

                task_rows = []
                for row, prob in zip(valid_rows, probs):
                    try:
                        pmap = prob[..., 0]
                        path = pmap_path(
                            row, model.architecture_id, model.date_range_id
                        )
                        np.save(path, pmap)
                        task_rows.append(
                            {
                                "fits_path": row.fits_path,
                                "mask_path": row.mask_path,
                                "pmap_path": path,
                            }
                        )
                    except Exception as e:
                        print(f"Error preparing outputs for {row.Index}: {e}")

                if not task_rows:
                    pbar.update(len(batch_rows))
                    continue

                fut = executor.submit(
                    render_batch,
                    {
                        "rows": task_rows,
                        "postprocessing": postprocessing,
                        "arch_id": model.architecture_id,
                        "date_id": model.date_range_id,
                        "fast_mask": True,
                    },
                )
                inflight.append(fut)

                while len(inflight) >= max_inflight_batches:
                    oldest = inflight.popleft()
                    try:
                        res = oldest.result()
                        pmap_paths.extend(res)
                    except Exception as e:
                        print(f"Error in plotting batch: {e}")

                pbar.update(len(batch_rows))

            while inflight:
                fut = inflight.popleft()
                try:
                    res = fut.result()
                    pmap_paths.extend(res)
                except Exception as e:
                    print(f"Error in plotting batch: {e}")


if __name__ == "__main__":
    main()
