#!/usr/bin/env python3
import os
import sys
import time
import multiprocessing as mp
from collections import deque
from concurrent.futures import ProcessPoolExecutor, Future
from types import SimpleNamespace

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


def render_job(job):
    """Worker: generates all plots for a single pmap; no TF usage."""
    row = SimpleNamespace(fits_path=job["fits_path"], mask_path=job["mask_path"])
    try:
        base_map = sunpy.map.Map(row.fits_path)
        oval = job.get("oval")
        if oval is None:
            oval = generate_omask(row)
            
        save_ch_map_unet(
            row,
            None,
            job["postprocessing"],
            job["pmap"],
            oval=oval,
            map_obj=base_map,
            arch_id=job["arch_id"],
            date_id=job["date_id"],
        )
        save_ch_map_unet(
            row,
            None,
            "P0",
            job["pmap"],
            oval=oval,
            map_obj=base_map,
            arch_id=job["arch_id"],
            date_id=job["date_id"],
        )
        save_ch_mask_only_unet(
            row,
            None,
            job["postprocessing"],
            job["pmap"],
            True,
        )
        return job["pmap_path"]
    except Exception as e:
        return f"Error: {e}"


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

    ctx = mp.get_context("spawn")
    total_infer_time = 0.0
    total_plot_wait = 0.0

    with ProcessPoolExecutor(max_workers=plot_workers, mp_context=ctx) as executor:
        with tqdm(total=len(rows), desc="Generating pmaps") as pbar:
            for offset in range(0, len(rows), batch_size):
                batch_rows = rows[offset : offset + batch_size]

                imgs = []
                valid_rows = []
                for row in batch_rows:
                    try:
                        _, img = prepare_fits(row.fits_path)
                        imgs.append(img)
                        valid_rows.append(row)
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

                infer_start = time.time()
                try:
                    probs = model.compiled_infer(x)
                except Exception as e:
                    print(
                        f"Error batch predicting starting at {batch_rows[0].Index}: {e}"
                    )
                    pbar.update(len(batch_rows))
                    continue
                total_infer_time += time.time() - infer_start

                for row, prob in zip(valid_rows, probs):
                    pmap = prob[..., 0]
                    save_pmap(model, row, pmap)
                    job = {
                        "fits_path": row.fits_path,
                        "mask_path": row.mask_path,
                        "pmap_path": pmap_path(row, model.architecture_id, model.date_range_id),
                        "pmap": pmap,
                        "postprocessing": postprocessing,
                        "arch_id": model.architecture_id,
                        "date_id": model.date_range_id,
                    }
                    fut = executor.submit(render_job, job)
                    inflight.append(fut)
                    del pmap

                    while len(inflight) >= max_inflight_plots:
                        oldest = inflight.popleft()
                        wait_start = time.time()
                        try:
                            res = oldest.result()
                            pmap_paths.append(res)
                        except Exception as e:
                            print(f"Error in plotting task: {e}")
                        total_plot_wait += time.time() - wait_start

                pbar.update(len(batch_rows))

            while inflight:
                fut = inflight.popleft()
                wait_start = time.time()
                try:
                    res = fut.result()
                    pmap_paths.append(res)
                except Exception as e:
                    print(f"Error in plotting task: {e}")
                total_plot_wait += time.time() - wait_start

    print(f"Inference time: {total_infer_time:.2f}s, plot wait time: {total_plot_wait:.2f}s")


if __name__ == "__main__":
    main()
