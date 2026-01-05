#!/usr/bin/env python3
import os

# Python warnings
os.environ["PYTHONWARNINGS"] = "ignore::FutureWarning,ignore::UserWarning"
# TensorFlow / XLA C++ logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # hide INFO/WARNING/ERROR
os.environ["ABSL_MIN_LOG_LEVEL"] = "3"     # silence absl logging

os.environ["TF_GPU_ALLOCATOR"] = "cuda_malloc_async"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

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
    """Worker: generates all plots for a single pmap on disk; no TF usage."""
    row = SimpleNamespace(fits_path=job["fits_path"], mask_path=job["mask_path"])
    try:
        pmap = np.load(job["pmap_path"])
        base_map = sunpy.map.Map(row.fits_path)
        oval = job.get("oval") or generate_omask(row)

        save_ch_map_unet(
            row,
            None,
            job["postprocessing"],
            pmap,
            oval,
            base_map,
            arch_id=job["arch_id"],
            date_id=job["date_id"],
        )
        save_ch_map_unet(
            row,
            None,
            "P0",
            pmap,
            oval,
            base_map,
            arch_id=job["arch_id"],
            date_id=job["date_id"],
        )
        save_ch_mask_only_unet(
            row,
            None,
            job["postprocessing"],
            pmap,
            True,
            arch_id=job["arch_id"],
            date_id=job["date_id"],
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
    chunk_size = apply_config["chunk_size"]
    plot_workers = apply_config["plot_threads"]

    # Just in case we try masssive chunks
    max_inflight_plots = max(1, apply_config["max_inflight_plots"] // chunk_size)
    target_size = model.architecture["img_size"]


    rows = list(df.itertuples())
    pmap_paths = []
    inflight: deque[Future] = deque()

    ctx = mp.get_context("spawn")
    total_infer_time = 0.0
    total_plot_wait = 0.0
    total_drain_wait = 0.0

    alpha = 0.1

    def update_ema(current, value):
        return value if current == 0.0 else alpha * value + (1 - alpha) * current

    ema = {k: 0.0 for k in ["load", "stack", "resize", "infer", "submit", "wait"]}
    totals = {k: 0.0 for k in ["load", "stack", "resize", "infer", "submit", "wait"]}

    with ProcessPoolExecutor(max_workers=plot_workers, mp_context=ctx) as executor:
        with tqdm(total=len(rows), desc="Generating pmaps") as pbar:
            for offset in range(0, len(rows), batch_size):
                batch_rows = rows[offset : offset + batch_size]

                batch_start = time.perf_counter()
                load_start = batch_start

                imgs = []
                valid_rows = []
                for row in batch_rows:
                    try:
                        _, img = prepare_fits(row.fits_path)
                        imgs.append(img)
                        valid_rows.append(row)
                    except Exception as e:
                        print(f"Error loading {row.Index}: {e}")

                load_time = time.perf_counter() - load_start

                if not imgs:
                    pbar.update(len(batch_rows))
                    continue

                # Accumulate per-batch timings across chunks
                stack_time = 0.0
                resize_time = 0.0
                infer_time = 0.0
                submit_time = 0.0
                wait_time = 0.0

                for start in range(0, len(valid_rows), chunk_size):
                    chunk_rows = valid_rows[start : start + chunk_size]
                    chunk_imgs = imgs[start : start + chunk_size]

                    stack_start = time.perf_counter()
                    x = np.stack(chunk_imgs)[..., np.newaxis].astype(np.float32)
                    stack_time += time.perf_counter() - stack_start

                    resize_start = time.perf_counter()
                    if x.shape[1] != target_size:
                        x = tf.image.resize(
                            x, [target_size, target_size], method="bilinear"
                        ).numpy()
                    resize_time += time.perf_counter() - resize_start

                    if x.shape[0] < chunk_size:
                        pad = chunk_size - x.shape[0]
                        x = np.pad(x, ((0, pad), (0, 0), (0, 0), (0, 0)), mode="constant")
                        chunk_rows = chunk_rows + [None] * pad
                        mask_keep = [1] * (chunk_size - pad) + [0] * pad
                    else:
                        mask_keep = [1] * chunk_size

                    infer_start = time.perf_counter()
                    try:
                        probs = model.compiled_infer(x)
                    except Exception as e:
                        print(
                            f"Error batch predicting starting at {batch_rows[0].Index}: {e}"
                        )
                        pbar.update(len(batch_rows))
                        continue
                    chunk_infer = time.perf_counter() - infer_start
                    infer_time += chunk_infer
                    total_infer_time += chunk_infer

                    submit_start = time.perf_counter()
                    for row, prob, keep in zip(chunk_rows, probs, mask_keep):
                        if keep == 0 or row is None:
                            continue
                        pmap = prob[..., 0]
                        pmap_file, _ = save_pmap(model, row, pmap)
                        job = {
                            "fits_path": row.fits_path,
                            "mask_path": row.mask_path,
                            "pmap_path": pmap_file,
                            "postprocessing": postprocessing,
                            "arch_id": model.architecture_id,
                            "date_id": model.date_range_id,
                        }
                        fut = executor.submit(render_job, job)
                        inflight.append(fut)
                        del pmap

                        while len(inflight) >= max_inflight_plots:
                            oldest = inflight.popleft()
                            wait_start = time.perf_counter()
                            try:
                                res = oldest.result()
                                pmap_paths.append(res)
                            except Exception as e:
                                print(f"Error in plotting task: {e}")
                            dt = time.perf_counter() - wait_start
                            wait_time += dt
                            total_plot_wait += dt

                    submit_time += time.perf_counter() - submit_start

                batch_wall = time.perf_counter() - batch_start
                imgs_n = len(valid_rows)
                it_s = imgs_n / batch_wall if batch_wall > 0 else 0.0

                ema["load"] = update_ema(ema["load"], load_time)
                ema["stack"] = update_ema(ema["stack"], stack_time)
                ema["resize"] = update_ema(ema["resize"], resize_time)
                ema["infer"] = update_ema(ema["infer"], infer_time)
                ema["submit"] = update_ema(ema["submit"], submit_time)
                ema["wait"] = update_ema(ema["wait"], wait_time)
                totals["load"] += load_time
                totals["stack"] += stack_time
                totals["resize"] += resize_time
                totals["infer"] += infer_time
                totals["submit"] += submit_time
                totals["wait"] += wait_time

                # print(
                #     f"imgs={imgs_n} "
                #     f"load={load_time:.3f}/{ema['load']:.3f} "
                #     f"stack={stack_time:.3f}/{ema['stack']:.3f} "
                #     f"resize={resize_time:.3f}/{ema['resize']:.3f} "
                #     f"infer={infer_time:.3f}/{ema['infer']:.3f} "
                #     f"submit={submit_time:.3f}/{ema['submit']:.3f} "
                #     f"wait={wait_time:.3f}/{ema['wait']:.3f} "
                #     f"drain=0.000 "
                #     f"batch_wall={batch_wall:.3f} "
                #     f"it/s={it_s:.2f}"
                # )
                pbar.update(len(batch_rows))

            while inflight:
                fut = inflight.popleft()
                wait_start = time.perf_counter()
                try:
                    res = fut.result()
                    pmap_paths.append(res)
                except Exception as e:
                    print(f"Error in plotting task: {e}")
                total_drain_wait += time.perf_counter() - wait_start

    print(
        "Totals: "
        f"load={totals['load']:.2f}s stack={totals['stack']:.2f}s resize={totals['resize']:.2f}s "
        f"infer={totals['infer']:.2f}s submit={totals['submit']:.2f}s "
        f"wait={totals['wait']:.2f}s drain={total_drain_wait:.2f}s"
    )
    print(
        f"Total measured sum: "
        f"{(totals['load'] + totals['stack'] + totals['resize'] + totals['infer'] + totals['submit'] + totals['wait'] + total_drain_wait):.2f}s"
    )
    print(
        "EMA: "
        f"load={ema['load']:.3f}s stack={ema['stack']:.3f}s resize={ema['resize']:.3f}s "
        f"infer={ema['infer']:.3f}s submit={ema['submit']:.3f}s wait={ema['wait']:.3f}s"
    )


if __name__ == "__main__":
    main()
