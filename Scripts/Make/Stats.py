#!/usr/bin/env python3
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from types import SimpleNamespace
import multiprocessing as mp

import numpy as np
import pandas as pd
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

from Library.Config import apply_config, paths
from Library.IO import pmap_path, prepare_mask
from Library.Metrics import (
    abs_area,
    compute_fourier_descriptor,
    compute_zernike_descriptor,
    dice,
    generate_omask,
    iou,
    shape_distance,
)
from Library.Processing import get_postprocessing_params, pmap_to_mask


def stats_from_masks(m1, m2, oval=None):
    stats = {}

    stats["fourier_distance"] = shape_distance(
        compute_fourier_descriptor(m1, omask=oval),
        compute_fourier_descriptor(m2, omask=oval),
    )
    stats["zernike_distance"] = shape_distance(
        compute_zernike_descriptor(m1, omask=oval),
        compute_zernike_descriptor(m2, omask=oval),
    )

    c1 = abs_area(m1, oval)
    c2 = abs_area(m2, oval)

    if c1 + c2 == 0:
        stats["rel_area"] = 0.0
    elif c1 * c2 == 0:
        stats["rel_area"] = 1.0
    else:
        stats["rel_area"] = 1 - (min(c1, c2) / max(c1, c2))

    stats["iou"] = iou(m1, m2, oval)
    stats["dice"] = dice(m1, m2, oval)

    return stats


def _empty_stats():
    return {
        "fourier_distance": np.nan,
        "zernike_distance": np.nan,
        "rel_area": np.nan,
        "iou": np.nan,
        "dice": np.nan,
    }


def load_pmap(mask_path, architecture_id, date_range_id):
    stub = SimpleNamespace(mask_path=mask_path)
    path = pmap_path(stub, architecture_id, date_range_id)
    if not Path(path).exists():
        raise FileNotFoundError(f"Missing pmap: {path}")
    return np.load(path)


def worker_stats(
    position, fits_path, mask_path, architecture_id, date_range_id, smoothing_params
):
    try:
        pmap = load_pmap(mask_path, architecture_id, date_range_id)
        m1 = prepare_mask(mask_path)
        m2 = pmap_to_mask(pmap, smoothing_params)

        row = SimpleNamespace(fits_path=fits_path, mask_path=mask_path)
        oval = generate_omask(row)

        return (
            position,
            stats_from_masks(m1, m2, oval=None),
            stats_from_masks(m1, m2, oval=oval),
            False,
        )
    except Exception as e:
        print(f"Warning: stats skipped for {mask_path}: {e}")
        empty = _empty_stats()
        return position, empty, empty, True


def compute_stats_df(df, architecture_id, date_range_id, smoothing_params, workers):
    stats_results = [None] * len(df)
    stats_oval_results = [None] * len(df)
    skipped = 0

    rows = list(df.itertuples())
    if workers <= 1:
        iterator = enumerate(rows)
        for pos, row in tqdm(iterator, total=len(rows), desc="stats"):
            _, stats_full, stats_oval, was_skipped = worker_stats(
                pos,
                row.fits_path,
                row.mask_path,
                architecture_id,
                date_range_id,
                smoothing_params,
            )
            stats_results[pos] = stats_full
            stats_oval_results[pos] = stats_oval
            if was_skipped:
                skipped += 1
    else:
        ctx = mp.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as executor:
            futures = []
            for pos, row in enumerate(rows):
                futures.append(
                    executor.submit(
                        worker_stats,
                        pos,
                        row.fits_path,
                        row.mask_path,
                        architecture_id,
                        date_range_id,
                        smoothing_params,
                    )
                )
            for future in tqdm(as_completed(futures), total=len(futures), desc="stats"):
                pos, stats_full, stats_oval, was_skipped = future.result()
                stats_results[pos] = stats_full
                stats_oval_results[pos] = stats_oval
                if was_skipped:
                    skipped += 1

    stats_df = pd.DataFrame(stats_results, index=df.index)
    stats_oval_df = pd.DataFrame(stats_oval_results, index=df.index).add_suffix("_oval")
    if skipped:
        print(f"Warning: skipped {skipped} rows due to missing/corrupt pmaps.")
    return pd.concat([stats_df, stats_oval_df], axis=1)


def build_output_path(architecture_id, date_range_id, postprocessing):
    out_dir = Path("./Outputs") / "Stats"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{architecture_id}{date_range_id}{postprocessing} Stats.parquet"


def main(argv):
    if len(argv) not in (4, 5):
        print(
            "Usage: python Scripts/Make.py Stats <architecture_id> <date_range_id> <postprocessing> [synoptic]"
        )
        print("Example: python Scripts/Make.py Stats A1 D1 P1")
        return 1

    architecture_id, date_range_id, postprocessing = argv[1:4]
    use_synoptic = len(argv) == 5
    if use_synoptic and argv[4] != "synoptic":
        print("Optional 4th argument must be 'synoptic' if provided.")
        return 1

    smoothing_params = get_postprocessing_params(postprocessing)

    paths_name = "Paths (Synoptic).parquet" if use_synoptic else "Paths.parquet"
    paths_parquet = Path(paths["artifact_root"]) / paths_name
    if not paths_parquet.exists():
        print(f"Missing {paths_parquet}")
        return 1

    df = pd.read_parquet(paths_parquet)
    if df.empty:
        print("Paths.parquet is empty.")
        return 1

    workers = max(1, int(apply_config["plot_threads"]))
    stats_df = compute_stats_df(
        df, architecture_id, date_range_id, smoothing_params, workers
    )
    stats_df.index.name = "key"

    out_path = build_output_path(architecture_id, date_range_id, postprocessing)
    stats_df.to_parquet(out_path)
    print(f"Saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
