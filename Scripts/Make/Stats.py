#!/usr/bin/env python3
import sys
import re
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


def build_output_path(architecture_id, date_range_id, postprocessing, date_range_str=None):
    out_dir = Path("./Outputs") / "Stats"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    filename = f"{architecture_id}{date_range_id}{postprocessing}"
    if date_range_str:
        filename += f" {date_range_str}"
    filename += " Stats.parquet"
    
    return out_dir / filename


def parse_spec(spec_str):
    """
    Parse spec string like 'A1D1P1' into (arch, date_range, postprocessing).
    Returns tuple or None if invalid.
    """
    match = re.match(r'A(\d)D(\d)P(.+)', spec_str)
    if match:
        arch, date_range, postproc = match.groups()
        return f"A{arch}", f"D{date_range}", f"P{postproc}"
    return None


def main(argv):
    if len(argv) < 3:
        print("Usage: python Scripts/Make.py Stats <start> <end> [<spec> ...] [--synoptic]")
        print("\nwhere <spec> is formatted like A1D1P1 (defaults: A1D1P1, A2D1P1, A2D2P1)")
        print("\nExamples:")
        print("  python Scripts/Make.py Stats 20100101 20101231")
        print("  python Scripts/Make.py Stats 20100101 20101231 A1D1P1")
        print("  python Scripts/Make.py Stats 20100101 20101231 A1D1P1 A2D2P1 --synoptic")
        return 1

    start_date = argv[1]
    end_date = argv[2]

    # Parse specs and flags from remaining arguments
    specs = []
    use_synoptic = False

    for arg in argv[3:]:
        if arg == "--synoptic":
            use_synoptic = True
        else:
            parsed = parse_spec(arg)
            if parsed:
                specs.append(parsed)
            else:
                print(f"Warning: invalid spec '{arg}', skipping")

    # Use defaults if no specs provided
    if not specs:
        specs = [
            ("A1", "D1", "P1"),
            ("A2", "D1", "P1"),
            ("A2", "D2", "P1"),
        ]
        print(f"Using default specs: {' '.join(f'{a}{d}{p}' for a, d, p in specs)}")

    paths_name = "Paths (Synoptic).parquet" if use_synoptic else "Paths.parquet"
    paths_parquet = Path(paths["artifact_root"]) / paths_name
    if not paths_parquet.exists():
        print(f"Missing {paths_parquet}")
        return 1

    df = pd.read_parquet(paths_parquet)
    
    # String-based date filtering (index is typically YYYYMMDD strings)
    df = df[(df.index >= start_date) & (df.index <= end_date)]
    if df.empty:
        print(f"No data found in date range {start_date} to {end_date}")
        return 1

    print(f"Filtered to {len(df)} rows")

    print(f"Processing {len(df)} rows across {len(specs)} models")

    workers = max(1, int(apply_config["plot_threads"]))

    # Process each model/postprocessing combo
    for architecture_id, date_range_id, postprocessing in specs:
        smoothing_params = get_postprocessing_params(postprocessing)
        stats_df = compute_stats_df(
            df, architecture_id, date_range_id, smoothing_params, workers
        )
        stats_df.index.name = "key"

        # Build date range string for filename
        date_range_str = f"{start_date} to {end_date}"

        out_path = build_output_path(
            architecture_id, date_range_id, postprocessing, date_range_str
        )
        stats_df.to_parquet(out_path)
        print(f"Saved {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
