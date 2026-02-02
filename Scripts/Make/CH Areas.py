#!/usr/bin/env python3
import sys
from pathlib import Path

import multiprocessing as mp
from functools import partial
from types import SimpleNamespace
import pandas as pd
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

import numpy as np
import sunpy.map
from sunpy.map.maputils import all_coordinates_from_map, coordinate_is_on_solar_disk

from Library.CH import project
from Library.Config import apply_config, paths
from Library.IO import pmap_path, prepare_pmap
from Library.Metrics import generate_omask
from Library.Processing import get_postprocessing_params, pmap_to_mask
from Library.IO import prepare_mask


def chunk_rows(rows, chunks):
    if chunks <= 1:
        return [rows]
    size = max(1, len(rows) // chunks)
    return [rows[i : i + size] for i in range(0, len(rows), size)]


def worker_compute(rows, specs, smoothing_params):
    results = []
    geom_cache = {}

    def get_geom(row):
        key = row[1]
        cached = geom_cache.get(key)
        if cached is not None:
            return cached

        row_ns = SimpleNamespace(fits_path=row[1], mask_path=row[2])
        m = sunpy.map.Map(row[1])
        omask = generate_omask(row_ns)

        # precompute 1/mu by projecting a unit mask once
        inv_mu = project(m, np.ones_like(omask, dtype=float))

        # precompute sun area once per fits
        hpc_coords = all_coordinates_from_map(m)
        disk_mask = coordinate_is_on_solar_disk(hpc_coords)
        sun = project(m, disk_mask).sum()

        geom_cache[key] = (omask, inv_mu, sun)
        return geom_cache[key]

    def rel_area_from_mask(row, mask_2d):
        omask, inv_mu, sun = get_geom(row)
        ch_mask = mask_2d * omask
        proj = np.zeros_like(ch_mask, dtype=float)
        good = (ch_mask != 0) & (inv_mu > 1e-3)
        proj[good] = ch_mask[good] * inv_mu[good]
        ch_area = np.nan_to_num(proj, 0).sum()
        return ch_area / sun

    for row in rows:
        record = {
            "key": row[0],
            "fits_path": row[1],
            "mask_path": row[2],
        }

        idl_mask = prepare_mask(row[2])
        record["s_idl"] = rel_area_from_mask(row, idl_mask)

        for arch, date_id in specs:
            spec = f"{arch}{date_id}"
            row_ns = SimpleNamespace(fits_path=row[1], mask_path=row[2])
            pmap_file = pmap_path(row_ns, arch, date_id)
            if not Path(pmap_file).exists():
                record[f"s_{spec.lower()}"] = float("nan")
                continue
            pmap = prepare_pmap(pmap_file)
            model_mask = pmap_to_mask(pmap, smoothing_params)
            record[f"s_{spec.lower()}"] = rel_area_from_mask(row, model_mask)

        results.append(record)

    return results


def main(argv):
    if len(argv) < 3:
        print("Usage: python -m Scripts.Make CH\\ Areas <start> <end>")
        print("Example: python -m Scripts.Make CH\\ Areas 20100101 20101231")
        return 1

    start = argv[1]
    end = argv[2]

    paths_parquet = Path(paths["artifact_root"]) / "Paths.parquet"
    if not paths_parquet.exists():
        print(f"Missing {paths_parquet}")
        return 1

    df = pd.read_parquet(paths_parquet)
    df = df[start:end]
    if df.empty:
        print("No rows in requested date range.")
        return 1

    specs = [("A1", "D1"), ("A2", "D1"), ("A2", "D2")]
    smoothing_params = get_postprocessing_params("P0")

    rows = list(df[["fits_path", "mask_path"]].itertuples(index=True, name=None))
    max_workers = max(1, int(apply_config.get("plot_threads", 1)))
    chunks = chunk_rows(rows, max_workers)

    results = []
    ctx = mp.get_context("spawn")
    worker_fn = partial(worker_compute, specs=specs, smoothing_params=smoothing_params)
    with ctx.Pool(processes=min(max_workers, len(chunks))) as pool:
        iterator = pool.imap_unordered(worker_fn, chunks)
        for chunk_result in tqdm(iterator, total=len(chunks), desc="CH areas"):
            results.extend(chunk_result)

    out_df = pd.DataFrame(results).set_index("key")

    out_dir = Path("./Outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"CH Areas {start}-{end}.parquet"
    out_df.to_parquet(out_path)
    print(f"Saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
