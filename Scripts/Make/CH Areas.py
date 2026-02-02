#!/usr/bin/env python3
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

from Library.CH import ch_rel_area
from Library.Config import paths
from Library.IO import pmap_path, prepare_pmap


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

    rows = list(df.itertuples())
    results = []

    for row in tqdm(rows, desc="CH areas"):
        record = {
            "key": row.Index,
            "fits_path": row.fits_path,
            "mask_path": row.mask_path,
        }

        record["s_idl"] = ch_rel_area(row, reference_mode=True)

        for arch, date_id in specs:
            spec = f"{arch}{date_id}"
            pmap_file = pmap_path(row, arch, date_id)
            if not Path(pmap_file).exists():
                record[f"s_{spec.lower()}"] = float("nan")
                continue
            pmap = prepare_pmap(pmap_file)
            record[f"s_{spec.lower()}"] = ch_rel_area(
                row, pmap=pmap, reference_mode=False
            )

        results.append(record)

    out_df = pd.DataFrame(results).set_index("key")

    out_dir = Path("./Outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"CH Areas {start}-{end}.parquet"
    out_df.to_parquet(out_path)
    print(f"Saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
