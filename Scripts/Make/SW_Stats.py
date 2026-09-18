#!/usr/bin/env python3
"""Generate solar-wind forecast statistics from archived series products."""

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

from Library.SW.Archive import load_series
from Library.SW.Config import get_satellite_config, parse_satellite_ids
from Library.SW.Stats import (
    export_sw_forecast_cr_stats_csv,
    export_sw_forecast_stats_csv,
    restore_observed_and_recurrent_series,
    restore_swx_series,
)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("start", help="Inclusive UTC start timestamp")
    parser.add_argument("end", help="Exclusive UTC end timestamp")
    parser.add_argument(
        "--archive-root",
        type=Path,
        default=ROOT_DIR / "Outputs" / "SW" / "Archive",
    )
    parser.add_argument("--output-dir", type=Path, default=ROOT_DIR / "Outputs" / "SW")
    parser.add_argument(
        "--satellites",
        help="Comma-separated satellite IDs; default is the configured enabled set",
    )
    parser.add_argument("--swx-parquet", type=Path)
    args = parser.parse_args(argv)

    start_dt = pd.Timestamp(args.start)
    end_dt = pd.Timestamp(args.end)
    assert start_dt < end_dt
    satellite_ids = parse_satellite_ids(args.satellites)
    assert satellite_ids, "Statistics require at least one enabled satellite"

    series = load_series(args.archive_root, start_dt, end_dt)
    assert "satellite" in series.columns.get_level_values(0)
    available_satellites = set(series["satellite"].columns.get_level_values(0))
    missing = set(satellite_ids) - available_satellites
    assert not missing, f"Satellites absent from archived series: {sorted(missing)}"

    comparison_frames = {
        sat_id: series["satellite", sat_id].copy()
        for sat_id in satellite_ids
    }
    for frame in comparison_frames.values():
        if "v_noaa" in frame.columns:
            frame.drop(columns="v_noaa", inplace=True)

    comparison_frames = restore_observed_and_recurrent_series(
        comparison_frames=comparison_frames,
        start_dt=start_dt,
        end_dt=end_dt,
    )
    comparison_frames = restore_swx_series(
        comparison_frames,
        swx_path=args.swx_parquet,
    )
    sat_labels = {
        sat_id: get_satellite_config(sat_id).label
        for sat_id in satellite_ids
    }
    stamp = f"{start_dt:%Y%m%d_%H%M}-{end_dt:%Y%m%d_%H%M}"
    whole_path = args.output_dir / f"SW Stats {stamp}.csv"
    cr_path = args.output_dir / f"SW Per-CR Stats {stamp}.csv"
    whole = export_sw_forecast_stats_csv(
        csv_outfile=whole_path,
        comparison_frames=comparison_frames,
        start_dt=start_dt,
        end_dt=end_dt,
        sat_labels=sat_labels,
    )
    per_cr = export_sw_forecast_cr_stats_csv(
        csv_outfile=cr_path,
        comparison_frames=comparison_frames,
        start_dt=start_dt,
        end_dt=end_dt,
        sat_labels=sat_labels,
    )
    print(f"Saved {whole_path} | {len(whole)} rows")
    print(f"Saved {cr_path} | {len(per_cr)} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
