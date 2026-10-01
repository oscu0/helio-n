#!/usr/bin/env python3
"""Reconstitute archived solar-wind series and write the two analysis workbooks."""

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from Library.SW.Archive import load_series, resolve_cr_or_date_range
from Library.SW.Config import (
    ALL_VALIDATION_SATELLITES,
    get_satellite_config,
    parse_satellite_ids,
)
from Library.SW.Inputs import add_sdo_observation_geometry
from Library.SW.Report import (
    build_satellite_data_frames,
    write_per_cr_stats_workbook,
    write_satellite_data_workbook,
)
from Library.SW.Stats import (
    build_sw_forecast_cr_stats_csv_frame,
    restore_observed_and_recurrent_series,
    restore_swx_series,
)


def stamp_for_range(start_dt, end_dt):
    return f"{start_dt:%Y%m%d}-{end_dt:%Y%m%d}"


def _available_satellites(series):
    available = list(
        dict.fromkeys(series["satellite"].columns.get_level_values(0))
    )
    supported_order = (*ALL_VALIDATION_SATELLITES, "ace_earth")
    unsupported = set(available).difference(supported_order)
    assert not unsupported, f"Unsupported satellite IDs in archive: {sorted(unsupported)}"
    return [sat for sat in supported_order if sat in available]


def _build_comparison_frames(series, satellite_ids):
    frames = {}
    for sat_id in satellite_ids:
        frame = series["satellite", sat_id].copy()
        geometry_columns = {"hee_beta_deg", "sdo_observation_age_days"}
        position_columns = {"x_hee_au", "y_hee_au", "z_hee_au"}
        if not geometry_columns.issubset(frame.columns) and position_columns.issubset(
            frame.columns
        ):
            frame = add_sdo_observation_geometry(frame)
        if "v_noaa" in frame.columns:
            frame.drop(columns="v_noaa", inplace=True)
        frames[sat_id] = frame
    return frames


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "start_or_cr", help="CR number, or inclusive UTC start timestamp when end is given"
    )
    parser.add_argument("end", nargs="?", help="Exclusive UTC end timestamp for a date range")
    parser.add_argument(
        "--archive-root",
        type=Path,
        default=ROOT_DIR / "Outputs" / "SW" / "Archive",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT_DIR / "Outputs" / "SW"
    )
    parser.add_argument(
        "--satellites",
        help="Comma-separated satellite IDs or 'all'; default is every supported ID in the archives",
    )
    parser.add_argument("--swx-parquet", type=Path)
    parser.add_argument(
        "--satellite-data-out",
        type=Path,
        help="Optional path for the hourly multi-sheet satellite data workbook.",
    )
    parser.add_argument(
        "--per-cr-stats-out",
        type=Path,
        help="Optional path for the per-CR, per-satellite statistics workbook.",
    )
    args = parser.parse_args(argv)

    _cr, start_dt, end_dt = resolve_cr_or_date_range(args.start_or_cr, args.end)
    series = load_series(args.archive_root, start_dt, end_dt)
    assert "satellite" in series.columns.get_level_values(0), (
        f"No satellite comparisons in [{start_dt}, {end_dt})"
    )
    present_satellites = _available_satellites(series)
    if args.satellites is None or args.satellites == "all":
        satellite_ids = present_satellites
    else:
        satellite_ids = parse_satellite_ids(args.satellites)
    assert satellite_ids, "Statistics require at least one satellite"
    missing = set(satellite_ids).difference(present_satellites)
    assert not missing, f"Satellites absent from archived series: {sorted(missing)}"
    assert "ace" in satellite_ids, (
        "The hourly workbook requires native ACE ('ace') for relative coordinates"
    )

    comparison_frames = _build_comparison_frames(series, satellite_ids)
    comparison_frames = restore_observed_and_recurrent_series(
        comparison_frames=comparison_frames,
        start_dt=start_dt,
        end_dt=end_dt,
    )
    # The per-CR archive is canonical and already carries the native SWX
    # forecast samples. Match the former exporter: replace them only when the
    # caller explicitly supplies a separate SWX source.
    if args.swx_parquet is not None:
        comparison_frames = restore_swx_series(
            comparison_frames,
            swx_path=args.swx_parquet,
        )

    satellite_frames = build_satellite_data_frames(
        reproduction_frame=series,
        comparison_frames=comparison_frames,
        start_dt=start_dt,
        end_dt=end_dt,
    )
    satellite_labels = {
        sat_id: get_satellite_config(sat_id).label for sat_id in satellite_ids
    }
    per_cr_stats = build_sw_forecast_cr_stats_csv_frame(
        comparison_frames=comparison_frames,
        start_dt=start_dt,
        end_dt=end_dt,
        sat_labels=satellite_labels,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = stamp_for_range(start_dt, end_dt)
    satellite_data_path = args.satellite_data_out or (
        args.output_dir / f"SW {stamp}.xlsx"
    )
    per_cr_stats_path = args.per_cr_stats_out or (
        args.output_dir / f"SW {stamp} (Per-CR).xlsx"
    )
    write_satellite_data_workbook(satellite_frames, satellite_data_path)
    write_per_cr_stats_workbook(per_cr_stats, per_cr_stats_path)
    print(f"Saved {satellite_data_path} | {sum(map(len, satellite_frames.values()))} hourly rows")
    print(f"Saved {per_cr_stats_path} | {len(per_cr_stats)} per-CR statistic rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
