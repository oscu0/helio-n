#!/usr/bin/env python3
"""Render an arbitrary archived SW time range without propagating it again."""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
os.environ.setdefault("MPLCONFIGDIR", "/tmp/helio_n_matplotlib")
sys.path.append(str(ROOT_DIR))

from Library.SW.Archive import load_cube, load_series, resolve_cr_or_date_range
from Library.SW.Config import (
    DEFAULT_ENABLED_SATELLITES,
    get_satellite_config,
    load_empirical_spec,
    load_sw_runtime_spec,
    parse_satellite_ids,
)
from Library.SW.Constants import CARRINGTON_ROTATION_DAYS
from Library.SW.Visualization import export_polar_animation


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "start_or_cr", help="CR number, or inclusive UTC start timestamp when end is given"
    )
    parser.add_argument("end", nargs="?", help="Exclusive UTC end timestamp for a date range")
    parser.add_argument(
        "--pad-days", type=float,
        help="Pad each animation end (default: 3 days for a CR, 0 for a date range)",
    )
    parser.add_argument("--archive-root", type=Path, default=ROOT_DIR / "Outputs" / "SW" / "Archive")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--dpi", type=int)
    parser.add_argument(
        "--satellites",
        help="Comma-separated satellite IDs to plot, or 'none' for polar-only (default: all)",
    )
    args = parser.parse_args(argv)
    assert args.fps > 0
    cr, start, end = resolve_cr_or_date_range(args.start_or_cr, args.end)
    pad_days = (
        (3.0 if cr is not None else 0.0)
        if args.pad_days is None else args.pad_days
    )
    assert start < end and pad_days >= 0
    pad = pd.Timedelta(days=pad_days)
    start -= pad
    end += pad

    cube = load_cube(args.archive_root, start, end)
    panel_history_days = float(CARRINGTON_ROTATION_DAYS - 7.0)
    panel_future_days = 7.0
    series = load_series(
        args.archive_root,
        start - pd.Timedelta(days=panel_history_days),
        end + pd.Timedelta(days=panel_future_days),
    )
    times = pd.DatetimeIndex(cube.time.values)
    speed = cube["speed"].values
    finite = speed[np.isfinite(speed)]
    assert len(finite) > 0, "No finite speeds in the requested animation range"
    comparison_frames = {}
    if "satellite" in series.columns.get_level_values(0):
        satellite = series["satellite"]
        for name in satellite.columns.get_level_values(0).unique():
            frame = satellite[name].copy()
            frame.attrs["sat"] = name
            frame.attrs["label"] = get_satellite_config(name).label
            comparison_frames[name] = frame
    selected_satellites = parse_satellite_ids(args.satellites)
    if args.satellites is None and not set(selected_satellites).issubset(comparison_frames):
        selected_satellites = list(comparison_frames)
    stamp = f"{start:%Y%m%d}-{end:%Y%m%d}"
    output = args.output or Path(args.archive_root) / f"SW {stamp}.mp4"
    runtime = load_sw_runtime_spec()
    result = export_polar_animation(
        anim_outfile=output,
        time_axis=times,
        phi_axis=cube.phi.values,
        r_axis=cube.r.values,
        grid_raw=speed,
        post_vlims_raw=(float(finite.min()), float(finite.max())),
        slow_sw_pred_mask=cube["is_slow_wind"].values,
        slow_sw_speed=load_empirical_spec().slow_sw_speed(times),
        comparison_frames=comparison_frames,
        satellites=selected_satellites,
        anim_fps=args.fps,
        anim_dpi=args.dpi or runtime["animation_dpi"],
    )
    print(f"Saved {output} | {int(result['frames'])} frames")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
