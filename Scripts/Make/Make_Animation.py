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

from Library.SW.Archive import cr_bounds, load_cube, load_series
from Library.SW.Config import load_empirical_spec, load_sw_runtime_spec
from Library.SW.Visualization import export_polar_animation


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("start_or_cr", help="CR number or inclusive UTC start timestamp")
    parser.add_argument("end", nargs="?", help="Exclusive UTC timestamp")
    parser.add_argument("--pad-days", type=float, help="Pad each end (default: 7 days in CR mode)")
    parser.add_argument("--archive-root", type=Path, default=ROOT_DIR / "Outputs" / "SW" / "Archive")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--dpi", type=int)
    args = parser.parse_args(argv)
    assert args.fps > 0
    if args.end is None:
        cr = int(args.start_or_cr)
        start, end = cr_bounds(cr)
        pad_days = 7.0 if args.pad_days is None else args.pad_days
    else:
        cr = None
        start, end = pd.Timestamp(args.start_or_cr), pd.Timestamp(args.end)
        pad_days = 0.0 if args.pad_days is None else args.pad_days
    assert start < end and pad_days >= 0
    pad = pd.Timedelta(days=pad_days)
    start -= pad
    end += pad

    cube = load_cube(args.archive_root, start, end)
    series = load_series(args.archive_root, start, end)
    times = pd.DatetimeIndex(cube.time.values)
    speed = cube["speed"].values
    finite = speed[np.isfinite(speed)]
    assert len(finite) > 0, "No finite speeds in the requested animation range"
    comparison_frames = {}
    satellite = series["satellite"]
    for name in satellite.columns.get_level_values(0).unique():
        frame = satellite[name].copy()
        frame.attrs["sat"] = name
        frame.attrs["label"] = {"ace_earth": "ACE @ Earth", "stereo_a": "STEREO-A"}.get(name, name)
        comparison_frames[name] = frame
    stamp = f"{start:%Y%m%d_%H%M}-{end:%Y%m%d_%H%M}"
    label = f"CR{cr} " if cr is not None else ""
    output = args.output or Path(args.archive_root) / f"SW Animation {label}{stamp}.mp4"
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
        anim_fps=args.fps,
        anim_dpi=args.dpi or runtime["animation_dpi"],
    )
    print(f"Saved {output} | {int(result['frames'])} frames")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
