#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Library.SW.Report import build_hourly_report_frame, write_report  # noqa: E402
from Library.SW.Stats import restore_observed_and_recurrent_series  # noqa: E402


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description=(
            "Build Julia's single-sheet hourly Excel report from an exact "
            "SW reproduction parquet and the native spacecraft observations."
        )
    )
    parser.add_argument(
        "start", help="Inclusive run start datetime accepted by pandas.Timestamp"
    )
    parser.add_argument(
        "end", help="Exclusive run end datetime accepted by pandas.Timestamp"
    )
    parser.add_argument(
        "--output-dir",
        default="Outputs/SW",
        help="Directory containing SW outputs and receiving the report.",
    )
    parser.add_argument(
        "--freq",
        default="1h",
        choices=["1h"],
        help="Report frequency. This paper exporter is hourly.",
    )
    parser.add_argument(
        "--report-out",
        default=None,
        help="Optional explicit Excel output path.",
    )
    return parser.parse_args(argv[1:])


def stamp_for_range(start_dt, end_dt):
    return f"{start_dt:%Y%m%d_%H%M}-{end_dt:%Y%m%d_%H%M}"


def main(argv):
    args = parse_args(argv)
    start_dt = pd.Timestamp(args.start)
    end_dt = pd.Timestamp(args.end)
    assert start_dt < end_dt, (
        f"Expected start datetime before end datetime; got start={start_dt} "
        f"and end={end_dt}"
    )

    output_dir = Path(args.output_dir)
    stamp = stamp_for_range(start_dt, end_dt)
    reproduction_path = output_dir / f"SW Reproduction Series {stamp}.parquet"
    assert reproduction_path.exists(), (
        f"Missing exact matching reproduction parquet: {reproduction_path}"
    )

    report_out = (
        Path(args.report_out)
        if args.report_out is not None
        else output_dir / f"SW Report {stamp}.xlsx"
    )

    reproduction_frame = pd.read_parquet(reproduction_path)
    reproduction_frame.index = pd.to_datetime(reproduction_frame.index)
    reproduction_frame = reproduction_frame.sort_index().sort_index(axis=1)
    comparison_frames = {
        sat_name: reproduction_frame["satellite", sat_name].copy()
        for sat_name in ["ace_earth", "stereo_a"]
    }
    for frame in comparison_frames.values():
        if "v_noaa" in frame.columns:
            frame.drop(columns="v_noaa", inplace=True)
    comparison_frames = restore_observed_and_recurrent_series(
        comparison_frames=comparison_frames,
        start_dt=start_dt,
        end_dt=end_dt,
        freq=args.freq,
    )

    report_frame = build_hourly_report_frame(
        reproduction_frame=reproduction_frame,
        comparison_frames=comparison_frames,
        start_dt=start_dt,
        end_dt=end_dt,
        freq=args.freq,
    )
    write_report(report_frame, report_out)
    print("Saved SW report:", report_out)
    print(
        "Source:",
        reproduction_path,
        "| rows:",
        len(report_frame),
        "| freq:",
        args.freq,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
