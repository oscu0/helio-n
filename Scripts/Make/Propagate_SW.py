#!/usr/bin/env python3
"""Propagate one rounded Carrington rotation into the SW archive."""

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

from Library.Paths import data_path, resolve_repo_path
from Library.SW.Archive import cr_bounds, iter_crs, write_cr
from Library.SW.Ballistic import cube_stats, propagate_ballistic
from Library.SW.Config import (
    DEFAULT_ENABLED_SATELLITES,
    get_satellite_config,
    load_ballistic_spec,
    load_empirical_spec,
    load_slow_sw_patch_spec,
    parse_satellite_ids,
)
from Library.SW.Coords import compute_rotation_state
from Library.SW.Inputs import (
    build_ace_earth_swx_frame,
    build_model_input_series,
    load_enlil_prediction_frames,
    load_sw_input_frame,
    load_satellite_frames,
)
from Library.SW.Visualization import build_satellite_comparison_frame


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cr", type=int, nargs="?", help="One Carrington rotation number")
    parser.add_argument("--start", help="Start UTC for a sequential CR archive run")
    parser.add_argument("--end", help="Exclusive end UTC for a sequential CR archive run")
    parser.add_argument("--archive-root", type=Path, default=ROOT_DIR / "Outputs" / "SW" / "Archive")
    parser.add_argument("--input-source", choices=("parquet", "sql"), default="sql")
    parser.add_argument("--input-parquet", type=Path, default=data_path("CH Area.parquet"))
    parser.add_argument("--source-guard-days", type=float, default=40.0)
    parser.add_argument("--mode", choices=("hindcast", "forecast"), default="hindcast")
    parser.add_argument("--enlil", action="store_true")
    parser.add_argument("--enlil-parquet", type=Path)
    parser.add_argument("--slow-sw", action="store_true", help="Use the empirical slow-wind patch for ACE")
    parser.add_argument(
        "--satellites",
        default=','.join(DEFAULT_ENABLED_SATELLITES),
        help="Comma-separated configured satellite IDs to include in series.parquet",
    )
    args = parser.parse_args(argv)
    enabled_satellites = parse_satellite_ids(args.satellites)
    assert enabled_satellites, "Propagation requires at least one enabled satellite"
    assert args.source_guard_days > 0
    assert (args.cr is not None) != (args.start is not None or args.end is not None), (
        "Supply one CR number, or both --start and --end"
    )
    if args.cr is None:
        assert args.start is not None and args.end is not None
        for cr in iter_crs(args.start, args.end):
            directory = args.archive_root / f"CR{cr:04d}"
            if directory.exists():
                manifest = json.loads((directory / "manifest.json").read_text())
                assert manifest["cr"] == cr
                for filename in (
                    "cube.h5", "inputs.parquet", "prepared_inputs.parquet", "series.parquet"
                ):
                    assert (directory / filename).exists(), f"Incomplete archive: {directory}"
                print(f"Reusing {directory}")
                continue
            command = [
                sys.executable, str(Path(__file__).resolve()), str(cr),
                "--archive-root", str(args.archive_root),
                "--input-source", args.input_source,
                "--input-parquet", str(args.input_parquet),
                "--source-guard-days", str(args.source_guard_days),
                "--mode", args.mode,
            ]
            if args.enlil:
                command.append("--enlil")
            if args.enlil_parquet is not None:
                command.extend(("--enlil-parquet", str(args.enlil_parquet)))
            if args.slow_sw:
                command.append("--slow-sw")
            command.extend(("--satellites", ",".join(enabled_satellites)))
            subprocess.run(command, check=True)
        return 0

    cr_start, cr_end = cr_bounds(args.cr)
    source_start = cr_start - pd.Timedelta(days=args.source_guard_days)
    input_path = resolve_repo_path(args.input_parquet)
    ballistic = load_ballistic_spec()
    empirical = load_empirical_spec()
    slow_patch_empirical = load_slow_sw_patch_spec()
    step = int(ballistic["output_step_minutes"])
    rotation = compute_rotation_state(ballistic["phi_step_minutes"])

    inputs = load_sw_input_frame(
        start_dt=source_start,
        end_dt=cr_end,
        source=args.input_source,
        input_parquet_path=input_path,
    )
    inputs["source_quality"] = np.where(
        inputs["ch_relative_area"].notna(), "available", "missing"
    )
    prepared = build_model_input_series(
        sdo_input_df=inputs,
        empirical=empirical,
        output_step_minutes=step,
        simulation_pad_days=ballistic["simulation_pad_days"],
    )
    grid, speed, stats = propagate_ballistic(
        df_v_run=prepared["df_v"],
        sim_start=min(prepared["sim_start"], cr_start),
        sim_end=max(prepared["sim_end"], cr_end),
        output_step_minutes=step,
        rotation_state=rotation,
        r0=ballistic["r0"],
        r_max=ballistic["r_max"],
        r_step=ballistic["r_step"],
        maximum_source_gap_hours=ballistic["maximum_input_gap_hours"],
    )
    cube_info = cube_stats(speed, empirical.slow_sw_speed(grid.time_axis))
    frequency = f"{step}min"
    satellite_frames = load_satellite_frames(
        satellite_ids=enabled_satellites,
        time_axis=grid.time_axis,
        time_freq=frequency,
    )
    swx = build_ace_earth_swx_frame(prepared["sdo_input_df"])
    enlil = (
        load_enlil_prediction_frames(
            grid.time_axis, frequency,
            enlil_path=args.enlil_parquet if args.enlil_parquet is not None else None,
        )
        if args.enlil else {}
    )
    comparisons = {}
    for sat, frame in satellite_frames.items():
        comparisons[sat] = build_satellite_comparison_frame(
            time_axis=grid.time_axis,
            phi_axis=grid.phi_axis,
            r_axis=grid.r_axis,
            grid_raw=speed,
            slow_sw_pred_mask=cube_info.slow_wind_mask,
            df_sat=frame,
            df_swx=swx if sat == "ace_earth" else None,
            df_noaa=enlil.get(sat),
            phi_target=ballistic["earth_phi_target"] if sat == "ace_earth" else 0.0,
            r_target=ballistic["earth_r_target"],
            slow_sw_speed=slow_patch_empirical.slow_sw_speed(grid.time_axis),
            slow_sw_patch=args.slow_sw,
            draw_slow_sw=True,
        )

    core = (grid.time_axis >= cr_start) & (grid.time_axis < cr_end)
    assert core.any(), "No output time centres lie in the requested CR"
    satellite_series = pd.concat(comparisons, axis="columns")
    input_series = pd.concat(
        {
            "ch_area": prepared["df_ch_area"],
            "model_input": prepared["df_v"].rename(columns={"v": "v_empirical"}),
        }, axis="columns",
    )
    series = pd.concat(
        {"satellite": satellite_series, "input": input_series}, axis="columns"
    ).sort_index()
    series = series.loc[(series.index >= cr_start) & (series.index < cr_end)]
    sha256 = (
        hashlib.sha256(input_path.read_bytes()).hexdigest()
        if args.input_source == "parquet" else None
    )
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT_DIR, text=True
    ).strip()
    metadata = {
        "model": "continuous_ballistic",
        "mode": args.mode,
        "code_revision": revision,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "input_source": args.input_source,
        "input_path": str(input_path) if args.input_source == "parquet" else None,
        "input_sha256": sha256,
        "source_guard_days": args.source_guard_days,
        "source_start_utc": source_start.isoformat(),
        "previous_cr": args.cr - 1,
        "next_cr": args.cr + 1,
        "input_rows": len(inputs),
        "input_missing_rows": int(inputs["ch_relative_area"].isna().sum()),
        "source_points": stats.source_points,
        "source_segments": stats.source_segments,
        "maximum_input_gap_hours": ballistic["maximum_input_gap_hours"],
        "interpolation_policy": (
            "native knots; continuous segments across gaps up to "
            f"{ballistic['maximum_input_gap_hours']} hours; no extrapolation"
        ),
        "output_step_minutes": step,
        "satellites": enabled_satellites,
        "satellite_coord_frame": {
            sat: get_satellite_config(sat).coord_frame
            for sat in enabled_satellites
        },
        "config": json.loads(Path(ballistic["json_path"]).read_text()),
        "upstream_sql_fill_unresolved": args.input_source == "sql",
    }
    directory = write_cr(
        archive_root=args.archive_root,
        cr=args.cr,
        time_axis=grid.time_axis[core],
        phi_axis=grid.phi_axis,
        r_axis=grid.r_axis,
        speed=speed[core],
        is_slow_wind=cube_info.slow_wind_mask[core],
        inputs=inputs,
        prepared_inputs=prepared["sdo_input_df"],
        series=series,
        metadata=metadata,
    )
    print(
        f"Saved {directory} | {int(core.sum())} frames | "
        f"{stats.prop_seconds:.2f}s propagation | "
        f"{100.0 * np.isfinite(speed[core]).mean():.1f}% finite"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
