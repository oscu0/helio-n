#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
os.environ.setdefault("MPLCONFIGDIR", "/tmp/helio_n_matplotlib")
sys.path.append(str(ROOT_DIR))

from Library.SW.Ballistic import (  # noqa: E402
    postprocess_max_field,
    propagate_ballistic,
)
from Library.Paths import data_path, resolve_repo_path  # noqa: E402
from Library.SW.Config import (  # noqa: E402
    load_ballistic_spec,
    load_empirical_spec,
    load_slow_sw_patch_spec,
    load_sw_runtime_spec,
)
from Library.SW.Constants import CARRINGTON_ROTATION_DAYS  # noqa: E402
from Library.SW.Coords import (  # noqa: E402
    build_centered_time_axis,
    compute_rotation_state,
)
from Library.SW.Inputs import (  # noqa: E402
    build_ace_earth_swx_frame,
    build_model_input_series,
    load_enlil_prediction_frames,
    load_ace_earth_frame,
    load_stereo_a_frame,
    load_sw_input_frame,
)
from Library.SW.Visualization import (  # noqa: E402
    build_satellite_comparison_frame,
    export_polar_animation,
    find_phi_index,
)


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Propagate SW and export the animation and reproduction parquets."
    )
    parser.add_argument(
        "start", help="Inclusive start datetime accepted by pandas.Timestamp"
    )
    parser.add_argument(
        "end", help="Exclusive end datetime accepted by pandas.Timestamp"
    )
    parser.add_argument(
        "--input-source",
        choices=["parquet", "sql"],
        default="sql",
        help="Propagation input source.",
    )
    parser.add_argument(
        "--input-parquet",
        default=str(data_path("CH Area.parquet")),
        help="Parquet input used when --input-source parquet.",
    )
    parser.add_argument(
        "--output-dir",
        default="Outputs/SW",
        help="Directory for generated SW artifacts.",
    )
    parser.add_argument(
        "--animation-out",
        default=None,
        help="Optional explicit animation output path.",
    )
    parser.add_argument(
        "--parquet-out",
        default=None,
        help="Optional explicit satellite-series parquet output path.",
    )
    parser.add_argument(
        "--reproduction-parquet-out",
        default=None,
        help="Optional explicit reproduction parquet output path with cached inputs.",
    )
    parser.add_argument(
        "--skip-animation",
        action="store_true",
        help="Skip MP4 export.",
    )
    parser.add_argument(
        "--targets-only",
        action="store_true",
        help=(
            "Propagate only full-grid longitude bins sampled by Earth and "
            "STEREO-A. Requires --skip-animation and preserves their time series."
        ),
    )
    parser.add_argument(
        "--skip-parquet",
        action="store_true",
        help="Skip satellite-series and reproduction parquet exports.",
    )
    parser.add_argument(
        "--animation-fps",
        type=int,
        default=30,
        help="Animation frames per second.",
    )
    parser.add_argument(
        "--animation-dpi",
        type=int,
        default=None,
        help="Optional override for animation DPI.",
    )
    parser.add_argument(
        "--enlil-parquet",
        default=None,
        help="Optional ENLIL parquet path used with --enlil.",
    )
    parser.add_argument(
        "--enlil",
        action="store_true",
        help="Include ENLIL/NOAA series in the reproduction parquet. Default: off.",
    )
    parser.add_argument(
        "--slow-sw",
        action="store_true",
        help=(
            "Apply the empirical slow-wind patch to ACE. "
            "Default: use the raw constant-filled prediction."
        ),
    )
    parser.add_argument(
        "--stereo-next-cr",
        action="store_true",
        help=(
            "Use the following Carrington rotation's CH input for the "
            "STEREO-A comparison. Default: use the wrapped "
            "previous-rotation branch."
        ),
    )
    return parser.parse_args(argv[1:])


def main(argv):
    args = parse_args(argv)
    assert not args.targets_only or args.skip_animation, (
        "--targets-only requires --skip-animation because it does not build "
        "the full longitude grid needed by the polar animation."
    )
    start_dt = pd.Timestamp(args.start)
    end_dt = pd.Timestamp(args.end)
    assert start_dt < end_dt, (
        f"Expected start datetime before end datetime; got start={start_dt} "
        f"and end={end_dt}"
    )
    input_end_dt = end_dt
    if args.stereo_next_cr:
        input_end_dt += pd.Timedelta(days=CARRINGTON_ROTATION_DAYS)

    empirical = load_empirical_spec()
    slow_sw_patch_empirical = load_slow_sw_patch_spec()
    ballistic = load_ballistic_spec()
    runtime = load_sw_runtime_spec()
    output_step_minutes = int(ballistic["output_step_minutes"])
    assert output_step_minutes > 0
    output_frequency = f"{output_step_minutes}min"

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = f"{start_dt:%Y%m%d_%H%M}-{end_dt:%Y%m%d_%H%M}"
    animation_out = (
        Path(args.animation_out)
        if args.animation_out is not None
        else output_dir / f"SW Polar Animation {stamp}.mp4"
    )
    parquet_out = (
        Path(args.parquet_out)
        if args.parquet_out is not None
        else output_dir / f"SW Satellite Series {stamp}.parquet"
    )
    reproduction_parquet_out = (
        Path(args.reproduction_parquet_out)
        if args.reproduction_parquet_out is not None
        else output_dir / f"SW Reproduction Series {stamp}.parquet"
    )
    df_sdo_sw = load_sw_input_frame(
        start_dt=start_dt,
        end_dt=input_end_dt,
        source=args.input_source,
        input_parquet_path=resolve_repo_path(args.input_parquet),
    )
    prepared = build_model_input_series(
        sdo_input_df=df_sdo_sw,
        empirical=empirical,
        output_step_minutes=output_step_minutes,
        simulation_pad_days=ballistic["simulation_pad_days"],
    )

    rotation = compute_rotation_state(
        phi_step_minutes=ballistic["phi_step_minutes"],
    )
    simulation_time_axis = build_centered_time_axis(
        prepared["sim_start"],
        prepared["sim_end"],
        output_step_minutes,
    )
    satellite_frames = {
        "ace_earth": load_ace_earth_frame(),
        "stereo_a": load_stereo_a_frame(
            time_axis=simulation_time_axis,
            time_freq=output_frequency,
        ),
    }
    df_v_run = (
        prepared["df_v"]
        .loc[
            (prepared["df_v"].index >= simulation_time_axis.min())
            & (prepared["df_v"].index <= simulation_time_axis.max())
        ]
        .copy()
    )

    phi_values = None
    if args.targets_only:
        full_phi_axis = np.arange(
            0.0,
            360.0,
            rotation.phi_step,
            dtype=float,
        )
        stereo_phi = (
            satellite_frames["stereo_a"]["phi_target"]
            .interpolate(method="time")
            .ffill()
            .bfill()
            .to_numpy(dtype=float)
        )
        target_phi_indices = {
            find_phi_index(full_phi_axis, ballistic["earth_phi_target"])
        }
        target_phi_indices.update(
            find_phi_index(full_phi_axis, value)
            for value in stereo_phi[np.isfinite(stereo_phi)]
        )
        target_phi_values = full_phi_axis[sorted(target_phi_indices)]
        print(
            "Target-only propagation:",
            len(target_phi_values),
            "/",
            len(full_phi_axis),
            "full-grid longitude bins",
        )
        phi_values = target_phi_values

    grid, V_grid, stats = propagate_ballistic(
        df_v_run=df_v_run,
        sim_start=prepared["sim_start"],
        sim_end=prepared["sim_end"],
        output_step_minutes=output_step_minutes,
        rotation_state=rotation,
        r0=ballistic["r0"],
        r_max=ballistic["r_max"],
        r_step=ballistic["r_step"],
        maximum_source_gap_hours=ballistic["maximum_input_gap_hours"],
        phi_values=phi_values,
    )
    cr_steps = int(round(rotation.cr_time / (output_step_minutes * 60.0)))
    if args.stereo_next_cr:
        requested_time_axis = build_centered_time_axis(
            start_dt,
            end_dt,
            output_step_minutes,
        )
        required_grid_end = requested_time_axis[-1] + pd.Timedelta(
            minutes=output_step_minutes * cr_steps
        )
        assert grid.time_axis.max() >= required_grid_end, (
            "The propagated grid does not cover the full STEREO next-CR "
            f"comparison interval: required through {required_grid_end}, "
            f"available through {grid.time_axis.max()}"
        )
        print(
            "STEREO CH source: next Carrington rotation",
            "| grid offset steps:",
            cr_steps,
            "| input end:",
            input_end_dt,
        )

    print(
        "Propagation runtime:",
        f"{stats.prop_seconds:.2f}s",
        "| source points:",
        stats.source_points,
        "| source segments:",
        stats.source_segments,
        "| filled cells:",
        stats.filled,
        "/",
        stats.total,
        "| radial-bin visits:",
        stats.radial_bin_visits,
        "| base-plane cells:",
        stats.base_plane_cells,
    )

    post = postprocess_max_field(
        V_grid=V_grid,
        slow_sw_speed=empirical.slow_sw_speed(grid.time_axis),
    )
    print(
        "Post-max cells:",
        "slow:",
        post.slow_cells,
        "| non-slow:",
        post.non_slow_cells,
        "| non-slow / filled:",
        f"{100.0 * post.non_slow_fraction_filled:.2f}%",
    )
    slow_sw_speed = empirical.slow_sw_speed(grid.time_axis)
    slow_sw_patch_speed = slow_sw_patch_empirical.slow_sw_speed(grid.time_axis)
    plot_sats = [
        {
            "sat": "ace_earth",
            "label": "ACE @ Earth",
            "phi_target": ballistic["earth_phi_target"],
            "r_target": ballistic["earth_r_target"],
        },
        {
            "sat": "stereo_a",
            "label": "STEREO-A",
            "phi_target": 0.0,
            "r_target": ballistic["earth_r_target"],
        },
    ]
    satellite_swx_frames = {
        "ace_earth": build_ace_earth_swx_frame(prepared["sdo_input_df"])
    }
    enlil_frames = {}
    if args.enlil:
        enlil_frames = load_enlil_prediction_frames(
            time_axis=grid.time_axis,
            time_freq=output_frequency,
            enlil_path=args.enlil_parquet
            if args.enlil_parquet is not None
            else None,
        )
    comparison_frames = {}
    for sat_spec in plot_sats:
        sat_name = sat_spec["sat"]
        df_sat = satellite_frames[sat_name].copy()
        df_sat.attrs["label"] = sat_spec["label"]
        comparison_frames[sat_name] = build_satellite_comparison_frame(
            time_axis=grid.time_axis,
            phi_axis=grid.phi_axis,
            r_axis=grid.r_axis,
            grid_raw=post.V_grid,
            slow_sw_pred_mask=post.max_slow_sw_pred_mask,
            df_sat=df_sat,
            df_swx=satellite_swx_frames.get(sat_name),
            df_noaa=enlil_frames.get(sat_name),
            phi_target=sat_spec["phi_target"],
            r_target=sat_spec["r_target"],
            slow_sw_speed=slow_sw_patch_speed,
            slow_sw_patch=args.slow_sw,
            draw_slow_sw=True,
            prediction_time_offset_steps=(
                cr_steps
                if args.stereo_next_cr and sat_name == "stereo_a"
                else 0
            ),
        )
    satellite_frame_window = pd.concat(
        {
            sat_name: frame.loc[grid.time_axis.min() : grid.time_axis.max()]
            for sat_name, frame in comparison_frames.items()
        },
        axis="columns",
    )
    input_frame_window = pd.concat(
        {
            "ch_area": prepared["df_ch_area"].loc[
                grid.time_axis.min() : grid.time_axis.max()
            ],
            "model_input": prepared["df_v"].loc[
                grid.time_axis.min() : grid.time_axis.max()
            ].rename(columns={"v": "v_empirical"}),
        },
        axis="columns",
    )
    reproduction_frame = pd.concat(
        {
            "satellite": satellite_frame_window,
            "input": input_frame_window,
        },
        axis="columns",
    )

    if not args.skip_animation:
        animation_stats = export_polar_animation(
            anim_outfile=animation_out,
            time_axis=grid.time_axis,
            phi_axis=grid.phi_axis,
            r_axis=grid.r_axis,
            grid_raw=post.V_grid,
            post_vlims_raw=post.max_vlims_raw,
            slow_sw_pred_mask=post.max_slow_sw_pred_mask,
            comparison_frames=comparison_frames,
            slow_sw_speed=slow_sw_speed,
            draw_slow_sw=True,
            anim_fps=args.animation_fps,
            anim_dpi=(
                runtime["animation_dpi"]
                if args.animation_dpi is None
                else args.animation_dpi
            ),
        )
        print(
            "Saved animation:",
            animation_out,
            "| frames:",
            int(animation_stats["frames"]),
            "| fps:",
            int(animation_stats["fps"]),
        )

    if not args.skip_parquet:
        satellite_frame_window.to_parquet(parquet_out)
        print("Saved satellite-series parquet:", parquet_out)
        reproduction_frame.to_parquet(reproduction_parquet_out)
        print("Saved reproduction parquet:", reproduction_parquet_out)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
