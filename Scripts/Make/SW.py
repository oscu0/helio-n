#!/usr/bin/env python3
import argparse
import os
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
os.environ.setdefault("MPLCONFIGDIR", "/tmp/helio_n_matplotlib")
sys.path.append(str(ROOT_DIR))

from Library.SW.Ballistic import (  # noqa: E402
    init_accumulators,
    postprocess_max_field,
    prepare_seed_inputs,
    run_bulk_propagation,
)
from Library.Paths import data_path, resolve_repo_path  # noqa: E402
from Library.SW.Config import (  # noqa: E402
    load_ballistic_spec,
    load_empirical_spec,
    load_slow_sw_patch_spec,
    load_sw_runtime_spec,
)
from Library.SW.Coords import (  # noqa: E402
    build_grid_axes,
    build_transport_state,
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
from Library.SW.Stats import (  # noqa: E402
    export_sw_forecast_cr_stats_csv,
    export_sw_forecast_stats_csv,
)
from Library.SW.Visualization import (  # noqa: E402
    build_satellite_comparison_frame,
    export_polar_animation,
    export_solar_wind_plot,
)


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Build the SW animation and satellite-series parquet for a date range."
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
        "--skip-parquet",
        action="store_true",
        help="Skip satellite-series and reproduction parquet exports.",
    )
    parser.add_argument(
        "--skip-stats",
        action="store_true",
        help="Skip forecast stats CSV export.",
    )
    parser.add_argument(
        "--skip-cr-stats",
        action="store_true",
        help="Skip per-Carrington-rotation forecast stats CSV export.",
    )
    parser.add_argument(
        "--skip-sw-plot",
        action="store_true",
        help="Skip solar wind time-series PDF export.",
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
        help="Optional ENLIL parquet path for v_noaa comparison traces.",
    )
    parser.add_argument(
        "--skip-enlil",
        action="store_true",
        help="Skip ENLIL/NOAA comparison traces and stats.",
    )
    parser.add_argument(
        "--stats-out",
        default=None,
        help="Optional explicit forecast stats CSV output path.",
    )
    parser.add_argument(
        "--cr-stats-out",
        default=None,
        help="Optional explicit per-Carrington-rotation stats CSV output path.",
    )
    parser.add_argument(
        "--sw-plot-out",
        default=None,
        help="Optional explicit solar wind time-series PDF output path.",
    )
    return parser.parse_args(argv[1:])


def main(argv):
    args = parse_args(argv)
    start_dt = pd.Timestamp(args.start)
    end_dt = pd.Timestamp(args.end)
    assert start_dt < end_dt, (
        f"Expected start datetime before end datetime; got start={start_dt} "
        f"and end={end_dt}"
    )

    empirical = load_empirical_spec()
    slow_sw_patch_empirical = load_slow_sw_patch_spec()
    ballistic = load_ballistic_spec()
    runtime = load_sw_runtime_spec()
    superresolution_enabled = bool(ballistic["superresolution_enabled"])
    time_step_minutes = (
        int(ballistic["superresolution_step_minutes"])
        if superresolution_enabled
        else int(ballistic["base_time_step_minutes"])
    )
    time_step_hours = float(time_step_minutes) / 60.0
    time_freq = f"{int(time_step_minutes)}min"

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
    stats_out = (
        Path(args.stats_out)
        if args.stats_out is not None
        else animation_out.with_name(f"{animation_out.stem} Stats.csv")
    )
    cr_stats_out = (
        Path(args.cr_stats_out)
        if args.cr_stats_out is not None
        else animation_out.with_name(f"{animation_out.stem} Per-CR Stats.csv")
    )
    sw_plot_out = (
        Path(args.sw_plot_out)
        if args.sw_plot_out is not None
        else output_dir / f"SW Time Series {stamp}.pdf"
    )

    df_sdo_sw = load_sw_input_frame(
        start_dt=start_dt,
        end_dt=end_dt,
        source=args.input_source,
        input_parquet_path=resolve_repo_path(args.input_parquet),
    )
    prepared = build_model_input_series(
        sdo_input_df=df_sdo_sw,
        empirical=empirical,
        superresolution_enabled=superresolution_enabled,
        time_freq=time_freq,
        simulation_pad_days=ballistic["simulation_pad_days"],
    )

    rotation = compute_rotation_state(
        phi_step_minutes=ballistic["phi_step_minutes"],
    )
    grid = build_grid_axes(
        sim_start=prepared["sim_start"],
        sim_end=prepared["sim_end"],
        time_freq=time_freq,
        phi_step=rotation.phi_step,
        r0=ballistic["r0"],
        r_max=ballistic["r_max"],
        r_step=ballistic["r_step"],
        dense_memory_budget_gb=runtime["dense_memory_budget_gb"],
        memory_guard_enabled=ballistic["memory_guard_enabled"],
    )
    transport = build_transport_state(
        time_axis=grid.time_axis,
        phi_axis=grid.phi_axis,
        rotation_state=rotation,
        horizon_hours=ballistic["horizon_hours"],
        time_step_hours=time_step_hours,
    )

    df_v_run = (
        prepared["df_v"]
        .loc[
            (prepared["df_v"].index >= grid.time_axis.min())
            & (prepared["df_v"].index <= grid.time_axis.max())
        ]
        .copy()
    )
    accumulators = init_accumulators(
        n_t=len(grid.time_axis),
        n_p=len(grid.phi_axis),
        n_r=len(grid.r_axis),
    )
    (
        seed_vals,
        v_prev,
        v_next,
        seed_t_idx,
        seed_cr_idx_arr,
        seed_r_idx,
    ) = prepare_seed_inputs(
        df_v_run=df_v_run,
        cr_steps=transport.cr_steps,
        horizon_steps=transport.horizon_steps,
        time_freq=time_freq,
        t0_ref=transport.t0_ref,
        time_step_hours=time_step_hours,
        r_kernel_scale=transport.r_kernel_scale,
        r0=ballistic["r0"],
        r_axis=grid.r_axis,
    )
    stats = run_bulk_propagation(
        seed_vals=seed_vals,
        v_prev=v_prev,
        v_next=v_next,
        seed_t_idx=seed_t_idx,
        seed_cr_idx_arr=seed_cr_idx_arr,
        seed_r_idx=seed_r_idx,
        h_step_idx=transport.h_step_idx,
        phi_delay_offsets=transport.phi_delay_offsets,
        phi_delay_alpha=transport.phi_delay_alpha,
        n_t=len(grid.time_axis),
        n_p=len(grid.phi_axis),
        n_r=len(grid.r_axis),
        V_accum_max=accumulators.V_accum_max,
        cr_flat=accumulators.cr_flat,
        max_seed_batch=runtime["max_seed_batch"],
    )
    print(
        "Propagation runtime:",
        f"{stats.prop_seconds:.2f}s",
        "| seeds:",
        stats.seeds_processed,
        "| filled cells:",
        stats.filled,
        "/",
        stats.total,
        "| deposits:",
        stats.deposits,
        "| swept-skip deposits:",
        stats.swept_skip_deposits,
        "| avg deposits/seed:",
        f"{stats.avg_deposits_per_seed:.1f}",
        "| avg swept-skip deposits/seed:",
        f"{stats.avg_swept_skip_deposits_per_seed:.1f}",
        "| avg deposits/filled cell:",
        f"{stats.avg_deposits_per_filled_cell:.2f}",
    )

    post = postprocess_max_field(
        V_accum_max=accumulators.V_accum_max,
        slow_sw_speed=empirical.slow_sw_speed(grid.time_axis),
        post_chunk_t=runtime["post_chunk_t"],
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
    satellite_frames = {
        "ace_earth": load_ace_earth_frame(),
        "stereo_a": load_stereo_a_frame(
            time_axis=grid.time_axis,
            time_freq=time_freq,
        ),
    }
    satellite_swx_frames = {
        "ace_earth": build_ace_earth_swx_frame(prepared["sdo_input_df"])
    }
    enlil_frames = {}
    if not args.skip_enlil:
        enlil_frames = load_enlil_prediction_frames(
            time_axis=grid.time_axis,
            time_freq=time_freq,
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
            slow_sw_patch=True,
            draw_slow_sw=True,
        )
    sat_labels = {spec["sat"]: spec["label"] for spec in plot_sats}
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
            time_step_minutes=time_step_minutes,
            slow_sw_speed=slow_sw_speed,
            draw_slow_sw=True,
            anim_fps=args.animation_fps,
            anim_1h_mult=runtime["animation_1h_mult"],
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
            "| stride:",
            int(animation_stats["stride"]),
            "| fps:",
            int(animation_stats["fps"]),
        )

    if not args.skip_parquet:
        satellite_frame_window.to_parquet(parquet_out)
        print("Saved satellite-series parquet:", parquet_out)
        reproduction_frame.to_parquet(reproduction_parquet_out)
        print("Saved reproduction parquet:", reproduction_parquet_out)

    if not args.skip_sw_plot:
        export_solar_wind_plot(
            plot_outfile=sw_plot_out,
            comparison_frames=comparison_frames,
            start_dt=start_dt,
            end_dt=end_dt,
            sat_labels=sat_labels,
        )
        print("Saved solar wind plot:", sw_plot_out)

    if not args.skip_stats:
        _stats_tables, stats_csv_frame = export_sw_forecast_stats_csv(
            csv_outfile=stats_out,
            comparison_frames=comparison_frames,
            start_dt=start_dt,
            end_dt=end_dt,
            time_axis=grid.time_axis,
            slow_sw_speed=slow_sw_speed,
            sat_labels=sat_labels,
        )
        print("Saved forecast stats CSV:", stats_out, "| rows:", len(stats_csv_frame))
        if not args.skip_cr_stats:
            cr_stats_csv_frame = export_sw_forecast_cr_stats_csv(
                csv_outfile=cr_stats_out,
                comparison_frames=comparison_frames,
                start_dt=start_dt,
                end_dt=end_dt,
                time_axis=grid.time_axis,
                slow_sw_speed=slow_sw_speed,
                sat_labels=sat_labels,
            )
            print(
                "Saved per-CR forecast stats CSV:",
                cr_stats_out,
                "| rows:",
                len(cr_stats_csv_frame),
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
