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
from Library.SW.Config import (  # noqa: E402
    load_ballistic_spec,
    load_empirical_spec,
    load_sw_runtime_spec,
)
from Library.SW.Coords import (  # noqa: E402
    build_grid_axes,
    build_packet_geometry,
    build_transport_state,
    compute_rotation_state,
)
from Library.SW.Inputs import (  # noqa: E402
    build_forecast_earth_frame,
    build_model_input_series,
    load_ace_at_earth,
    load_sw_input_frame,
)
from Library.SW.Visualization import (  # noqa: E402
    build_earth_comparison_frame,
    export_polar_animation,
)


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Build the SW animation and Earth-series parquet for a date range."
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
        default="parquet",
        help="Propagation input source.",
    )
    parser.add_argument(
        "--input-parquet",
        default="Data/CH Area.parquet",
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
        help="Optional explicit Earth-series parquet output path.",
    )
    parser.add_argument(
        "--skip-animation",
        action="store_true",
        help="Skip MP4 export.",
    )
    parser.add_argument(
        "--skip-parquet",
        action="store_true",
        help="Skip Earth-series parquet export.",
    )
    parser.add_argument(
        "--animation-style",
        choices=["mesh", "scatter"],
        default="mesh",
        help="Polar animation rendering style.",
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
    return parser.parse_args(argv[1:])


def default_output_paths(output_dir, start_dt, end_dt):
    stamp = f"{start_dt:%Y%m%d_%H%M}-{end_dt:%Y%m%d_%H%M}"
    animation_out = output_dir / f"SW Polar Animation {stamp}.mp4"
    parquet_out = output_dir / f"SW Earth Series {stamp}.parquet"
    return animation_out, parquet_out


def main(argv):
    args = parse_args(argv)
    start_dt = pd.Timestamp(args.start)
    end_dt = pd.Timestamp(args.end)

    empirical = load_empirical_spec()
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
    default_animation_out, default_parquet_out = default_output_paths(
        output_dir=output_dir,
        start_dt=start_dt,
        end_dt=end_dt,
    )
    animation_out = (
        Path(args.animation_out)
        if args.animation_out is not None
        else default_animation_out
    )
    parquet_out = (
        Path(args.parquet_out) if args.parquet_out is not None else default_parquet_out
    )

    df_sdo_sw = load_sw_input_frame(
        start_dt=start_dt,
        end_dt=end_dt,
        source=args.input_source,
        input_parquet_path=args.input_parquet,
    )
    prepared = build_model_input_series(
        sdo_input_df=df_sdo_sw,
        empirical=empirical,
        superresolution_enabled=superresolution_enabled,
        time_freq=time_freq,
        simulation_pad_days=ballistic["simulation_pad_days"],
    )

    rotation = compute_rotation_state(
        cr_days=ballistic["cr_days"],
        phi_step_minutes=ballistic["phi_step_minutes"],
    )
    grid = build_grid_axes(
        sim_start=prepared["sim_start"],
        sim_end=prepared["sim_end"],
        time_freq=time_freq,
        phi_step=rotation.phi_step,
        r0=ballistic["r0"],
        r_max=ballistic["r_max"],
        dense_memory_budget_gb=runtime["dense_memory_budget_gb"],
        memory_guard_enabled=ballistic["memory_guard_enabled"],
    )
    transport = build_transport_state(
        time_axis=grid.time_axis,
        phi_axis=grid.phi_axis,
        rotation_state=rotation,
        horizon_hours=ballistic["horizon_hours"],
        time_step_hours=time_step_hours,
        field_half_width_h=ballistic["field_half_width_h"],
        r_solar_km=ballistic["r_solar_km"],
    )

    df_v_run = (
        prepared["df_v"]
        .loc[
            (prepared["df_v"].index >= grid.time_axis.min())
            & (prepared["df_v"].index <= grid.time_axis.max())
        ]
        .copy()
    )
    packet_p, packet_off, packet_alpha = build_packet_geometry(
        phi_delay_steps=transport.phi_delay_steps,
        field_half_width_steps=transport.field_half_width_steps,
    )
    accumulators = init_accumulators(
        n_t=len(grid.time_axis),
        n_p=len(grid.phi_axis),
        n_r=len(grid.r_axis),
        use_cr_reset=ballistic["use_cr_reset"],
    )
    (
        _seed_times,
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
    )
    stats = run_bulk_propagation(
        seed_vals=seed_vals,
        v_prev=v_prev,
        v_next=v_next,
        seed_t_idx=seed_t_idx,
        seed_cr_idx_arr=seed_cr_idx_arr,
        seed_r_idx=seed_r_idx,
        h_step_idx=transport.h_step_idx,
        packet_off=packet_off,
        packet_p=packet_p,
        packet_alpha=packet_alpha,
        n_t=len(grid.time_axis),
        n_p=len(grid.phi_axis),
        n_r=len(grid.r_axis),
        max_flat=accumulators.max_flat,
        cr_flat=accumulators.cr_flat,
        use_swept_cell=ballistic["use_swept_cell"],
        use_cr_reset=ballistic["use_cr_reset"],
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
    )

    post = postprocess_max_field(
        V_accum_max=accumulators.V_accum_max,
        slow_sw_speed=empirical.slow_sw_speed(grid.time_axis),
        post_chunk_t=runtime["post_chunk_t"],
    )
    df_ace_earth = load_ace_at_earth()
    df_forecast_earth = build_forecast_earth_frame(prepared["sdo_input_df"])
    earth_frame = build_earth_comparison_frame(
        time_axis=grid.time_axis,
        phi_axis=grid.phi_axis,
        r_axis=grid.r_axis,
        grid_raw=post.V_grid,
        slow_sw_pred_mask=post.max_slow_sw_pred_mask,
        slow_sw_speed=empirical.slow_sw_speed(grid.time_axis),
        df_ace_earth=df_ace_earth,
        df_forecast_earth=df_forecast_earth,
        phi_target=ballistic["earth_phi_target"],
        r_target=ballistic["earth_r_target"],
        draw_slow_sw=True,
        backfill_empty_with_300=False,
    )
    earth_frame_window = earth_frame.loc[grid.time_axis.min() : grid.time_axis.max()]

    if not args.skip_animation:
        animation_stats = export_polar_animation(
            anim_outfile=animation_out,
            time_axis=grid.time_axis,
            phi_axis=grid.phi_axis,
            r_axis=grid.r_axis,
            grid_raw=post.V_grid,
            post_vlims_raw=post.max_vlims_raw,
            slow_sw_pred_mask=post.max_slow_sw_pred_mask,
            earth_frame=earth_frame,
            time_step_minutes=time_step_minutes,
            superresolution_enabled=superresolution_enabled,
            slow_sw_speed=empirical.slow_sw_speed(grid.time_axis),
            r0=ballistic["r0"],
            cr_days=ballistic["cr_days"],
            draw_slow_sw=True,
            backfill_empty_with_300=False,
            anim_fps=args.animation_fps,
            anim_dpi=(
                runtime["animation_dpi"]
                if args.animation_dpi is None
                else args.animation_dpi
            ),
            anim_plot_style=args.animation_style,
        )
        print(
            "Saved animation:",
            animation_out,
            "| frames:",
            int(animation_stats["frames"]),
            "| achieved speedup:",
            f"{animation_stats['achieved_speedup']:.2f}x",
        )

    if not args.skip_parquet:
        earth_frame_window.to_parquet(parquet_out)
        print("Saved Earth-series parquet:", parquet_out)

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
