import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/helio_n_matplotlib")

import matplotlib.pyplot as plt
import numba as nb
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from Library.SW.Config import load_ballistic_spec, load_empirical_spec
from Library.SW.Constants import SOLAR_RADIUS_KM
from Library.SW.Coords import build_r_axis, compute_rotation_state
from Library.SW.Inputs import load_stereo_a_frame, load_sw_input_frame


# One-off comparison controls.
CANDIDATE_START = pd.Timestamp("2018-04-01 12:00:00")
CANDIDATE_END = pd.Timestamp("2018-04-28 12:00:00")
CANDIDATE_FREQ = "6h"
FIXED_SNAPSHOT_TIME = pd.Timestamp("2018-04-22 00:00:00")
INPUT_LEAD = pd.Timedelta(days=45)
INPUT_TRAIL = pd.Timedelta(days=10)
CADENCES_MINUTES = (60, 2)
SWEPT_CELLS_BY_CADENCE = {60: False, 2: True}
OUTPUT_DIR = ROOT_DIR / "Outputs" / "SW" / "Plots"
OUTPUT_STEM = "SW Snapshot 60min No-Swept vs 2min Swept"


@nb.njit(cache=True)
def nearest_axis_index(axis, value):
    hi = int(np.searchsorted(axis, value, side="left"))
    if hi <= 0:
        return 0
    if hi >= len(axis):
        return len(axis)
    lo = hi - 1
    if abs(value - axis[lo]) <= abs(axis[hi] - value):
        return lo
    return hi


@nb.njit(cache=True)
def propagate_snapshot_kernel(
    seed_values,
    seed_previous,
    seed_next,
    seed_time_indices,
    seed_cr_indices,
    snapshot_time_index,
    horizon_steps,
    time_step_hours,
    phi_delay_offsets,
    phi_delay_alpha,
    swept_cells_enabled,
    r0,
    r_axis,
):
    frame = np.full((len(phi_delay_offsets), len(r_axis)), np.nan, dtype=np.float32)
    frame_cr = np.full((len(phi_delay_offsets), len(r_axis)), -1, dtype=np.int32)

    for seed_idx in range(len(seed_values)):
        speed = seed_values[seed_idx]
        value_left = 0.5 * (seed_previous[seed_idx] + speed)
        value_right = 0.5 * (speed + seed_next[seed_idx])
        value_delta = value_right - value_left
        seed_time_idx = int(seed_time_indices[seed_idx])
        seed_cr_idx = int(seed_cr_indices[seed_idx])

        for phi_idx in range(len(phi_delay_offsets)):
            propagation_step = (
                snapshot_time_index
                - seed_time_idx
                - int(phi_delay_offsets[phi_idx])
            )
            if propagation_step < 0 or propagation_step >= horizon_steps:
                continue

            radius = (
                float(r0)
                + float(speed)
                * (float(propagation_step) * float(time_step_hours) * 3600.0)
                / float(SOLAR_RADIUS_KM)
            )
            radius_idx = nearest_axis_index(r_axis, radius)

            if not swept_cells_enabled or propagation_step == 0:
                radius_previous_idx = radius_idx
            else:
                radius_previous = (
                    float(r0)
                    + float(speed)
                    * (
                        float(propagation_step - 1)
                        * float(time_step_hours)
                        * 3600.0
                    )
                    / float(SOLAR_RADIUS_KM)
                )
                radius_previous_idx = nearest_axis_index(r_axis, radius_previous)

            lo = min(radius_previous_idx, radius_idx)
            hi = max(radius_previous_idx, radius_idx)
            if lo >= len(r_axis) or hi < 0:
                continue
            lo = max(0, lo)
            hi = min(len(r_axis) - 1, hi)

            value = value_left + phi_delay_alpha[phi_idx] * value_delta
            for radius_cell_idx in range(lo, hi + 1):
                if frame_cr[phi_idx, radius_cell_idx] != seed_cr_idx:
                    frame_cr[phi_idx, radius_cell_idx] = seed_cr_idx
                    frame[phi_idx, radius_cell_idx] = value
                elif value > frame[phi_idx, radius_cell_idx]:
                    frame[phi_idx, radius_cell_idx] = value

    return frame


def prepare_cadence_state(
    hourly_speed,
    cadence_minutes,
    rotation,
    phi_axis,
    swept_cells_enabled=True,
):
    cadence = pd.Timedelta(minutes=int(cadence_minutes))
    cadence_index = pd.date_range(
        hourly_speed.index.min(),
        hourly_speed.index.max(),
        freq=cadence,
    )
    speed = hourly_speed.reindex(cadence_index).interpolate(method="time")
    assert speed.notna().all()

    values = speed.to_numpy(dtype=np.float32)
    previous = np.empty_like(values)
    following = np.empty_like(values)
    previous[0] = values[0]
    previous[1:] = values[:-1]
    following[-1] = values[-1]
    following[:-1] = values[1:]

    step_hours = float(cadence / pd.Timedelta(hours=1))
    seed_time_indices = np.arange(len(speed), dtype=np.int32)
    cr_steps = int(round((rotation.cr_time / 3600.0) / step_hours))
    seed_cr_indices = (seed_time_indices // cr_steps).astype(np.int32)
    phi_delay_hours = (phi_axis / rotation.omega) / 3600.0
    phi_delay_steps = phi_delay_hours / step_hours

    return {
        "cadence_minutes": int(cadence_minutes),
        "swept_cells_enabled": bool(swept_cells_enabled),
        "index": speed.index,
        "values": values,
        "previous": previous,
        "following": following,
        "seed_time_indices": seed_time_indices,
        "seed_cr_indices": seed_cr_indices,
        "step_hours": step_hours,
        "horizon_steps": int(round(168.0 / step_hours)),
        "phi_delay_offsets": np.floor(phi_delay_steps).astype(np.int32),
        "phi_delay_alpha": np.clip(
            phi_delay_steps - np.floor(phi_delay_steps),
            0.0,
            1.0,
        ).astype(np.float32),
    }


def propagate_snapshot(state, snapshot_time, r0, r_axis):
    snapshot_time = pd.Timestamp(snapshot_time).floor(
        f"{state['cadence_minutes']}min"
    )
    snapshot_time_index = int(
        (snapshot_time - state["index"][0])
        / pd.Timedelta(minutes=state["cadence_minutes"])
    )
    return propagate_snapshot_kernel(
        seed_values=state["values"],
        seed_previous=state["previous"],
        seed_next=state["following"],
        seed_time_indices=state["seed_time_indices"],
        seed_cr_indices=state["seed_cr_indices"],
        snapshot_time_index=snapshot_time_index,
        horizon_steps=state["horizon_steps"],
        time_step_hours=state["step_hours"],
        phi_delay_offsets=state["phi_delay_offsets"],
        phi_delay_alpha=state["phi_delay_alpha"],
        swept_cells_enabled=state["swept_cells_enabled"],
        r0=float(r0),
        r_axis=r_axis,
    )


def add_polar_frame(
    axis,
    frame,
    phi_axis,
    r_axis,
    title,
    coverage,
    vmin,
    vmax,
    earth_r,
    stereo_phi,
    stereo_r,
):
    cmap = plt.cm.plasma.copy()
    cmap.set_bad("white")
    mesh = axis.pcolormesh(
        np.deg2rad(phi_axis),
        r_axis,
        frame.T,
        shading="nearest",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        rasterized=True,
    )
    axis.set_title(title, fontsize=15, pad=18)
    axis.text(
        0.78,
        0.94,
        f"finite cells: {coverage:.1%}",
        transform=axis.transAxes,
        ha="center",
        va="center",
        fontsize=10,
        color="0.25",
        bbox={
            "boxstyle": "round,pad=0.28",
            "facecolor": "white",
            "edgecolor": "0.8",
            "alpha": 0.88,
        },
        zorder=10,
    )
    axis.set_ylim(0.0, float(r_axis.max()) + 1.0)
    axis.set_yticks([50, 100, 150, 200])
    axis.set_yticklabels([])
    axis.grid(color="0.75", linewidth=0.7, alpha=0.7)
    axis.scatter([0.0], [0.0], s=95, color="#f6c945", edgecolor="#8c6a00", zorder=8)
    axis.plot(
        [0.0],
        [min(float(earth_r), float(r_axis.max()) - 1.0)],
        marker="o",
        color="black",
        linestyle="None",
        markersize=7,
        label="Earth",
        zorder=9,
    )
    if np.isfinite(stereo_phi) and np.isfinite(stereo_r):
        axis.plot(
            [np.deg2rad(stereo_phi)],
            [min(float(stereo_r), float(r_axis.max()) - 1.0)],
            marker="s",
            color="black",
            linestyle="None",
            markersize=7,
            label="STEREO-A",
            zorder=9,
        )
    axis.legend(
        loc="upper left",
        bbox_to_anchor=(-0.03, 1.02),
        fontsize=8,
        framealpha=0.82,
    )
    return mesh


def save_single_panel(
    path,
    frame,
    phi_axis,
    r_axis,
    title,
    coverage,
    snapshot_time,
    vmin,
    vmax,
    earth_r,
    stereo_phi,
    stereo_r,
):
    figure = plt.figure(figsize=(7.2, 7.0))
    axis = figure.add_subplot(111, projection="polar")
    mesh = add_polar_frame(
        axis=axis,
        frame=frame,
        phi_axis=phi_axis,
        r_axis=r_axis,
        title=title,
        coverage=coverage,
        vmin=vmin,
        vmax=vmax,
        earth_r=earth_r,
        stereo_phi=stereo_phi,
        stereo_r=stereo_r,
    )
    figure.subplots_adjust(left=0.05, right=0.84, top=0.82, bottom=0.06)
    colorbar_axis = figure.add_axes([0.88, 0.16, 0.025, 0.58])
    colorbar = figure.colorbar(mesh, cax=colorbar_axis)
    colorbar.set_label("Solar-wind speed (km/s)")
    figure.suptitle(
        f"Snapshot: {snapshot_time:%Y-%m-%d %H:%M UTC}",
        fontsize=11,
        color="0.25",
        y=0.97,
    )
    figure.savefig(path, dpi=300, facecolor="white")
    plt.close(figure)


def save_circle_only(
    path,
    frame,
    phi_axis,
    r_axis,
    vmin,
    vmax,
    earth_r,
    stereo_phi,
    stereo_r,
):
    figure = plt.figure(figsize=(6.0, 6.0))
    axis = figure.add_subplot(111, projection="polar")
    cmap = plt.cm.plasma.copy()
    cmap.set_bad("white")
    axis.pcolormesh(
        np.deg2rad(phi_axis),
        r_axis,
        frame.T,
        shading="nearest",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        rasterized=True,
    )
    axis.set_ylim(0.0, float(r_axis.max()) + 1.0)
    axis.set_xticks(np.deg2rad(np.arange(0.0, 360.0, 45.0)))
    axis.set_xticklabels([])
    axis.set_yticks([50, 100, 150, 200])
    axis.set_yticklabels([])
    axis.tick_params(axis="both", length=0)
    axis.grid(color="0.75", linewidth=0.7, alpha=0.7)
    axis.set_facecolor("white")
    axis.scatter([0.0], [0.0], s=95, color="#f6c945", edgecolor="#8c6a00", zorder=8)
    axis.plot(
        [0.0],
        [min(float(earth_r), float(r_axis.max()) - 1.0)],
        marker="o",
        color="black",
        linestyle="None",
        markersize=7,
        zorder=9,
    )
    if np.isfinite(stereo_phi) and np.isfinite(stereo_r):
        axis.plot(
            [np.deg2rad(stereo_phi)],
            [min(float(stereo_r), float(r_axis.max()) - 1.0)],
            marker="s",
            color="black",
            linestyle="None",
            markersize=7,
            zorder=9,
        )
    figure.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
    figure.savefig(path, dpi=300, transparent=True, pad_inches=0)
    plt.close(figure)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    ballistic = load_ballistic_spec()
    empirical = load_empirical_spec()
    rotation = compute_rotation_state(ballistic["phi_step_minutes"])
    phi_axis = np.arange(0.0, 360.0, rotation.phi_step, dtype=np.float64)
    r_axis = build_r_axis(
        r0=ballistic["r0"],
        r_max=ballistic["r_max"],
        r_step=ballistic["r_step"],
    ).astype(np.float64)

    query_start = CANDIDATE_START - INPUT_LEAD
    query_end = CANDIDATE_END + INPUT_TRAIL
    source = load_sw_input_frame(query_start, query_end, source="sql")
    source = source.dropna(subset=["dt", "ch_relative_area"]).sort_values("dt")
    launch_time = (pd.to_datetime(source["dt"]) + pd.Timedelta(minutes=30)).dt.floor("1h")
    source_speed = empirical.v_from_area(
        source["ch_relative_area"].to_numpy(dtype=float),
        t=launch_time,
        parameter_time=pd.to_datetime(source["forecast_dt"]),
    )
    native_hourly = (
        pd.DataFrame({"time": launch_time, "v": source_speed})
        .groupby("time")["v"]
        .mean()
        .sort_index()
    )
    hourly_index = pd.date_range(
        native_hourly.index.min(),
        native_hourly.index.max(),
        freq="1h",
    )
    hourly_speed = native_hourly.reindex(hourly_index).interpolate(method="time")
    assert hourly_speed.notna().all()

    maximum_lookback = pd.Timedelta(days=34.5)
    assert hourly_speed.index.min() <= CANDIDATE_START - maximum_lookback
    assert hourly_speed.index.max() >= CANDIDATE_END + pd.Timedelta(hours=1)

    states = {
        cadence: prepare_cadence_state(
            hourly_speed=hourly_speed,
            cadence_minutes=cadence,
            rotation=rotation,
            phi_axis=phi_axis,
            swept_cells_enabled=SWEPT_CELLS_BY_CADENCE[cadence],
        )
        for cadence in CADENCES_MINUTES
    }

    scan_rows = []
    best_score = -np.inf
    best_time = None
    best_frames = None
    candidate_times = pd.DatetimeIndex([FIXED_SNAPSHOT_TIME])
    for snapshot_time in candidate_times:
        frame_60 = propagate_snapshot(
            states[60], snapshot_time, ballistic["r0"], r_axis
        )
        frame_2 = propagate_snapshot(
            states[2], snapshot_time, ballistic["r0"], r_axis
        )
        coverage_60 = float(np.isfinite(frame_60).mean())
        coverage_2 = float(np.isfinite(frame_2).mean())
        overlap = np.isfinite(frame_60) & np.isfinite(frame_2)
        overlap_mae = float(np.mean(np.abs(frame_2[overlap] - frame_60[overlap])))
        fine_values = frame_2[np.isfinite(frame_2)]
        fine_span = float(np.quantile(fine_values, 0.95) - np.quantile(fine_values, 0.05))
        score = 10000.0 * coverage_2 + 50.0 * (coverage_2 - coverage_60) + 0.1 * fine_span
        scan_rows.append(
            {
                "snapshot_time": snapshot_time,
                "coverage_60min": coverage_60,
                "coverage_2min": coverage_2,
                "coverage_gain": coverage_2 - coverage_60,
                "overlap_mae_km_s": overlap_mae,
                "fine_p95_minus_p05_km_s": fine_span,
                "selection_score": score,
            }
        )
        if score > best_score:
            best_score = score
            best_time = snapshot_time
            best_frames = {60: frame_60.copy(), 2: frame_2.copy()}

    scan_frame = pd.DataFrame(scan_rows).sort_values("snapshot_time")
    scan_path = OUTPUT_DIR / f"{OUTPUT_STEM} Candidate Scan.csv"
    scan_frame.to_csv(scan_path, index=False)

    finite_combined = np.concatenate(
        [best_frames[60][np.isfinite(best_frames[60])], best_frames[2][np.isfinite(best_frames[2])]]
    )
    vmin = float(np.floor(np.quantile(finite_combined, 0.01) / 10.0) * 10.0)
    vmax = float(np.ceil(np.quantile(finite_combined, 0.99) / 10.0) * 10.0)

    stereo_time_axis = pd.date_range(
        best_time - pd.Timedelta(hours=12),
        best_time + pd.Timedelta(hours=12),
        freq="1h",
    )
    stereo_frame = load_stereo_a_frame(
        time_axis=stereo_time_axis,
        time_freq="1h",
    )
    stereo_frame = stereo_frame.interpolate(method="time").ffill().bfill()
    stereo_row = stereo_frame.loc[best_time]
    stereo_phi = float(stereo_row["phi_target"])
    stereo_r = float(stereo_row["r_target"])

    selected_row = scan_frame.loc[scan_frame["snapshot_time"] == best_time].iloc[0]
    combined_path = OUTPUT_DIR / f"{OUTPUT_STEM}.png"
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(14.8, 7.4),
        subplot_kw={"projection": "polar"},
    )
    mesh = add_polar_frame(
        axis=axes[0],
        frame=best_frames[60],
        phi_axis=phi_axis,
        r_axis=r_axis,
        title="60-minute cadence · point deposition",
        coverage=float(selected_row["coverage_60min"]),
        vmin=vmin,
        vmax=vmax,
        earth_r=ballistic["earth_r_target"],
        stereo_phi=stereo_phi,
        stereo_r=stereo_r,
    )
    add_polar_frame(
        axis=axes[1],
        frame=best_frames[2],
        phi_axis=phi_axis,
        r_axis=r_axis,
        title="2-minute cadence · swept deposition",
        coverage=float(selected_row["coverage_2min"]),
        vmin=vmin,
        vmax=vmax,
        earth_r=ballistic["earth_r_target"],
        stereo_phi=stereo_phi,
        stereo_r=stereo_r,
    )
    figure.subplots_adjust(left=0.035, right=0.88, top=0.78, bottom=0.09, wspace=0.13)
    colorbar_axis = figure.add_axes([0.91, 0.16, 0.018, 0.58])
    colorbar = figure.colorbar(mesh, cax=colorbar_axis)
    colorbar.set_label("Solar-wind speed (km/s)")
    figure.suptitle(
        "Ballistic snapshot with cadence-specific deposition\n"
        f"{best_time:%Y-%m-%d %H:%M UTC} · same gap-filled hourly source and spatial grid",
        fontsize=15,
        y=0.97,
    )
    figure.text(
        0.5,
        0.012,
        "60 min: point deposition. 2 min: swept-cell deposition. Identical max-overlap rule; white = no parcel.",
        ha="center",
        fontsize=9,
        color="0.3",
    )
    figure.savefig(combined_path, dpi=300, facecolor="white")
    plt.close(figure)

    panel_paths = {}
    for cadence in CADENCES_MINUTES:
        panel_path = OUTPUT_DIR / f"{OUTPUT_STEM} {cadence}min.png"
        coverage = float(selected_row[f"coverage_{cadence}min"])
        save_single_panel(
            path=panel_path,
            frame=best_frames[cadence],
            phi_axis=phi_axis,
            r_axis=r_axis,
            title=(
                f"{cadence}-minute cadence · "
                f"{'swept' if SWEPT_CELLS_BY_CADENCE[cadence] else 'point'} deposition"
            ),
            coverage=coverage,
            snapshot_time=best_time,
            vmin=vmin,
            vmax=vmax,
            earth_r=ballistic["earth_r_target"],
            stereo_phi=stereo_phi,
            stereo_r=stereo_r,
        )
        panel_paths[cadence] = panel_path

    circle_paths = {
        60: OUTPUT_DIR / "SW Snapshot L 60min No-Swept.png",
        2: OUTPUT_DIR / "SW Snapshot R 2min Swept.png",
    }
    for cadence in CADENCES_MINUTES:
        save_circle_only(
            path=circle_paths[cadence],
            frame=best_frames[cadence],
            phi_axis=phi_axis,
            r_axis=r_axis,
            vmin=vmin,
            vmax=vmax,
            earth_r=ballistic["earth_r_target"],
            stereo_phi=stereo_phi,
            stereo_r=stereo_r,
        )

    frame_path = OUTPUT_DIR / f"{OUTPUT_STEM} Frames.npz"
    np.savez_compressed(
        frame_path,
        frame_60min=best_frames[60],
        frame_2min=best_frames[2],
        phi_axis=phi_axis,
        r_axis=r_axis,
    )
    metadata = {
        "snapshot_time": best_time.isoformat(),
        "query_start": query_start.isoformat(),
        "query_end": query_end.isoformat(),
        "native_hourly_samples": int(len(native_hourly)),
        "filled_hourly_samples": int(len(hourly_speed)),
        "cadence_seed_counts": {
            str(cadence): int(len(state["values"])) for cadence, state in states.items()
        },
        "swept_cells_by_cadence": {
            str(cadence): bool(enabled)
            for cadence, enabled in SWEPT_CELLS_BY_CADENCE.items()
        },
        "coverage_60min": float(selected_row["coverage_60min"]),
        "coverage_2min": float(selected_row["coverage_2min"]),
        "overlap_mae_km_s": float(selected_row["overlap_mae_km_s"]),
        "vmin": vmin,
        "vmax": vmax,
        "combined_plot": str(combined_path),
        "panel_60min": str(panel_paths[60]),
        "panel_2min": str(panel_paths[2]),
        "circle_left_60min": str(circle_paths[60]),
        "circle_right_2min": str(circle_paths[2]),
        "candidate_scan": str(scan_path),
        "frames": str(frame_path),
    }
    metadata_path = OUTPUT_DIR / f"{OUTPUT_STEM} Metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
