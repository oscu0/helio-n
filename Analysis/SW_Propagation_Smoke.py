#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/helio_n_matplotlib")
os.environ.setdefault("SUNPY_CONFIGDIR", "/tmp/helio_n_sunpy")

import matplotlib.pyplot as plt
import numba as nb
import numpy as np
import pandas as pd
from sunpy.coordinates.sun import (
    carrington_rotation_number,
    carrington_rotation_time,
)

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from Library.SW.Config import load_ballistic_spec
from Library.SW.Ballistic import propagate_continuous_boundary
from Library.SW.Constants import SOLAR_RADIUS_KM
from Library.SW.Coords import build_r_axis, compute_rotation_state


DEFAULT_REFERENCE_PARQUET = (
    ROOT_DIR
    / "Outputs"
    / "SW"
    / "SW Reproduction Series 20180101_0000-20190101_0000.parquet"
)
DEFAULT_OUTPUT_DIR = ROOT_DIR / "Outputs" / "SW" / "Smoke"
DEFAULT_SNAPSHOT_TIME = pd.Timestamp("2018-04-22 00:00:00")
LEGACY_REFERENCE_STEP_MINUTES = 2
LEGACY_HORIZON_HOURS = 168.0


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description=(
            "Run SW propagation contract tests, produce legacy and analytic "
            "comparison frames, and record broad-comparison metrics."
        )
    )
    parser.add_argument(
        "--reference-parquet",
        type=Path,
        default=DEFAULT_REFERENCE_PARQUET,
        help="Frozen reproduction parquet containing model input and legacy output.",
    )
    parser.add_argument(
        "--snapshot-time",
        default=str(DEFAULT_SNAPSHOT_TIME),
        help="Characteristic time used for the polar comparison.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for frames and metrics.",
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip the fast synthetic contract suite.",
    )
    parser.add_argument(
        "--no-open",
        action="store_true",
        help="Write outputs without opening the candidate and reference frames.",
    )
    return parser.parse_args(argv[1:])


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


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
def legacy_snapshot_kernel(
    seed_values,
    seed_previous,
    seed_next,
    seed_time_indices,
    seed_cr_indices,
    snapshot_time_index,
    horizon_steps,
    time_step_seconds,
    phi_delay_offsets,
    phi_delay_alpha,
    launch_radius,
    radius_axis,
):
    """Frozen two-minute swept/CR-reset reference for one polar frame."""

    frame = np.full((len(phi_delay_offsets), len(radius_axis)), np.nan, dtype=np.float32)
    frame_cr = np.full((len(phi_delay_offsets), len(radius_axis)), -1, dtype=np.int32)

    for seed_idx in range(len(seed_values)):
        speed = seed_values[seed_idx]
        if not np.isfinite(speed):
            continue
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
                launch_radius
                + speed
                * (float(propagation_step) * time_step_seconds)
                / SOLAR_RADIUS_KM
            )
            radius_idx = nearest_axis_index(radius_axis, radius)
            if propagation_step == 0:
                previous_radius_idx = radius_idx
            else:
                previous_radius = (
                    launch_radius
                    + speed
                    * (float(propagation_step - 1) * time_step_seconds)
                    / SOLAR_RADIUS_KM
                )
                previous_radius_idx = nearest_axis_index(
                    radius_axis, previous_radius
                )

            radial_start = min(previous_radius_idx, radius_idx)
            radial_end = max(previous_radius_idx, radius_idx)
            if radial_start >= len(radius_axis) or radial_end < 0:
                continue
            radial_start = max(0, radial_start)
            radial_end = min(len(radius_axis) - 1, radial_end)

            value = value_left + phi_delay_alpha[phi_idx] * value_delta
            for radius_idx in range(radial_start, radial_end + 1):
                if frame_cr[phi_idx, radius_idx] != seed_cr_idx:
                    frame_cr[phi_idx, radius_idx] = seed_cr_idx
                    frame[phi_idx, radius_idx] = value
                elif value > frame[phi_idx, radius_idx]:
                    frame[phi_idx, radius_idx] = value

    return frame

def run_contract_suite():
    suite = unittest.defaultTestLoader.discover(
        start_dir=str(ROOT_DIR / "Tests"),
        pattern="test_*.py",
        top_level_dir=str(ROOT_DIR),
    )
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    assert result.wasSuccessful(), "SW contract tests failed"
    return {
        "tests_run": int(result.testsRun),
        "failures": int(len(result.failures)),
        "errors": int(len(result.errors)),
    }


def paired_metrics(left, right):
    left = np.asarray(left, dtype=float)
    right = np.asarray(right, dtype=float)
    valid = np.isfinite(left) & np.isfinite(right)
    if np.count_nonzero(valid) < 2:
        return {
            "paired_samples": int(np.count_nonzero(valid)),
            "correlation": None,
            "mae_km_s": None,
            "p95_absolute_error_km_s": None,
        }
    difference = np.abs(left[valid] - right[valid])
    return {
        "paired_samples": int(np.count_nonzero(valid)),
        "correlation": float(np.corrcoef(left[valid], right[valid])[0, 1]),
        "mae_km_s": float(np.mean(difference)),
        "p95_absolute_error_km_s": float(np.quantile(difference, 0.95)),
    }


def best_lag(reference, candidate, step_minutes, maximum_lag_hours=12.0):
    reference = np.asarray(reference, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    maximum_steps = int(round(maximum_lag_hours * 60.0 / step_minutes))
    best_correlation = -np.inf
    best_lag_steps = 0

    for lag_steps in range(-maximum_steps, maximum_steps + 1):
        if lag_steps < 0:
            reference_slice = reference[-lag_steps:]
            candidate_slice = candidate[:lag_steps]
        elif lag_steps > 0:
            reference_slice = reference[:-lag_steps]
            candidate_slice = candidate[lag_steps:]
        else:
            reference_slice = reference
            candidate_slice = candidate

        valid = np.isfinite(reference_slice) & np.isfinite(candidate_slice)
        if np.count_nonzero(valid) < 3:
            continue
        correlation = float(
            np.corrcoef(reference_slice[valid], candidate_slice[valid])[0, 1]
        )
        if np.isfinite(correlation) and correlation > best_correlation:
            best_correlation = correlation
            best_lag_steps = lag_steps

    assert np.isfinite(best_correlation), "No finite lagged comparison is available"
    return {
        "lag_minutes": int(best_lag_steps * step_minutes),
        "correlation": float(best_correlation),
    }


def frame_metrics(reference, candidate):
    reference_valid = np.isfinite(reference)
    candidate_valid = np.isfinite(candidate)
    overlap = reference_valid & candidate_valid
    union = reference_valid | candidate_valid
    metrics = paired_metrics(reference[overlap], candidate[overlap])
    metrics.update(
        {
            "reference_coverage": float(reference_valid.mean()),
            "candidate_coverage": float(candidate_valid.mean()),
            "finite_mask_jaccard": float(
                np.count_nonzero(overlap) / np.count_nonzero(union)
            )
            if np.count_nonzero(union)
            else 1.0,
        }
    )
    return metrics


def save_polar_frame(
    path,
    frame,
    phi_axis,
    radius_axis,
    snapshot_time,
    title,
    vmin,
    vmax,
    earth_radius,
):
    figure = plt.figure(figsize=(7.2, 7.0))
    axis = figure.add_subplot(111, projection="polar")
    color_map = plt.cm.plasma.copy()
    color_map.set_bad("white")
    mesh = axis.pcolormesh(
        np.deg2rad(phi_axis),
        radius_axis,
        frame.T,
        shading="nearest",
        cmap=color_map,
        vmin=vmin,
        vmax=vmax,
        rasterized=True,
    )
    axis.set_ylim(0.0, float(radius_axis.max()) + 1.0)
    axis.set_yticks([50, 100, 150, 200])
    axis.set_yticklabels([])
    axis.grid(color="0.75", linewidth=0.7, alpha=0.7)
    axis.scatter(
        [0.0],
        [0.0],
        s=95,
        color="#f6c945",
        edgecolor="#8c6a00",
        zorder=8,
    )
    axis.plot(
        [0.0],
        [earth_radius],
        marker="o",
        color="black",
        linestyle="None",
        markersize=7,
        zorder=9,
    )
    coverage = float(np.isfinite(frame).mean())
    axis.set_title(title, fontsize=14, pad=20)
    axis.text(
        0.5,
        -0.08,
        f"{snapshot_time:%Y-%m-%d %H:%M UTC} · finite cells {coverage:.1%}",
        transform=axis.transAxes,
        ha="center",
        fontsize=10,
        color="0.25",
    )
    figure.subplots_adjust(left=0.04, right=0.84, top=0.88, bottom=0.10)
    colorbar_axis = figure.add_axes([0.88, 0.18, 0.025, 0.58])
    colorbar = figure.colorbar(mesh, cax=colorbar_axis)
    colorbar.set_label("Solar-wind speed (km/s)")
    figure.savefig(path, dpi=250, facecolor="white")
    plt.close(figure)


def carrington_bounds(timestamp):
    rotation_number = int(np.floor(carrington_rotation_number(timestamp)))
    rotation_start = pd.Timestamp(carrington_rotation_time(rotation_number).datetime)
    rotation_end = pd.Timestamp(carrington_rotation_time(rotation_number + 1).datetime)
    return rotation_number, rotation_start, rotation_end


def main(argv):
    args = parse_args(argv)
    snapshot_time = pd.Timestamp(args.snapshot_time)
    assert args.reference_parquet.exists(), (
        f"Reference parquet does not exist: {args.reference_parquet}"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    test_result = None if args.skip_tests else run_contract_suite()
    ballistic = load_ballistic_spec()
    reference_step_minutes = LEGACY_REFERENCE_STEP_MINUTES
    candidate_step_minutes = int(ballistic["output_step_minutes"])
    assert candidate_step_minutes > 0
    reference_step = pd.Timedelta(minutes=reference_step_minutes)
    candidate_step = pd.Timedelta(minutes=candidate_step_minutes)

    reproduction = pd.read_parquet(args.reference_parquet)
    launch_speed = reproduction[
        ("input", "model_input", "v_empirical")
    ].dropna()
    source_boundary = launch_speed.resample("1h").first().to_frame(name="v")
    legacy_earth_all = reproduction[
        ("satellite", "ace_earth", "v_predict_raw")
    ]
    ace_earth_all = reproduction[
        ("satellite", "ace_earth", "v_real")
    ]

    assert launch_speed.index.is_monotonic_increasing
    assert launch_speed.index.is_unique
    assert (launch_speed.index.to_series().diff().dropna() == reference_step).all()
    assert snapshot_time in launch_speed.index

    rotation = compute_rotation_state(ballistic["phi_step_minutes"])
    phi_axis = np.arange(0.0, 360.0, rotation.phi_step, dtype=np.float64)
    radius_axis = build_r_axis(
        r0=ballistic["r0"],
        r_max=ballistic["r_max"],
        r_step=ballistic["r_step"],
    ).astype(np.float64)

    values_all = launch_speed.to_numpy(dtype=np.float32)
    previous_all = np.empty_like(values_all)
    next_all = np.empty_like(values_all)
    previous_all[0] = values_all[0]
    previous_all[1:] = values_all[:-1]
    next_all[-1] = values_all[-1]
    next_all[:-1] = values_all[1:]

    input_start = launch_speed.index[0]
    step_seconds = float(reference_step / pd.Timedelta(seconds=1))
    seed_time_indices_all = (
        (launch_speed.index - input_start) / reference_step
    ).to_numpy(dtype=np.int64)
    snapshot_time_index = int((snapshot_time - input_start) / reference_step)
    phi_delay_seconds = (phi_axis / rotation.omega).astype(np.float64)
    phi_delay_steps = phi_delay_seconds / step_seconds
    phi_delay_offsets = np.floor(phi_delay_steps).astype(np.int32)
    phi_delay_alpha = np.clip(
        phi_delay_steps - phi_delay_offsets,
        0.0,
        1.0,
    ).astype(np.float32)
    horizon_steps = int(
        round(LEGACY_HORIZON_HOURS * 3600.0 / step_seconds)
    )
    cr_steps = int(round(rotation.cr_time / step_seconds))

    earliest_seed_idx = (
        snapshot_time_index
        - horizon_steps
        - int(phi_delay_offsets.max())
        - 2
    )
    relevant = (
        (seed_time_indices_all >= earliest_seed_idx)
        & (seed_time_indices_all <= snapshot_time_index)
    )
    seed_values = values_all[relevant]
    seed_previous = previous_all[relevant]
    seed_next = next_all[relevant]
    seed_time_indices = seed_time_indices_all[relevant]
    seed_cr_indices = (seed_time_indices // cr_steps).astype(np.int32)
    reference_frame = legacy_snapshot_kernel(
        seed_values=seed_values,
        seed_previous=seed_previous,
        seed_next=seed_next,
        seed_time_indices=seed_time_indices,
        seed_cr_indices=seed_cr_indices,
        snapshot_time_index=snapshot_time_index,
        horizon_steps=horizon_steps,
        time_step_seconds=step_seconds,
        phi_delay_offsets=phi_delay_offsets,
        phi_delay_alpha=phi_delay_alpha,
        launch_radius=float(ballistic["r0"]),
        radius_axis=radius_axis,
    )
    candidate_cube, candidate_frame_stats = propagate_continuous_boundary(
        df_v_run=source_boundary,
        time_axis=pd.DatetimeIndex([snapshot_time]),
        phi_axis=phi_axis,
        r_axis=radius_axis,
        rotation_state=rotation,
        r0=ballistic["r0"],
        output_step_minutes=candidate_step_minutes,
        maximum_source_gap_hours=ballistic["maximum_input_gap_hours"],
        show_progress=False,
    )
    candidate_frame = candidate_cube[0]

    assert reference_frame.shape == candidate_frame.shape
    candidate_finite = candidate_frame[np.isfinite(candidate_frame)]
    assert len(candidate_finite) > 0
    assert float(candidate_finite.min()) >= float(np.nanmin(seed_values))
    assert float(candidate_finite.max()) <= float(np.nanmax(seed_values))

    rotation_number, rotation_start, rotation_end = carrington_bounds(snapshot_time)
    output_index = pd.date_range(
        rotation_start.ceil(candidate_step),
        rotation_end,
        freq=candidate_step,
        inclusive="left",
    )
    candidate_earth_cube, candidate_earth_stats = propagate_continuous_boundary(
        df_v_run=source_boundary,
        time_axis=output_index,
        phi_axis=[0.0],
        r_axis=[ballistic["earth_r_target"]],
        rotation_state=rotation,
        r0=ballistic["r0"],
        output_step_minutes=candidate_step_minutes,
        maximum_source_gap_hours=ballistic["maximum_input_gap_hours"],
        show_progress=False,
    )
    candidate_earth = candidate_earth_cube[:, 0, 0]
    legacy_earth = legacy_earth_all.reindex(output_index).to_numpy(dtype=float)
    ace_earth = ace_earth_all.reindex(output_index).to_numpy(dtype=float)

    finite_frames = np.concatenate(
        [
            reference_frame[np.isfinite(reference_frame)],
            candidate_frame[np.isfinite(candidate_frame)],
        ]
    )
    vmin = float(np.floor(np.quantile(finite_frames, 0.01) / 10.0) * 10.0)
    vmax = float(np.ceil(np.quantile(finite_frames, 0.99) / 10.0) * 10.0)
    stamp = f"{snapshot_time:%Y%m%d_%H%M}"
    reference_path = args.output_dir / f"SW Propagation Reference {stamp}.png"
    candidate_suffix = f" {candidate_step_minutes}min"
    candidate_path = (
        args.output_dir / f"SW Propagation Candidate{candidate_suffix} {stamp}.png"
    )
    frames_path = (
        args.output_dir / f"SW Propagation Frames{candidate_suffix} {stamp}.npz"
    )
    metrics_path = (
        args.output_dir / f"SW Propagation Metrics{candidate_suffix} {stamp}.json"
    )

    save_polar_frame(
        path=reference_path,
        frame=reference_frame,
        phi_axis=phi_axis,
        radius_axis=radius_axis,
        snapshot_time=snapshot_time,
        title=(
            f"Reference · legacy {reference_step_minutes}-minute swept propagation"
        ),
        vmin=vmin,
        vmax=vmax,
        earth_radius=float(ballistic["earth_r_target"]),
    )
    save_polar_frame(
        path=candidate_path,
        frame=candidate_frame,
        phi_axis=phi_axis,
        radius_axis=radius_axis,
        snapshot_time=snapshot_time,
        title=f"Candidate · {candidate_step_minutes}-minute centred bins",
        vmin=vmin,
        vmax=vmax,
        earth_radius=float(ballistic["earth_r_target"]),
    )
    np.savez_compressed(
        frames_path,
        reference_frame=reference_frame,
        candidate_frame=candidate_frame,
        phi_axis=phi_axis,
        radius_axis=radius_axis,
    )
    with np.load(frames_path) as saved:
        np.testing.assert_array_equal(saved["reference_frame"], reference_frame)
        np.testing.assert_array_equal(saved["candidate_frame"], candidate_frame)

    frame_comparison = frame_metrics(reference_frame, candidate_frame)
    earth_comparison = paired_metrics(legacy_earth, candidate_earth)
    lag_comparison = best_lag(
        reference=legacy_earth,
        candidate=candidate_earth,
        step_minutes=candidate_step_minutes,
    )
    legacy_ace = paired_metrics(legacy_earth, ace_earth)
    candidate_ace = paired_metrics(candidate_earth, ace_earth)

    review_flags = []
    coverage_loss = (
        frame_comparison["reference_coverage"]
        - frame_comparison["candidate_coverage"]
    )
    if coverage_loss > 0.02:
        review_flags.append(
            "candidate frame coverage is lower by more than two percentage points"
        )
    if frame_comparison["finite_mask_jaccard"] < 0.65:
        review_flags.append("frame finite-mask Jaccard is below 0.65")
    if earth_comparison["correlation"] is None or earth_comparison["correlation"] < 0.8:
        review_flags.append("legacy/candidate Earth correlation is below 0.8")
    if abs(lag_comparison["lag_minutes"]) > 240:
        review_flags.append("best legacy/candidate Earth lag exceeds four hours")
    if (
        legacy_ace["correlation"] is not None
        and candidate_ace["correlation"] is not None
        and candidate_ace["correlation"] < legacy_ace["correlation"] - 0.1
    ):
        review_flags.append("candidate ACE correlation is lower by more than 0.1")
    if (
        legacy_ace["mae_km_s"] is not None
        and candidate_ace["mae_km_s"] is not None
        and candidate_ace["mae_km_s"] > 1.2 * legacy_ace["mae_km_s"]
    ):
        review_flags.append("candidate ACE MAE is more than 20% larger")

    metrics = {
        "snapshot_time": snapshot_time.isoformat(),
        "carrington_rotation": rotation_number,
        "comparison_start": output_index[0].isoformat(),
        "comparison_end_exclusive": rotation_end.isoformat(),
        "reference_parquet": str(args.reference_parquet),
        "reference_sha256": sha256(args.reference_parquet),
        "reference_step_minutes": reference_step_minutes,
        "candidate_step_minutes": candidate_step_minutes,
        "reference_samples_total": int(len(values_all)),
        "reference_samples_for_frame": int(len(seed_values)),
        "candidate_source_points": int(candidate_frame_stats.source_points),
        "candidate_source_segments": int(candidate_frame_stats.source_segments),
        "candidate_frame_seconds": float(candidate_frame_stats.prop_seconds),
        "candidate_earth_seconds": float(candidate_earth_stats.prop_seconds),
        "contract_tests": test_result,
        "frame": frame_comparison,
        "earth_legacy_vs_candidate": earth_comparison,
        "earth_best_lag": lag_comparison,
        "earth_legacy_vs_ace": legacy_ace,
        "earth_candidate_vs_ace": candidate_ace,
        "review_flags": review_flags,
        "reference_frame": str(reference_path),
        "candidate_frame": str(candidate_path),
        "frames": str(frames_path),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))

    if not args.no_open:
        subprocess.run(
            ["open", str(candidate_path), str(reference_path)],
            check=True,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
