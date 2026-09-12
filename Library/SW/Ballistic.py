from dataclasses import dataclass
import time

import numba as nb
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from Library.SW.Constants import SOLAR_RADIUS_KM
from Library.SW.Coords import build_grid_axes


@dataclass(frozen=True)
class PropagationStats:
    """Runtime counters for continuous ballistic propagation."""

    filled: int
    total: int
    prop_seconds: float
    source_points: int
    source_segments: int
    radial_bin_visits: int
    base_plane_cells: int


@dataclass(frozen=True)
class CubeStats:
    """Inferred slow-wind mask and summary of a propagated speed cube."""

    slow_wind_mask: np.ndarray
    speed_range: tuple[float, float]
    filled_cells: int
    slow_cells: int
    non_slow_cells: int
    non_slow_fraction_filled: float


@nb.njit(cache=True)
def _arrival_time(source_time, offset, speed, slope, distance_km):
    return source_time + offset + distance_km / (speed + slope * offset)


@nb.njit(cache=True)
def _inverse_arrival_on_branch(
    target_time,
    source_time,
    speed,
    slope,
    distance_km,
    offset_left,
    offset_right,
    arrival_left,
    arrival_right,
):
    """Invert one monotonic branch of the characteristic arrival curve."""

    left = offset_left
    right = offset_right
    increasing = arrival_right >= arrival_left
    for _ in range(40):
        middle = 0.5 * (left + right)
        arrival_middle = _arrival_time(
            source_time,
            middle,
            speed,
            slope,
            distance_km,
        )
        if increasing:
            if arrival_middle < target_time:
                left = middle
            else:
                right = middle
        elif arrival_middle > target_time:
            left = middle
        else:
            right = middle
    return 0.5 * (left + right)


@nb.njit(cache=True)
def _deposit_monotonic_branch(
    base_plane,
    radius_index,
    base_left_edge_seconds,
    output_step_seconds,
    source_time,
    speed,
    slope,
    distance_km,
    offset_left,
    offset_right,
):
    if offset_right <= offset_left:
        return 0

    arrival_left = _arrival_time(
        source_time,
        offset_left,
        speed,
        slope,
        distance_km,
    )
    arrival_right = _arrival_time(
        source_time,
        offset_right,
        speed,
        slope,
        distance_km,
    )
    arrival_min = min(arrival_left, arrival_right)
    arrival_max = max(arrival_left, arrival_right)
    first_bin = int(
        np.floor((arrival_min - base_left_edge_seconds) / output_step_seconds)
    )
    last_bin = int(
        np.floor((arrival_max - base_left_edge_seconds) / output_step_seconds)
    )
    first_bin = max(0, first_bin)
    last_bin = min(base_plane.shape[1] - 1, last_bin)
    if last_bin < first_bin:
        return 0

    increasing_arrival = arrival_right >= arrival_left
    visits = 0
    for output_index in range(first_bin, last_bin + 1):
        bin_start = base_left_edge_seconds + output_index * output_step_seconds
        bin_end = bin_start + output_step_seconds
        overlap_start = max(bin_start, arrival_min)
        overlap_end = min(bin_end, arrival_max)
        if overlap_end <= overlap_start:
            continue

        if slope > 0.0:
            target_time = overlap_end if increasing_arrival else overlap_start
        elif slope < 0.0:
            target_time = overlap_start if increasing_arrival else overlap_end
        else:
            target_time = overlap_start

        source_offset = _inverse_arrival_on_branch(
            target_time,
            source_time,
            speed,
            slope,
            distance_km,
            offset_left,
            offset_right,
            arrival_left,
            arrival_right,
        )
        candidate_speed = speed + slope * source_offset
        current_speed = base_plane[radius_index, output_index]
        if np.isnan(current_speed) or candidate_speed > current_speed:
            base_plane[radius_index, output_index] = candidate_speed
        visits += 1
    return visits


@nb.njit(cache=True)
def _propagate_radius(
    base_plane,
    radius_index,
    radius,
    launch_radius,
    source_times,
    source_speeds,
    base_left_edge_seconds,
    output_step_seconds,
    maximum_source_gap_seconds,
):
    distance_km = (radius - launch_radius) * SOLAR_RADIUS_KM
    visits = 0

    for source_index in range(len(source_times)):
        speed = source_speeds[source_index]
        if not np.isfinite(speed):
            continue
        arrival = source_times[source_index] + distance_km / speed
        output_index = int(
            np.floor((arrival - base_left_edge_seconds) / output_step_seconds)
        )
        if 0 <= output_index < base_plane.shape[1]:
            current_speed = base_plane[radius_index, output_index]
            if np.isnan(current_speed) or speed > current_speed:
                base_plane[radius_index, output_index] = speed
            visits += 1

    for source_index in range(len(source_times) - 1):
        source_time = source_times[source_index]
        next_source_time = source_times[source_index + 1]
        duration = next_source_time - source_time
        speed = source_speeds[source_index]
        next_speed = source_speeds[source_index + 1]
        if (
            duration <= 0.0
            or duration > maximum_source_gap_seconds
            or not np.isfinite(speed)
            or not np.isfinite(next_speed)
        ):
            continue

        slope = (next_speed - speed) / duration
        critical_offset = -1.0
        if slope > 0.0 and distance_km > 0.0:
            critical_speed = np.sqrt(distance_km * slope)
            if speed < critical_speed < next_speed:
                critical_offset = (critical_speed - speed) / slope

        if 0.0 < critical_offset < duration:
            visits += _deposit_monotonic_branch(
                base_plane,
                radius_index,
                base_left_edge_seconds,
                output_step_seconds,
                source_time,
                speed,
                slope,
                distance_km,
                0.0,
                critical_offset,
            )
            visits += _deposit_monotonic_branch(
                base_plane,
                radius_index,
                base_left_edge_seconds,
                output_step_seconds,
                source_time,
                speed,
                slope,
                distance_km,
                critical_offset,
                duration,
            )
        else:
            visits += _deposit_monotonic_branch(
                base_plane,
                radius_index,
                base_left_edge_seconds,
                output_step_seconds,
                source_time,
                speed,
                slope,
                distance_km,
                0.0,
                duration,
            )

    return visits


def propagate_continuous_boundary(
    df_v_run,
    time_axis,
    phi_axis,
    r_axis,
    rotation_state,
    r0,
    output_step_minutes,
    maximum_source_gap_hours,
    show_progress=True,
):
    """Map continuous boundary segments into bins centred on ``time_axis``."""

    time_axis = pd.DatetimeIndex(time_axis)
    phi_axis = np.asarray(phi_axis, dtype=np.float64)
    r_axis = np.asarray(r_axis, dtype=np.float64)
    assert len(time_axis) > 0
    assert len(phi_axis) > 0
    assert len(r_axis) > 0
    assert time_axis.is_monotonic_increasing and time_axis.is_unique
    assert np.all(r_axis >= float(r0))

    output_step_seconds = float(output_step_minutes) * 60.0
    assert output_step_seconds > 0.0
    expected_time = time_axis[0] + pd.to_timedelta(
        np.arange(len(time_axis)) * output_step_seconds,
        unit="s",
    )
    assert time_axis.equals(pd.DatetimeIndex(expected_time))

    source = df_v_run[["v"]].copy().sort_index()
    source.index = pd.DatetimeIndex(source.index)
    assert source.index.is_unique
    source_speeds = pd.to_numeric(source["v"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    finite_speeds = source_speeds[np.isfinite(source_speeds)]
    assert len(finite_speeds) > 0
    assert np.all(finite_speeds > 0.0)
    source_times = (
        (source.index - time_axis[0]) / pd.Timedelta(seconds=1)
    ).to_numpy(dtype=np.float64)
    assert np.all(np.diff(source_times) > 0.0)

    phi_delay_seconds = phi_axis / float(rotation_state.omega)
    phi_delay_steps_float = phi_delay_seconds / output_step_seconds
    phi_delay_steps = np.rint(phi_delay_steps_float).astype(np.int64)
    assert np.allclose(phi_delay_steps_float, phi_delay_steps, atol=1e-7), (
        "Longitude delays must align with the output time grid"
    )
    assert np.all(phi_delay_steps >= 0)
    maximum_phi_delay_steps = int(phi_delay_steps.max())
    base_time_bins = len(time_axis) + maximum_phi_delay_steps
    base_left_edge_seconds = (
        -maximum_phi_delay_steps * output_step_seconds
        - 0.5 * output_step_seconds
    )
    base_plane = np.full(
        (len(r_axis), base_time_bins),
        np.nan,
        dtype=np.float32,
    )

    maximum_source_gap_seconds = float(maximum_source_gap_hours) * 3600.0
    assert maximum_source_gap_seconds > 0.0
    source_segment_mask = (
        np.diff(source_times) <= maximum_source_gap_seconds
    ) & np.isfinite(source_speeds[:-1]) & np.isfinite(source_speeds[1:])

    prop_start = time.perf_counter()
    iterator = range(len(r_axis))
    if show_progress:
        iterator = tqdm(iterator, desc="Ballistic shells", unit="shell")
    radial_bin_visits = 0
    for radius_index in iterator:
        radial_bin_visits += _propagate_radius(
            base_plane=base_plane,
            radius_index=radius_index,
            radius=float(r_axis[radius_index]),
            launch_radius=float(r0),
            source_times=source_times,
            source_speeds=source_speeds,
            base_left_edge_seconds=base_left_edge_seconds,
            output_step_seconds=output_step_seconds,
            maximum_source_gap_seconds=maximum_source_gap_seconds,
        )

    V_grid = np.full(
        (len(time_axis), len(phi_axis), len(r_axis)),
        np.nan,
        dtype=np.float32,
    )
    for phi_index, delay_steps in enumerate(phi_delay_steps):
        base_index = maximum_phi_delay_steps - int(delay_steps)
        V_grid[:, phi_index, :] = base_plane[
            :,
            base_index : base_index + len(time_axis),
        ].T

    prop_seconds = time.perf_counter() - prop_start
    filled = int(np.count_nonzero(np.isfinite(V_grid)))
    return V_grid, PropagationStats(
        filled=filled,
        total=int(V_grid.size),
        prop_seconds=prop_seconds,
        source_points=int(np.count_nonzero(np.isfinite(source_speeds))),
        source_segments=int(np.count_nonzero(source_segment_mask)),
        radial_bin_visits=int(radial_bin_visits),
        base_plane_cells=int(base_plane.size),
    )


def propagate_ballistic(
    df_v_run,
    sim_start,
    sim_end,
    output_step_minutes,
    rotation_state,
    r0,
    r_max,
    r_step,
    maximum_source_gap_hours,
    phi_values=None,
    show_progress=True,
):
    grid = build_grid_axes(
        sim_start=sim_start,
        sim_end=sim_end,
        output_step_minutes=output_step_minutes,
        phi_step=rotation_state.phi_step,
        r0=r0,
        r_max=r_max,
        r_step=r_step,
        phi_values=phi_values,
    )
    V_grid, stats = propagate_continuous_boundary(
        df_v_run=df_v_run,
        time_axis=grid.time_axis,
        phi_axis=grid.phi_axis,
        r_axis=grid.r_axis,
        rotation_state=rotation_state,
        r0=r0,
        output_step_minutes=output_step_minutes,
        maximum_source_gap_hours=maximum_source_gap_hours,
        show_progress=show_progress,
    )
    return grid, V_grid, stats


def cube_stats(
    speed_cube,
    slow_sw_speed,
):
    """Summarize without changing the cube.

    The mask uses output-time speed equality, not launch-source provenance.
    """
    slow_sw_values = np.asarray(slow_sw_speed, dtype=float)
    if slow_sw_values.ndim == 0:
        slow_sw_values = np.full(speed_cube.shape[0], float(slow_sw_values))
    assert len(slow_sw_values) == speed_cube.shape[0]

    slow_wind_mask = np.isclose(
        speed_cube,
        slow_sw_values[:, None, None],
    )

    if np.isfinite(speed_cube).any():
        speed_range = (float(np.nanmin(speed_cube)), float(np.nanmax(speed_cube)))
    else:
        speed_range = (float("nan"), float("nan"))
    filled_cells = int(np.count_nonzero(np.isfinite(speed_cube)))
    slow_cells = int(np.count_nonzero(slow_wind_mask))
    non_slow_cells = int(filled_cells - slow_cells)
    non_slow_fraction_filled = (
        float(non_slow_cells / filled_cells) if filled_cells else 0.0
    )
    return CubeStats(
        slow_wind_mask=slow_wind_mask,
        speed_range=speed_range,
        filled_cells=filled_cells,
        slow_cells=slow_cells,
        non_slow_cells=non_slow_cells,
        non_slow_fraction_filled=non_slow_fraction_filled,
    )
