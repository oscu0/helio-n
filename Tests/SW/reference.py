"""Small independent reference calculations for SW contract tests."""

import numpy as np
import pandas as pd


def ballistic_arrival_time(
    launch_time,
    speed_km_s,
    phi_degrees,
    omega_degrees_s,
    radius_rsun,
    launch_radius_rsun,
    solar_radius_km,
):
    phi_delay_seconds = float(phi_degrees) / float(omega_degrees_s)
    radial_delay_seconds = (
        (float(radius_rsun) - float(launch_radius_rsun))
        * float(solar_radius_km)
        / float(speed_km_s)
    )
    return pd.Timestamp(launch_time) + pd.Timedelta(
        seconds=phi_delay_seconds + radial_delay_seconds
    )


def centered_bin_index(arrival_time, first_bin_center, bin_width):
    arrival_ns = pd.Timestamp(arrival_time).value
    center_ns = pd.Timestamp(first_bin_center).value
    bin_ns = pd.Timedelta(bin_width).value
    left_edge_ns = center_ns - bin_ns // 2
    return int((arrival_ns - left_edge_ns) // bin_ns)


def propagate_analytic_reference(
    launch_times,
    speeds,
    phi_axis,
    radius_axis,
    output_first_center,
    output_bins,
    bin_width,
    omega_degrees_s,
    launch_radius_rsun,
    solar_radius_km,
):
    """Clear, deliberately small max-only ballistic reference."""

    launch_times = pd.DatetimeIndex(launch_times)
    speeds = np.asarray(speeds, dtype=float)
    phi_axis = np.asarray(phi_axis, dtype=float)
    radius_axis = np.asarray(radius_axis, dtype=float)
    cube = np.full(
        (int(output_bins), len(phi_axis), len(radius_axis)),
        np.nan,
        dtype=np.float32,
    )

    for launch_time, speed in zip(launch_times, speeds):
        if not np.isfinite(speed):
            continue
        for phi_idx, phi in enumerate(phi_axis):
            for radius_idx, radius in enumerate(radius_axis):
                arrival_time = ballistic_arrival_time(
                    launch_time=launch_time,
                    speed_km_s=speed,
                    phi_degrees=phi,
                    omega_degrees_s=omega_degrees_s,
                    radius_rsun=radius,
                    launch_radius_rsun=launch_radius_rsun,
                    solar_radius_km=solar_radius_km,
                )
                time_idx = centered_bin_index(
                    arrival_time=arrival_time,
                    first_bin_center=output_first_center,
                    bin_width=bin_width,
                )
                if time_idx < 0 or time_idx >= output_bins:
                    continue
                current = cube[time_idx, phi_idx, radius_idx]
                if np.isnan(current) or speed > current:
                    cube[time_idx, phi_idx, radius_idx] = speed

    return cube


def propagate_continuous_reference(
    source,
    time_axis,
    phi_axis,
    radius_axis,
    omega_degrees_s,
    launch_radius_rsun,
    solar_radius_km,
    output_step="2min",
    maximum_source_gap="12h",
    source_sample_step="250ms",
):
    """Dense-sampling oracle for small continuous-characteristic tests."""

    source = pd.Series(source, copy=True, dtype=float).sort_index()
    time_axis = pd.DatetimeIndex(time_axis)
    phi_axis = np.asarray(phi_axis, dtype=float)
    radius_axis = np.asarray(radius_axis, dtype=float)
    output_step_seconds = pd.Timedelta(output_step).total_seconds()
    maximum_gap = pd.Timedelta(maximum_source_gap)
    sample_step_seconds = pd.Timedelta(source_sample_step).total_seconds()
    output_first_center = time_axis[0]
    cube = np.full(
        (len(time_axis), len(phi_axis), len(radius_axis)),
        np.nan,
        dtype=np.float32,
    )

    launch_seconds = []
    launch_speeds = []
    for source_time, speed in source.items():
        if np.isfinite(speed):
            launch_seconds.append(
                (source_time - output_first_center).total_seconds()
            )
            launch_speeds.append(float(speed))

    source_items = list(source.items())
    for (left_time, left_speed), (right_time, right_speed) in zip(
        source_items[:-1], source_items[1:]
    ):
        duration = right_time - left_time
        if (
            duration <= pd.Timedelta(0)
            or duration > maximum_gap
            or not np.isfinite(left_speed)
            or not np.isfinite(right_speed)
        ):
            continue
        duration_seconds = duration.total_seconds()
        offsets = np.arange(0.0, duration_seconds, sample_step_seconds)
        fractions = offsets / duration_seconds
        speeds = float(left_speed) + fractions * (
            float(right_speed) - float(left_speed)
        )
        launch_seconds.extend(
            (left_time - output_first_center).total_seconds() + offsets
        )
        launch_speeds.extend(speeds)

    launch_seconds = np.asarray(launch_seconds, dtype=float)
    launch_speeds = np.asarray(launch_speeds, dtype=float)
    for phi_index, phi in enumerate(phi_axis):
        phi_delay = float(phi) / float(omega_degrees_s)
        for radius_index, radius in enumerate(radius_axis):
            radial_delay = (
                (float(radius) - float(launch_radius_rsun))
                * float(solar_radius_km)
                / launch_speeds
            )
            arrival_seconds = launch_seconds + phi_delay + radial_delay
            output_indices = np.floor(
                (arrival_seconds + 0.5 * output_step_seconds)
                / output_step_seconds
            ).astype(int)
            valid = (output_indices >= 0) & (output_indices < len(time_axis))
            for output_index, speed in zip(
                output_indices[valid], launch_speeds[valid]
            ):
                current = cube[output_index, phi_index, radius_index]
                if np.isnan(current) or speed > current:
                    cube[output_index, phi_index, radius_index] = speed

    return cube
