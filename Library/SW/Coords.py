from dataclasses import dataclass

import numpy as np
import pandas as pd

from Library.SW.Constants import CARRINGTON_ROTATION_DAYS


@dataclass(frozen=True)
class RotationState:
    """Rotation constants derived from Carrington cadence settings."""

    cr_time: float
    omega: float
    phi_step: float


@dataclass(frozen=True)
class GridState:
    """Dense propagation axes."""

    time_axis: pd.DatetimeIndex
    phi_axis: np.ndarray
    r_axis: np.ndarray


def compute_rotation_state(phi_step_minutes, cr_days=CARRINGTON_ROTATION_DAYS):
    cr_time = float(cr_days) * 24.0 * 3600.0
    omega = 360.0 / cr_time
    phi_step = (float(phi_step_minutes) / 60.0) * 3600.0 * omega
    return RotationState(
        cr_time=cr_time,
        omega=omega,
        phi_step=phi_step,
    )


def build_r_axis(r0, r_max, r_step, required_r_values=(215.0,)):
    r0 = float(r0)
    r_max = float(r_max)
    r_step = float(r_step)
    assert r_step > 0.0, "r_step must be positive"
    assert r_max >= r0, "r_max must be >= r0"

    r_axis = np.arange(r0, r_max + (0.5 * r_step), r_step, dtype=np.float64)
    r_axis = r_axis[r_axis <= r_max]
    required = [float(value) for value in required_r_values]
    for value in required:
        assert r0 <= value <= r_max, f"required r shell {value:g} must be within [r0, r_max]"
    r_axis = np.concatenate([r_axis, np.asarray([r_max, *required], dtype=np.float64)])
    return np.unique(np.round(r_axis, decimals=8)).astype(np.float32)


def build_centered_time_axis(interval_start, interval_end, output_step_minutes):
    """Return globally aligned bin centres contained in [start, end)."""

    interval_start = pd.Timestamp(interval_start)
    interval_end = pd.Timestamp(interval_end)
    bin_width = pd.Timedelta(minutes=float(output_step_minutes))
    assert bin_width > pd.Timedelta(0)
    assert interval_start < interval_end
    return pd.date_range(
        interval_start.ceil(bin_width),
        interval_end,
        freq=bin_width,
        inclusive="left",
    )


def build_grid_axes(
    sim_start,
    sim_end,
    output_step_minutes,
    phi_step,
    r0,
    r_max,
    r_step,
    phi_values=None,
):
    time_axis = build_centered_time_axis(
        sim_start,
        sim_end,
        output_step_minutes,
    )
    if phi_values is None:
        phi_axis = np.arange(0.0, 360.0, phi_step, dtype=float)
    else:
        phi_axis = np.mod(np.asarray(phi_values, dtype=float).reshape(-1), 360.0)
        assert phi_axis.size > 0, "phi_values must contain at least one target phi"
    r_axis = build_r_axis(r0=r0, r_max=r_max, r_step=r_step)
    return GridState(
        time_axis=time_axis,
        phi_axis=phi_axis,
        r_axis=r_axis,
    )
