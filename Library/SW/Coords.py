from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RotationState:
    """Rotation constants derived from Carrington cadence settings."""

    cr_time: float
    omega: float
    phi_step: float


@dataclass(frozen=True)
class GridState:
    """Dense propagation axes and memory estimate."""

    time_axis: pd.DatetimeIndex
    phi_axis: np.ndarray
    r_axis: np.ndarray
    n_cells: int
    est_runtime_gb: float


@dataclass(frozen=True)
class TransportState:
    """Precomputed time/radius transport helpers."""

    t0_ref: pd.Timestamp
    horizon_steps: int
    h_step_idx: np.ndarray
    r_kernel_scale: np.ndarray
    cr_steps: int
    phi_delay_steps: np.ndarray
    field_half_width_steps: float


def compute_rotation_state(cr_days, phi_step_minutes):
    cr_time = float(cr_days) * 24.0 * 3600.0
    omega = 360.0 / cr_time
    phi_step = (float(phi_step_minutes) / 60.0) * 3600.0 * omega
    return RotationState(
        cr_time=cr_time,
        omega=omega,
        phi_step=phi_step,
    )


def build_grid_axes(
    sim_start,
    sim_end,
    time_freq,
    phi_step,
    r0,
    r_max,
    dense_memory_budget_gb,
    memory_guard_enabled,
    phi_values=None,
):
    time_axis = pd.date_range(sim_start, sim_end, freq=time_freq)
    if phi_values is None:
        phi_axis = np.arange(0.0, 360.0, phi_step, dtype=float)
    else:
        phi_axis = np.mod(np.asarray(phi_values, dtype=float).reshape(-1), 360.0)
        assert phi_axis.size > 0, "phi_values must contain at least one target phi"
    r_axis = np.arange(int(r0), int(r_max) + 1, 1, dtype=np.int16)
    n_cells = int(len(time_axis) * len(phi_axis) * len(r_axis))
    est_runtime_gb = (n_cells * 4 * 1.6) / 1e9
    if memory_guard_enabled:
        assert est_runtime_gb <= float(dense_memory_budget_gb), (
            f"Estimated memory {est_runtime_gb:.2f} GB exceeds budget "
            f"{float(dense_memory_budget_gb):.2f} GB. Reduce date range, increase "
            "phi_step_minutes, or disable superresolution."
        )
    return GridState(
        time_axis=time_axis,
        phi_axis=phi_axis,
        r_axis=r_axis,
        n_cells=n_cells,
        est_runtime_gb=est_runtime_gb,
    )


def build_transport_state(
    time_axis,
    phi_axis,
    rotation_state,
    horizon_hours,
    time_step_hours,
    field_half_width_h,
    r_solar_km,
):
    t0_ref = time_axis[0]
    phi_delay_h = (phi_axis / rotation_state.omega) / 3600.0
    horizon_steps = int(round(float(horizon_hours) / float(time_step_hours)))
    h_step_idx = np.arange(horizon_steps, dtype=np.int32)
    h_step_hours = h_step_idx.astype(np.float64) * float(time_step_hours)
    r_kernel_scale = (h_step_hours * 3600.0) / float(r_solar_km)
    cr_steps = int(round((rotation_state.cr_time / 3600.0) / float(time_step_hours)))
    phi_delay_steps = phi_delay_h / float(time_step_hours)
    field_half_width_steps = float(field_half_width_h) / float(time_step_hours)
    return TransportState(
        t0_ref=t0_ref,
        horizon_steps=horizon_steps,
        h_step_idx=h_step_idx,
        r_kernel_scale=r_kernel_scale,
        cr_steps=cr_steps,
        phi_delay_steps=phi_delay_steps,
        field_half_width_steps=field_half_width_steps,
    )


def build_packet_geometry(phi_delay_steps, field_half_width_steps):
    packet_width_steps = 2.0 * float(field_half_width_steps)
    packet_p_list = []
    packet_off_list = []
    packet_alpha_list = []

    for phi_idx, center in enumerate(phi_delay_steps):
        left = float(center) - float(field_half_width_steps)
        right = float(center) + float(field_half_width_steps)
        j_lo = int(np.floor(left))
        j_hi = int(np.floor(right))
        for offset in range(j_lo, j_hi + 1):
            sample_center = float(offset) + 0.5
            alpha = (sample_center - left) / packet_width_steps
            alpha = float(np.clip(alpha, 0.0, 1.0))
            packet_p_list.append(phi_idx)
            packet_off_list.append(offset)
            packet_alpha_list.append(alpha)

    return (
        np.asarray(packet_p_list, dtype=np.int32),
        np.asarray(packet_off_list, dtype=np.int32),
        np.asarray(packet_alpha_list, dtype=np.float32),
    )
