from dataclasses import dataclass
import time

import numba as nb
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from Library.SW.Coords import (
    build_grid_axes,
    build_packet_geometry,
    build_transport_state,
)


@dataclass(frozen=True)
class AccumulatorState:
    """Dense accumulation arrays."""

    V_accum_max: np.ndarray  # (n_t, n_p, n_r) float32, NaN-initialized
    cr_flat: np.ndarray  # 1D int32, -1-initialized; tracks last-touching CR per cell


@dataclass(frozen=True)
class PropagationStats:
    """Basic propagation runtime counters."""

    filled: int
    total: int
    prop_seconds: float
    seeds_processed: int


@dataclass(frozen=True)
class PostProcessingState:
    """Max-only post-processing artifacts."""

    V_grid: np.ndarray
    max_slow_sw_pred_mask: np.ndarray
    max_vlims_raw: tuple[float, float]


def init_accumulators(n_t, n_p, n_r):
    V_accum_max = np.full((n_t, n_p, n_r), np.nan, dtype=np.float32)
    cr_flat = np.full(n_t * n_p * n_r, -1, dtype=np.int32)
    return AccumulatorState(V_accum_max=V_accum_max, cr_flat=cr_flat)


def prepare_seed_inputs(
    df_v_run,
    cr_steps,
    horizon_steps,
    time_freq,
    t0_ref,
    time_step_hours,
    r_kernel_scale,
    r0,
):
    seed_vals = df_v_run["v"].to_numpy(dtype=np.float32)

    v_prev = np.empty_like(seed_vals)
    v_next = np.empty_like(seed_vals)
    v_prev[0] = seed_vals[0]
    v_prev[1:] = seed_vals[:-1]
    v_next[-1] = seed_vals[-1]
    v_next[:-1] = seed_vals[1:]

    seed_index_floor = pd.DatetimeIndex(df_v_run.index).floor(time_freq)
    delta_hours = (seed_index_floor - t0_ref) / pd.Timedelta(hours=float(time_step_hours))
    seed_t_idx = np.asarray(delta_hours, dtype=np.int64).astype(np.int32)
    seed_cr_idx_arr = (seed_t_idx // int(cr_steps)).astype(np.int32)

    seed_r_idx = np.rint(
        seed_vals[:, None].astype(np.float64) * r_kernel_scale[None, :]
    ).astype(np.int16)

    return (
        seed_vals,
        v_prev,
        v_next,
        seed_t_idx,
        seed_cr_idx_arr,
        seed_r_idx,
    )


@nb.njit(cache=True)
def deposit_run_cr(
    t_seed,
    seed_cr_idx,
    k_start,
    k_end,
    lo,
    hi,
    h_step_idx,
    packet_off,
    packet_p,
    packet_alpha,
    n_t,
    n_p,
    n_r,
    v_left,
    dv,
    max_flat,
    cr_flat,
):
    for kk in range(k_start, k_end):
        base_step = t_seed + h_step_idx[kk]
        for q in range(packet_off.shape[0]):
            tv = base_step + packet_off[q]
            if tv < 0 or tv >= n_t:
                continue

            pv = packet_p[q]
            vv = v_left + packet_alpha[q] * dv
            flat_base = (tv * n_p + pv) * n_r

            for rr in range(lo, hi + 1):
                idx = flat_base + rr
                if cr_flat[idx] != seed_cr_idx:
                    cr_flat[idx] = seed_cr_idx
                    max_flat[idx] = vv
                elif vv > max_flat[idx]:
                    max_flat[idx] = vv


@nb.njit(cache=True)
def propagate_swept_cr_batch(
    v_prev_b,
    seed_vals_b,
    v_next_b,
    seed_t_b,
    seed_cr_b,
    seed_r_b,
    h_step_idx,
    packet_off,
    packet_p,
    packet_alpha,
    n_t,
    n_p,
    n_r,
    max_flat,
    cr_flat,
):
    n_seed = seed_vals_b.shape[0]
    h_len = seed_r_b.shape[1]

    for i in range(n_seed):
        v_left = 0.5 * (v_prev_b[i] + seed_vals_b[i])
        v_right = 0.5 * (seed_vals_b[i] + v_next_b[i])
        dv = v_right - v_left
        t_seed = int(seed_t_b[i])
        seed_cr_idx = int(seed_cr_b[i])

        r00 = int(seed_r_b[i, 0])
        if 0 <= r00 < n_r:
            deposit_run_cr(
                t_seed,
                seed_cr_idx,
                0,
                1,
                r00,
                r00,
                h_step_idx,
                packet_off,
                packet_p,
                packet_alpha,
                n_t,
                n_p,
                n_r,
                v_left,
                dv,
                max_flat,
                cr_flat,
            )

        run_active = False
        run_lo = 0
        run_hi = 0
        run_start = 1

        for k in range(1, h_len):
            ra = int(seed_r_b[i, k - 1])
            rb = int(seed_r_b[i, k])
            lo = ra if ra <= rb else rb
            hi = rb if ra <= rb else ra

            if lo > (n_r - 1):
                if run_active:
                    deposit_run_cr(
                        t_seed,
                        seed_cr_idx,
                        run_start,
                        k,
                        run_lo,
                        run_hi,
                        h_step_idx,
                        packet_off,
                        packet_p,
                        packet_alpha,
                        n_t,
                        n_p,
                        n_r,
                        v_left,
                        dv,
                        max_flat,
                        cr_flat,
                    )
                    run_active = False
                break

            if hi < 0:
                if run_active:
                    deposit_run_cr(
                        t_seed,
                        seed_cr_idx,
                        run_start,
                        k,
                        run_lo,
                        run_hi,
                        h_step_idx,
                        packet_off,
                        packet_p,
                        packet_alpha,
                        n_t,
                        n_p,
                        n_r,
                        v_left,
                        dv,
                        max_flat,
                        cr_flat,
                    )
                    run_active = False
                continue

            if lo < 0:
                lo = 0
            if hi > (n_r - 1):
                hi = n_r - 1

            if not run_active:
                run_active = True
                run_lo = lo
                run_hi = hi
                run_start = k
            elif lo == run_lo and hi == run_hi:
                continue
            else:
                deposit_run_cr(
                    t_seed,
                    seed_cr_idx,
                    run_start,
                    k,
                    run_lo,
                    run_hi,
                    h_step_idx,
                    packet_off,
                    packet_p,
                    packet_alpha,
                    n_t,
                    n_p,
                    n_r,
                    v_left,
                    dv,
                    max_flat,
                    cr_flat,
                )
                run_lo = lo
                run_hi = hi
                run_start = k

        if run_active:
            deposit_run_cr(
                t_seed,
                seed_cr_idx,
                run_start,
                h_len,
                run_lo,
                run_hi,
                h_step_idx,
                packet_off,
                packet_p,
                packet_alpha,
                n_t,
                n_p,
                n_r,
                v_left,
                dv,
                max_flat,
                cr_flat,
            )


def run_bulk_propagation(
    seed_vals,
    v_prev,
    v_next,
    seed_t_idx,
    seed_cr_idx_arr,
    seed_r_idx,
    h_step_idx,
    packet_off,
    packet_p,
    packet_alpha,
    n_t,
    n_p,
    n_r,
    V_accum_max,
    cr_flat,
    max_seed_batch,
    show_progress=True,
):
    max_flat = V_accum_max.ravel()
    prop_start = time.perf_counter()
    n_seed = len(seed_vals)
    n_batches = (n_seed + int(max_seed_batch) - 1) // int(max_seed_batch)
    iterator = range(0, n_seed, int(max_seed_batch))
    if show_progress:
        iterator = tqdm(iterator, total=n_batches, desc="2D propagate", unit="batch")

    for b0 in iterator:
        b1 = min(b0 + int(max_seed_batch), n_seed)
        propagate_swept_cr_batch(
            v_prev[b0:b1],
            seed_vals[b0:b1],
            v_next[b0:b1],
            seed_t_idx[b0:b1],
            seed_cr_idx_arr[b0:b1],
            seed_r_idx[b0:b1],
            h_step_idx,
            packet_off,
            packet_p,
            packet_alpha,
            n_t,
            n_p,
            n_r,
            max_flat,
            cr_flat,
        )

    prop_seconds = time.perf_counter() - prop_start
    filled = int(np.count_nonzero(~np.isnan(max_flat)))
    total = int(max_flat.size)
    return PropagationStats(
        filled=filled,
        total=total,
        prop_seconds=prop_seconds,
        seeds_processed=n_seed,
    )


def postprocess_max_field(
    V_accum_max,
    slow_sw_speed,
    post_chunk_t,
    show_progress=True,
):
    V_grid_max_raw = V_accum_max.copy()
    chunk_t = max(1, int(post_chunk_t))
    slow_sw_values = np.asarray(slow_sw_speed, dtype=float)
    if slow_sw_values.ndim == 0:
        slow_sw_values = np.full(V_grid_max_raw.shape[0], float(slow_sw_values))
    assert len(slow_sw_values) == V_grid_max_raw.shape[0]

    slow_sw_pred_max = np.zeros_like(V_grid_max_raw, dtype=bool)

    iterator = range(0, V_grid_max_raw.shape[0], chunk_t)
    if show_progress:
        iterator = tqdm(iterator, desc="Post-processing", unit="chunk")

    for t0 in iterator:
        t1 = min(t0 + chunk_t, V_grid_max_raw.shape[0])
        slab = V_grid_max_raw[t0:t1]
        slow_ref = slow_sw_values[t0:t1, None, None]
        slow_sw_pred_max[t0:t1] = ~np.isnan(slab) & np.isclose(slab, slow_ref)

    if np.isfinite(V_grid_max_raw).any():
        vlims = (
            float(np.nanmin(V_grid_max_raw)),
            float(np.nanmax(V_grid_max_raw)),
        )
    else:
        vlims = (float("nan"), float("nan"))

    return PostProcessingState(
        V_grid=V_grid_max_raw,
        max_slow_sw_pred_mask=slow_sw_pred_max,
        max_vlims_raw=vlims,
    )


def propagate_phi_targets(
    df_v_run,
    sim_start,
    sim_end,
    time_freq,
    rotation_state,
    r0,
    r_max,
    dense_memory_budget_gb,
    memory_guard_enabled,
    horizon_hours,
    time_step_hours,
    field_half_width_h,
    r_solar_km,
    max_seed_batch,
    phi_targets,
    show_progress=True,
):
    """Run the standard propagation pipeline on a sparse set of target phis.

    Returns (grid, transport, accumulators, stats).
    """

    grid = build_grid_axes(
        sim_start=sim_start,
        sim_end=sim_end,
        time_freq=time_freq,
        phi_step=rotation_state.phi_step,
        r0=r0,
        r_max=r_max,
        dense_memory_budget_gb=dense_memory_budget_gb,
        memory_guard_enabled=memory_guard_enabled,
        phi_values=phi_targets,
    )
    transport = build_transport_state(
        time_axis=grid.time_axis,
        phi_axis=grid.phi_axis,
        rotation_state=rotation_state,
        horizon_hours=horizon_hours,
        time_step_hours=time_step_hours,
        field_half_width_h=field_half_width_h,
        r_solar_km=r_solar_km,
    )
    packet_p, packet_off, packet_alpha = build_packet_geometry(
        phi_delay_steps=transport.phi_delay_steps,
        field_half_width_steps=transport.field_half_width_steps,
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
        r0=r0,
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
        V_accum_max=accumulators.V_accum_max,
        cr_flat=accumulators.cr_flat,
        max_seed_batch=max_seed_batch,
        show_progress=show_progress,
    )
    return grid, transport, accumulators, stats
