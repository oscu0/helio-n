from pathlib import Path

from matplotlib import animation
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from Library.SW.Coords import find_axis_index


def ensure_even_frame_size(fig, dpi):
    """Adjust figure inches so rasterized output lands on even pixel counts."""
    fig_w_px = int(fig.get_figwidth() * int(dpi))
    fig_h_px = int(fig.get_figheight() * int(dpi))
    if fig_w_px % 2 != 0:
        fig_w_px += 1
    if fig_h_px % 2 != 0:
        fig_h_px += 1
    fig.set_size_inches(fig_w_px / int(dpi), fig_h_px / int(dpi), forward=True)
    return fig_w_px, fig_h_px


def build_earth_comparison_frame(
    time_axis,
    phi_axis,
    r_axis,
    grid_raw,
    slow_sw_pred_mask,
    slow_sw_speed,
    df_ace_earth=None,
    df_forecast_earth=None,
    phi_target=0.0,
    r_target=215.0,
    draw_slow_sw=True,
    backfill_empty_with_300=False,
):
    phi_idx = find_axis_index(phi_axis, target=phi_target)
    r_idx = find_axis_index(r_axis, target=r_target)

    model_series_raw = pd.Series(
        grid_raw[:, phi_idx, r_idx],
        index=time_axis,
        name="v_model_earth_raw",
    )
    if backfill_empty_with_300:
        model_series = model_series_raw.fillna(float(slow_sw_speed)).rename(
            "v_model_earth"
        )
    else:
        model_series = model_series_raw.rename("v_model_earth")

    if not draw_slow_sw:
        slow_earth_mask = slow_sw_pred_mask[:, phi_idx, r_idx]
        model_series = model_series.mask(slow_earth_mask)

    earth_frame = pd.DataFrame(
        {"v_model_earth_raw": model_series_raw, "v_model_earth": model_series}
    )

    if df_ace_earth is not None and "v_ace" in df_ace_earth.columns:
        earth_frame = earth_frame.join(df_ace_earth[["v_ace"]], how="outer")
    if (
        df_forecast_earth is not None
        and "forecast_sw_speed" in df_forecast_earth.columns
    ):
        earth_frame = earth_frame.join(
            df_forecast_earth[["forecast_sw_speed"]], how="outer"
        )

    return earth_frame.sort_index()


def plot_polar_snapshot(
    date_str,
    time_axis,
    phi_axis,
    r_axis,
    grid_raw,
    post_vlims_raw,
    slow_sw_pred_mask,
    time_step_hours,
    slow_sw_speed,
    r0,
    draw_slow_sw=True,
    backfill_empty_with_300=False,
):
    current_time = pd.Timestamp(date_str)
    t0_ref = time_axis[0]
    t_idx = int((current_time - t0_ref) / pd.Timedelta(hours=float(time_step_hours)))

    raw_slice_pr = grid_raw[t_idx, :, :]
    if backfill_empty_with_300:
        slice_pr = np.where(np.isnan(raw_slice_pr), float(slow_sw_speed), raw_slice_pr)
        vmin_mode, vmax_mode = float(slow_sw_speed), post_vlims_raw[1]
    else:
        slice_pr = raw_slice_pr
        vmin_mode, vmax_mode = post_vlims_raw

    slice_pred_slow = slow_sw_pred_mask[t_idx, :, :]

    plot_mask = ~np.isnan(slice_pr)
    if not draw_slow_sw:
        plot_mask = plot_mask & (~slice_pred_slow)

    pi, ri = np.where(plot_mask)
    if len(pi) == 0:
        print(f"No data for {date_str} with current display settings")
        return

    phi_rad = np.deg2rad(phi_axis[pi].astype(float))
    radius = r_axis[ri].astype(float)
    colors = slice_pr[pi, ri].astype(float)

    fig = plt.figure(figsize=(7.2, 7.2))
    ax = fig.add_subplot(111, projection="polar")
    fig.subplots_adjust(top=0.84, right=0.86)

    scatter = ax.scatter(
        phi_rad,
        radius,
        c=colors,
        cmap="plasma",
        s=2,
        alpha=0.9,
        vmin=vmin_mode,
        vmax=vmax_mode,
    )
    colorbar = plt.colorbar(scatter, ax=ax, pad=0.14, fraction=0.05)
    colorbar.set_label("v (km/s)")

    ax.set_title(f"phi-R at {date_str}", y=1.14, pad=0)
    ax.text(
        0.5,
        1.06,
        f"draw_slow_sw={draw_slow_sw} | backfill_300={backfill_empty_with_300}",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=9,
        color="dimgray",
    )
    ax.text(
        0.02, 0.96, "new", transform=ax.transAxes, va="top", fontsize=9, color="dimgray"
    )
    ax.text(
        0.02,
        0.04,
        "old",
        transform=ax.transAxes,
        va="bottom",
        fontsize=9,
        color="dimgray",
    )
    ax.set_ylim(0.0, float(np.nanmax(r_axis)) + 1.0)
    ax.set_yticklabels([])
    ax.text(
        0.74,
        0.93,
        f"R0 = {int(r0)} Rs\nRmax = {int(np.nanmax(r_axis))} Rs\nEarth = 215 Rs",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox=dict(
            boxstyle="round,pad=0.28",
            facecolor="white",
            edgecolor="gray",
            alpha=0.85,
        ),
    )

    plt.show()


def export_polar_animation(
    anim_outfile,
    time_axis,
    phi_axis,
    r_axis,
    grid_raw,
    post_vlims_raw,
    slow_sw_pred_mask,
    earth_frame,
    time_step_minutes,
    superresolution_enabled,
    slow_sw_speed,
    r0,
    cr_days,
    draw_slow_sw=True,
    backfill_empty_with_300=False,
    target_speedup_vs_baseline=10.0,
    baseline_fps=12.0,
    anim_fps=30,
    speedup_multiplier=2.0,
    superres_reference_minutes=10.0,
    anim_dpi=100,
    anim_plot_style="mesh",
    show_progress=True,
):
    assert anim_plot_style in {"mesh", "scatter"}

    output_path = Path(anim_outfile)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    base_stride = max(
        1,
        int(
            round(
                float(target_speedup_vs_baseline)
                * float(baseline_fps)
                / float(anim_fps)
            )
        ),
    )
    if superresolution_enabled:
        superres_stride_factor = float(superres_reference_minutes) / float(
            time_step_minutes
        )
    else:
        superres_stride_factor = 1.0
    anim_stride = max(
        1,
        int(round(base_stride * superres_stride_factor * float(speedup_multiplier))),
    )
    achieved_speedup = (anim_stride * float(anim_fps)) / float(baseline_fps)

    if backfill_empty_with_300:
        anim_vmin, anim_vmax = float(slow_sw_speed), post_vlims_raw[1]
    else:
        anim_vmin, anim_vmax = post_vlims_raw

    frame_idx = np.arange(0, len(time_axis), anim_stride, dtype=np.int32)
    window_before_days = float(cr_days - 7.0)
    window_after_days = 7.0

    y_parts = []
    for column in ["v_model_earth", "v_ace", "forecast_sw_speed"]:
        if column in earth_frame.columns:
            values = earth_frame[column].to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            if finite.size > 0:
                y_parts.append(finite)
    if y_parts:
        y_all = np.concatenate(y_parts)
        earth_vmin = float(np.min(y_all))
        earth_vmax = float(np.max(y_all))
    else:
        earth_vmin = 250.0
        earth_vmax = 850.0
    earth_pad = max(10.0, 0.08 * (earth_vmax - earth_vmin + 1e-6))
    earth_ylim = (earth_vmin - earth_pad, earth_vmax + earth_pad)

    fig = plt.figure(figsize=(7.2, 9.0))
    grid_spec = fig.add_gridspec(2, 1, height_ratios=[3.4, 1.6], hspace=0.38)
    ax = fig.add_subplot(grid_spec[0], projection="polar")
    ax_earth = fig.add_subplot(grid_spec[1])
    fig.subplots_adjust(top=0.72, right=0.86)

    cmap = plt.cm.plasma.copy()
    cmap.set_bad((1, 1, 1, 0))

    frame_buf_2d = np.empty((len(r_axis), len(phi_axis)), dtype=np.float32)
    init_t = int(frame_idx[0])
    np.copyto(frame_buf_2d, grid_raw[init_t].T)
    if backfill_empty_with_300:
        np.nan_to_num(frame_buf_2d, copy=False, nan=float(slow_sw_speed))
    if not draw_slow_sw:
        frame_buf_2d[slow_sw_pred_mask[init_t].T] = np.nan

    if anim_plot_style == "mesh":
        theta = np.deg2rad(phi_axis.astype(np.float32))
        radius = r_axis.astype(np.float32)
        artist = ax.pcolormesh(
            theta,
            radius,
            frame_buf_2d,
            cmap=cmap,
            shading="nearest",
            vmin=anim_vmin,
            vmax=anim_vmax,
        )
    else:
        phi_rad_all = np.deg2rad(np.repeat(phi_axis.astype(np.float32), len(r_axis)))
        r_all = np.tile(r_axis.astype(np.float32), len(phi_axis))
        artist = ax.scatter(
            phi_rad_all,
            r_all,
            c=frame_buf_2d.T.reshape(-1),
            cmap=cmap,
            s=2,
            alpha=0.9,
            linewidths=0,
            vmin=anim_vmin,
            vmax=anim_vmax,
        )

    ax.set_ylim(0.0, float(np.nanmax(r_axis)) + 1.0)
    ax.set_yticklabels([])
    ax.text(
        0.74,
        0.93,
        f"R0 = {int(r0)} Rs\nRmax = {int(np.nanmax(r_axis))} Rs\nEarth = 215 Rs",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8,
        bbox=dict(
            boxstyle="round,pad=0.28",
            facecolor="white",
            edgecolor="gray",
            alpha=0.85,
        ),
    )

    colorbar = plt.colorbar(artist, pad=0.14, fraction=0.05)
    colorbar.set_label("v (km/s)")

    title = ax.set_title("", y=1.14, pad=0)
    ax.text(
        0.5,
        1.06,
        f"draw_slow_sw={draw_slow_sw} | backfill_300={backfill_empty_with_300} | style={anim_plot_style}",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=9,
        color="dimgray",
    )
    ax.text(
        0.02, 0.96, "new", transform=ax.transAxes, va="top", fontsize=9, color="dimgray"
    )
    ax.text(
        0.02,
        0.04,
        "old",
        transform=ax.transAxes,
        va="bottom",
        fontsize=9,
        color="dimgray",
    )

    model_line = None
    ace_line = None
    forecast_line = None
    if "v_model_earth" in earth_frame.columns:
        (model_line,) = ax_earth.plot(
            [], [], color="tab:orange", linewidth=1.4, label="v model @ (R=215, phi=0)"
        )
    if "v_ace" in earth_frame.columns:
        (ace_line,) = ax_earth.plot(
            [], [], color="tab:blue", linewidth=1.1, alpha=0.9, label="v_ace"
        )
    if "forecast_sw_speed" in earth_frame.columns:
        (forecast_line,) = ax_earth.plot(
            [], [], color="tab:green", linewidth=1.1, alpha=0.9, label="v_sw_predict"
        )
    ax_earth.axhline(
        float(slow_sw_speed),
        color="red",
        linestyle="--",
        linewidth=1.0,
        alpha=0.8,
        label=f"slow SW = {float(slow_sw_speed):.0f} km/s",
    )
    line_t = ax_earth.axvline(
        time_axis[init_t], color="black", linestyle="--", linewidth=1.0, alpha=0.7
    )
    ax_earth.set_ylabel("v (km/s)")
    ax_earth.set_title(
        f"Earth Running Window: t-{window_before_days:.0f}d to t+{window_after_days:.0f}d",
        fontsize=10,
    )
    ax_earth.grid(alpha=0.25)
    ax_earth.set_ylim(*earth_ylim)
    ax_earth.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax_earth.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
    ax_earth.legend(loc="upper left", fontsize=8)

    ensure_even_frame_size(fig, dpi=int(anim_dpi))

    writer = animation.FFMpegWriter(
        fps=int(anim_fps),
        codec="libx264",
        extra_args=[
            "-preset",
            "ultrafast",
            "-crf",
            "23",
            "-pix_fmt",
            "yuv420p",
            "-threads",
            "0",
        ],
    )

    iterator = frame_idx
    if show_progress:
        iterator = tqdm(frame_idx, desc="Export MP4", unit="frame")

    with writer.saving(fig, str(output_path), dpi=int(anim_dpi)):
        for t_idx in iterator:
            ti = int(t_idx)
            t_cur = time_axis[ti]
            title.set_text(f"phi-R at {t_cur}")

            np.copyto(frame_buf_2d, grid_raw[ti].T)
            if backfill_empty_with_300:
                np.nan_to_num(frame_buf_2d, copy=False, nan=float(slow_sw_speed))
            if not draw_slow_sw:
                frame_buf_2d[slow_sw_pred_mask[ti].T] = np.nan

            if anim_plot_style == "mesh":
                artist.set_array(frame_buf_2d.ravel())
            else:
                artist.set_array(frame_buf_2d.T.reshape(-1))

            window_start = t_cur - pd.Timedelta(days=window_before_days)
            window_end = t_cur + pd.Timedelta(days=window_after_days)
            earth_window = earth_frame.loc[window_start:window_end]
            if model_line is not None:
                model_line.set_data(
                    earth_window.index, earth_window["v_model_earth"].values
                )
            if ace_line is not None:
                ace_line.set_data(earth_window.index, earth_window["v_ace"].values)
            if forecast_line is not None:
                forecast_line.set_data(
                    earth_window.index, earth_window["forecast_sw_speed"].values
                )

            line_t.set_xdata([t_cur, t_cur])
            ax_earth.set_xlim(window_start, window_end)
            writer.grab_frame()

    plt.close(fig)
    return {
        "frames": float(len(frame_idx)),
        "stride": float(anim_stride),
        "fps": float(anim_fps),
        "achieved_speedup": float(achieved_speedup),
    }
