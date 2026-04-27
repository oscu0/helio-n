from pathlib import Path

from matplotlib import animation
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

PREDICT_COLUMN = "v_predict"
REAL_COLUMN = "v_real"
SWX_COLUMN = "v_swx"
MICROFORECAST_COLUMN = "v_1cr_ago"
SAT_MARKER_SHAPES = ("o", "s", "^", "D", "P", "X", "v", "<", ">", "h", "*")
PREDICT_LINE_COLOR = "tab:red"
REAL_LINE_COLOR = "tab:green"
SWX_LINE_COLOR = "tab:orange"
MICROFORECAST_LINE_COLOR = "tab:blue"
SLOW_SW_LINE_COLOR = "tab:gray"


def find_axis_index(axis_values, target):
    return int(np.argmin(np.abs(axis_values - float(target))))


def find_phi_index(phi_axis, phi_target):
    delta = np.abs((np.asarray(phi_axis, dtype=float) - float(phi_target) + 180.0) % 360.0 - 180.0)
    return int(np.argmin(delta))


def resolve_slow_sw_speed(time_axis, slow_sw_speed):
    values = np.asarray(slow_sw_speed, dtype=float)
    if values.ndim == 0:
        values = np.full(len(time_axis), float(values))
    assert len(values) == len(time_axis)
    return pd.Series(values, index=pd.DatetimeIndex(time_axis), name="slow_sw_speed")


def build_satellite_comparison_frame(
    time_axis,
    phi_axis,
    r_axis,
    grid_raw,
    slow_sw_pred_mask,
    df_sat=None,
    df_swx=None,
    phi_target=0.0,
    r_target=215.0,
    cr_days=27.0,
    draw_slow_sw=True,
):
    target_frame = pd.DataFrame(index=pd.DatetimeIndex(time_axis))
    if df_sat is not None and {"phi_target", "r_target"}.issubset(df_sat.columns):
        target_frame = target_frame.join(df_sat[["phi_target", "r_target"]], how="left")
        target_frame[["phi_target", "r_target"]] = (
            target_frame[["phi_target", "r_target"]]
            .interpolate(method="time")
            .ffill()
            .bfill()
        )
    else:
        target_frame["phi_target"] = float(phi_target)
        target_frame["r_target"] = float(r_target)

    v_predict_raw = np.full(len(time_axis), np.nan, dtype=float)
    v_predict = np.full(len(time_axis), np.nan, dtype=float)
    for time_idx, (phi_value, r_value) in enumerate(
        zip(
            target_frame["phi_target"].to_numpy(dtype=float),
            target_frame["r_target"].to_numpy(dtype=float),
        )
    ):
        if not (np.isfinite(phi_value) and np.isfinite(r_value)):
            continue

        phi_idx = find_phi_index(phi_axis, phi_value)
        r_idx = find_axis_index(r_axis, target=r_value)
        raw_value = float(grid_raw[time_idx, phi_idx, r_idx])
        v_predict_raw[time_idx] = raw_value

        model_value = raw_value
        if not draw_slow_sw and bool(slow_sw_pred_mask[time_idx, phi_idx, r_idx]):
            model_value = np.nan
        v_predict[time_idx] = model_value

    comparison_frame = pd.DataFrame(
        {
            "v_predict_raw": pd.Series(v_predict_raw, index=time_axis),
            "v_predict": pd.Series(v_predict, index=time_axis),
            "phi_target": target_frame["phi_target"],
            "r_target": target_frame["r_target"],
        }
    )
    if df_sat is not None and "v" in df_sat.columns:
        comparison_frame = comparison_frame.join(
            df_sat[["v"]].rename(columns={"v": "v_real"}),
            how="outer",
        )
        comparison_frame["v_real"] = comparison_frame["v_real"].interpolate(method="time")
        microforecast_index = pd.DatetimeIndex(df_sat.index) + pd.Timedelta(days=float(cr_days))
        df_microforecast = pd.DataFrame(
            {"v_1cr_ago": pd.to_numeric(df_sat["v"], errors="coerce").to_numpy()},
            index=microforecast_index,
        ).sort_index()
        comparison_frame = comparison_frame.join(df_microforecast, how="outer")
        comparison_frame["v_1cr_ago"] = comparison_frame["v_1cr_ago"].interpolate(
            method="time"
        )
    if df_sat is not None and "lat_hgs" in df_sat.columns:
        comparison_frame = comparison_frame.join(df_sat[["lat_hgs"]], how="left")
    if df_sat is not None and "lat_hge" in df_sat.columns:
        comparison_frame = comparison_frame.join(df_sat[["lat_hge"]], how="left")
    if df_swx is not None and "v_swx" in df_swx.columns:
        comparison_frame = comparison_frame.join(df_swx[["v_swx"]], how="outer")
        comparison_frame["v_swx"] = comparison_frame["v_swx"].interpolate(method="time")

    if df_sat is not None:
        comparison_frame.attrs.update(df_sat.attrs)
    elif df_swx is not None:
        comparison_frame.attrs.update(df_swx.attrs)
    return comparison_frame.sort_index()


def _build_satellite_plot_items(comparison_frames):
    sat_items = []
    for sat_idx, (sat_name, frame) in enumerate(comparison_frames.items()):
        sat_items.append(
            {
                "sat_name": sat_name,
                "frame": frame,
                "label": frame.attrs.get("label", sat_name),
                "predict_col": PREDICT_COLUMN if PREDICT_COLUMN in frame.columns else None,
                "real_col": REAL_COLUMN if REAL_COLUMN in frame.columns else None,
                "swx_col": SWX_COLUMN if SWX_COLUMN in frame.columns else None,
                "microforecast_col": MICROFORECAST_COLUMN if MICROFORECAST_COLUMN in frame.columns else None,
                "marker": SAT_MARKER_SHAPES[sat_idx % len(SAT_MARKER_SHAPES)],
            }
        )
    return sat_items


def _build_polar_frame(grid_raw, slow_sw_pred_mask, t_idx, draw_slow_sw, frame_buffer):
    np.copyto(frame_buffer, grid_raw[int(t_idx)].T)
    if not draw_slow_sw:
        frame_buffer[slow_sw_pred_mask[int(t_idx)].T] = np.nan
    return frame_buffer


def _format_title(current_time):
    return f"Heliosphere at {pd.Timestamp(current_time).strftime('%Y-%m-%d %H:%M:%S')}"


def _resolve_panel_ylim(sat_items):
    y_parts = []
    for sat_item in sat_items:
        for column in (
            sat_item["predict_col"],
            sat_item["real_col"],
            sat_item["swx_col"],
            sat_item["microforecast_col"],
        ):
            if column is None:
                continue
            values = sat_item["frame"][column].to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            if finite.size > 0:
                y_parts.append(finite)
    if y_parts:
        y_all = np.concatenate(y_parts)
        y_min = float(np.min(y_all))
        y_max = float(np.max(y_all))
    else:
        y_min, y_max = 250.0, 850.0
    y_pad = max(10.0, 0.08 * (y_max - y_min + 1e-6))
    return y_min - y_pad, y_max + y_pad


def _update_satellite_markers(sat_items, current_time, r_axis):
    current_time = pd.Timestamp(current_time)
    r_max = float(np.nanmax(r_axis))
    r_min = float(np.nanmin(r_axis))
    for sat_item in sat_items:
        sat_frame = sat_item["frame"]
        if current_time not in sat_frame.index:
            sat_item["polar_marker"].set_data([], [])
            continue
        phi_value = float(sat_frame.loc[current_time, "phi_target"])
        r_value = float(np.clip(float(sat_frame.loc[current_time, "r_target"]), r_min, r_max - 0.75))
        sat_item["polar_marker"].set_data([np.deg2rad(phi_value)], [r_value])


def _update_panel_windows(sat_items, slow_sw_series, current_time, window_before_days, window_after_days):
    current_time = pd.Timestamp(current_time)
    window_start = current_time - pd.Timedelta(days=float(window_before_days))
    window_end = current_time + pd.Timedelta(days=float(window_after_days))
    for sat_item in sat_items:
        compare_window = sat_item["frame"].loc[window_start:window_end]
        if sat_item["predict_line"] is not None:
            sat_item["predict_line"].set_data(
                compare_window.index,
                compare_window[sat_item["predict_col"]].values,
            )
        if sat_item["real_line"] is not None:
            _real = compare_window[sat_item["real_col"]].dropna()
            sat_item["real_line"].set_data(_real.index, _real.values)
        if sat_item["swx_line"] is not None:
            _swx = compare_window[sat_item["swx_col"]].dropna()
            sat_item["swx_line"].set_data(_swx.index, _swx.values)
        if sat_item["microforecast_line"] is not None:
            _microforecast = compare_window[sat_item["microforecast_col"]].dropna()
            sat_item["microforecast_line"].set_data(
                _microforecast.index,
                _microforecast.values,
            )
        slow_sw_window = slow_sw_series.loc[window_start:window_end]
        sat_item["slow_sw_line"].set_data(slow_sw_window.index, slow_sw_window.values)
        sat_item["time_marker"].set_xdata([current_time, current_time])
        sat_item["axis"].set_xlim(window_start, window_end)


def _initialize_panels(sat_axes, sat_items, slow_sw_series, current_time, window_before_days, window_after_days, y_lim):
    for sat_idx, sat_item in enumerate(sat_items):
        sat_axis = sat_axes[sat_idx]
        sat_item["axis"] = sat_axis
        sat_item["predict_line"] = None
        sat_item["real_line"] = None
        sat_item["swx_line"] = None
        sat_item["microforecast_line"] = None
        sat_item["slow_sw_line"] = None
        sat_item["time_marker"] = sat_axis.axvline(
            current_time, color="black", linestyle="--", linewidth=1.0, alpha=0.7
        )
        if sat_item["predict_col"] is not None:
            (sat_item["predict_line"],) = sat_axis.plot([], [], color=PREDICT_LINE_COLOR, linewidth=1.7, label="v_predict")
        if sat_item["real_col"] is not None:
            (sat_item["real_line"],) = sat_axis.plot([], [], color=REAL_LINE_COLOR, linewidth=1.4, alpha=0.95, linestyle="-", label="v_obs")
        if sat_item["swx_col"] is not None:
            (sat_item["swx_line"],) = sat_axis.plot([], [], color=SWX_LINE_COLOR, linewidth=1.35, alpha=0.95, linestyle="-", label="v_swx")
        if sat_item["microforecast_col"] is not None:
            (sat_item["microforecast_line"],) = sat_axis.plot([], [], color=MICROFORECAST_LINE_COLOR, linewidth=1.25, alpha=0.95, linestyle="--", label="1CR microforecast")
        (sat_item["slow_sw_line"],) = sat_axis.plot(
            slow_sw_series.index, slow_sw_series.values,
            color=SLOW_SW_LINE_COLOR, linestyle="-.", linewidth=1.0, alpha=0.75, label="v_min",
        )
        sat_axis.set_ylabel("v (km/s)")
        sat_axis.set_title(
            f"   {sat_item['label']}: t-{float(window_before_days):.0f}d to t+{float(window_after_days):.0f}d",
            fontsize=10, loc="left",
        )
        sat_axis.plot(
            [0.012], [1.035], transform=sat_axis.transAxes,
            linestyle="None", marker=sat_item["marker"],
            color="black", markerfacecolor="black", markeredgecolor="black",
            markersize=7, clip_on=False,
        )
        sat_axis.grid(alpha=0.25)
        sat_axis.set_ylim(*y_lim)
        sat_axis.legend(loc="upper left", fontsize=8)
        if sat_idx < len(sat_items) - 1:
            sat_axis.tick_params(labelbottom=False)
        sat_axis.tick_params(axis="x", labelsize=8)

    if sat_items:
        last_sat_axis = sat_axes[len(sat_items) - 1]
        last_sat_axis.xaxis.set_major_locator(mdates.AutoDateLocator())
        last_sat_axis.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
        _update_panel_windows(
            sat_items=sat_items,
            slow_sw_series=slow_sw_series,
            current_time=current_time,
            window_before_days=window_before_days,
            window_after_days=window_after_days,
        )


def _build_figure_axes(n_sat_panels, layout):
    """Create figure and (polar_ax, sat_axes) for a layout."""
    panel_height = 1.55
    if layout == "vertical":
        fig_height = max(8.6, 6.4 + panel_height * n_sat_panels)
        fig = plt.figure(figsize=(8.4, fig_height))
        grid_spec = fig.add_gridspec(
            n_sat_panels + 1, 1,
            height_ratios=[4.2] + [1.0] * n_sat_panels,
            hspace=0.5,
        )
        polar_ax = fig.add_subplot(grid_spec[0, 0], projection="polar")
        sat_axes = []
        shared_axis = None
        for sat_idx in range(n_sat_panels):
            sat_axis = fig.add_subplot(grid_spec[sat_idx + 1, 0], sharex=shared_axis)
            sat_axes.append(sat_axis)
            if shared_axis is None:
                shared_axis = sat_axis
        fig.subplots_adjust(top=0.94, right=0.86, left=0.12, bottom=0.06)
    elif layout == "horizontal":
        fig_height = max(6.8, panel_height * n_sat_panels + 0.9)
        fig = plt.figure(figsize=(14.2, fig_height))
        grid_spec = fig.add_gridspec(
            n_sat_panels, 2,
            width_ratios=[1.35, 1.0],
            height_ratios=[1.0] * n_sat_panels,
            wspace=0.34, hspace=0.48,
        )
        polar_ax = fig.add_subplot(grid_spec[:, 0], projection="polar")
        sat_axes = []
        shared_axis = None
        for sat_idx in range(n_sat_panels):
            sat_axis = fig.add_subplot(grid_spec[sat_idx, 1], sharex=shared_axis)
            sat_axes.append(sat_axis)
            if shared_axis is None:
                shared_axis = sat_axis
        fig.subplots_adjust(top=0.88, right=0.94, left=0.045, bottom=0.08)
    else:
        raise ValueError(f"Unknown layout: {layout!r}")
    return fig, polar_ax, sat_axes


def _build_polar_view(
    fig,
    polar_ax,
    sat_axes,
    time_axis,
    phi_axis,
    r_axis,
    grid_raw,
    post_vlims_raw,
    slow_sw_pred_mask,
    slow_sw_speed,
    comparison_frames,
    draw_slow_sw,
    cr_days,
    init_t_idx,
):
    """Build all visual content (polar mesh, sat markers, panels) for the figure.

    This is the SINGLE source of truth for plot/animation content. Any
    visual element that should appear identically in both static plots
    and animations must be added here, never in the entry-point wrappers.
    """
    slow_sw_series = resolve_slow_sw_speed(time_axis, slow_sw_speed)
    sat_items = _build_satellite_plot_items(comparison_frames)
    init_time = time_axis[int(init_t_idx)]

    frame_buffer = np.empty((len(r_axis), len(phi_axis)), dtype=np.float32)
    _build_polar_frame(grid_raw, slow_sw_pred_mask, init_t_idx, draw_slow_sw, frame_buffer)

    cmap = plt.cm.plasma.copy()
    cmap.set_bad((1, 1, 1, 0))
    artist = polar_ax.pcolormesh(
        np.deg2rad(phi_axis.astype(np.float32)),
        r_axis.astype(np.float32),
        frame_buffer,
        cmap=cmap,
        shading="nearest",
        vmin=float(post_vlims_raw[0]),
        vmax=float(post_vlims_raw[1]),
    )
    title = polar_ax.set_title(_format_title(init_time), y=1.14, pad=0)
    polar_ax.set_ylim(0.0, float(np.nanmax(r_axis)) + 1.0)
    polar_ax.set_yticklabels([])

    for sat_item in sat_items:
        (sat_item["polar_marker"],) = polar_ax.plot(
            [], [], linestyle="None",
            marker=sat_item["marker"],
            color="black", markerfacecolor="black", markeredgecolor="black",
            markersize=9, clip_on=False, zorder=6,
        )
    _update_satellite_markers(sat_items, init_time, r_axis)

    marker_handles = [
        plt.Line2D(
            [], [], linestyle="None",
            marker=sat_item["marker"],
            color="black", markerfacecolor="black", markeredgecolor="black",
            markersize=8, label=sat_item["label"],
        )
        for sat_item in sat_items
    ]
    if marker_handles:
        sat_legend = polar_ax.legend(
            handles=marker_handles, title="Satellites",
            loc="upper left", bbox_to_anchor=(0.0, 1.0),
            frameon=True, facecolor="white", edgecolor="gray",
            framealpha=0.72, borderaxespad=0.0,
            ncol=1, fontsize=8, title_fontsize=9,
        )
        polar_ax.add_artist(sat_legend)

    colorbar = plt.colorbar(artist, ax=polar_ax, pad=0.14, fraction=0.05)
    colorbar.set_label("v (km/s)")

    window_before_days = float(cr_days - 7.0)
    window_after_days = 7.0
    y_lim = _resolve_panel_ylim(sat_items)
    _initialize_panels(
        sat_axes=sat_axes,
        sat_items=sat_items,
        slow_sw_series=slow_sw_series,
        current_time=init_time,
        window_before_days=window_before_days,
        window_after_days=window_after_days,
        y_lim=y_lim,
    )

    return {
        "fig": fig,
        "polar_ax": polar_ax,
        "sat_axes": sat_axes,
        "sat_items": sat_items,
        "title": title,
        "artist": artist,
        "frame_buffer": frame_buffer,
        "slow_sw_series": slow_sw_series,
        "time_axis": time_axis,
        "r_axis": r_axis,
        "grid_raw": grid_raw,
        "slow_sw_pred_mask": slow_sw_pred_mask,
        "draw_slow_sw": draw_slow_sw,
        "window_before_days": window_before_days,
        "window_after_days": window_after_days,
    }


def _update_polar_view(state, t_idx):
    ti = int(t_idx)
    t_cur = state["time_axis"][ti]
    state["title"].set_text(_format_title(t_cur))
    _build_polar_frame(
        state["grid_raw"], state["slow_sw_pred_mask"],
        ti, state["draw_slow_sw"], state["frame_buffer"],
    )
    state["artist"].set_array(state["frame_buffer"].ravel())
    _update_panel_windows(
        sat_items=state["sat_items"],
        slow_sw_series=state["slow_sw_series"],
        current_time=t_cur,
        window_before_days=state["window_before_days"],
        window_after_days=state["window_after_days"],
    )
    _update_satellite_markers(state["sat_items"], t_cur, state["r_axis"])


def plot_polar_snapshot(
    date_str,
    time_axis,
    phi_axis,
    r_axis,
    grid_raw,
    post_vlims_raw,
    slow_sw_pred_mask,
    slow_sw_speed,
    comparison_frames,
    draw_slow_sw=True,
    cr_days=27.0,
):
    current_time = pd.Timestamp(date_str)
    t_idx = int(pd.DatetimeIndex(time_axis).get_loc(current_time))
    if not np.isfinite(grid_raw[t_idx]).any():
        print(f"No data for {date_str} with current display settings")
        return

    n_sat_panels = max(1, len(comparison_frames))
    fig, polar_ax, sat_axes = _build_figure_axes(n_sat_panels, layout="vertical")
    _build_polar_view(
        fig=fig, polar_ax=polar_ax, sat_axes=sat_axes,
        time_axis=time_axis, phi_axis=phi_axis, r_axis=r_axis,
        grid_raw=grid_raw, post_vlims_raw=post_vlims_raw,
        slow_sw_pred_mask=slow_sw_pred_mask, slow_sw_speed=slow_sw_speed,
        comparison_frames=comparison_frames,
        draw_slow_sw=draw_slow_sw, cr_days=cr_days,
        init_t_idx=t_idx,
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
    time_step_minutes,
    slow_sw_speed,
    cr_days,
    comparison_frames,
    draw_slow_sw=True,
    anim_fps=30,
    anim_1h_mult=1.0,
    anim_dpi=100,
    show_progress=True,
):
    assert float(anim_1h_mult) > 0.0
    output_path = Path(anim_outfile)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame_interval_minutes = 60.0 * float(anim_1h_mult)
    anim_stride = max(1, int(round(frame_interval_minutes / float(time_step_minutes))))
    frame_idx = np.arange(0, len(time_axis), anim_stride, dtype=np.int32)

    n_sat_panels = max(1, len(comparison_frames))
    fig, polar_ax, sat_axes = _build_figure_axes(n_sat_panels, layout="horizontal")
    state = _build_polar_view(
        fig=fig, polar_ax=polar_ax, sat_axes=sat_axes,
        time_axis=time_axis, phi_axis=phi_axis, r_axis=r_axis,
        grid_raw=grid_raw, post_vlims_raw=post_vlims_raw,
        slow_sw_pred_mask=slow_sw_pred_mask, slow_sw_speed=slow_sw_speed,
        comparison_frames=comparison_frames,
        draw_slow_sw=draw_slow_sw, cr_days=cr_days,
        init_t_idx=int(frame_idx[0]),
    )

    fig_w_px = int(fig.get_figwidth() * int(anim_dpi))
    fig_h_px = int(fig.get_figheight() * int(anim_dpi))
    if fig_w_px % 2 != 0:
        fig_w_px += 1
    if fig_h_px % 2 != 0:
        fig_h_px += 1
    fig.set_size_inches(fig_w_px / int(anim_dpi), fig_h_px / int(anim_dpi), forward=True)

    writer = animation.FFMpegWriter(
        fps=int(anim_fps),
        codec="libx264",
        extra_args=["-preset", "ultrafast", "-crf", "23", "-pix_fmt", "yuv420p", "-threads", "0"],
    )

    iterator = frame_idx
    if show_progress:
        iterator = tqdm(frame_idx, desc="Export MP4", unit="frame")

    with writer.saving(fig, str(output_path), dpi=int(anim_dpi)):
        for t_idx in iterator:
            _update_polar_view(state, int(t_idx))
            writer.grab_frame()

    plt.close(fig)
    return {
        "frames": float(len(frame_idx)),
        "stride": float(anim_stride),
        "frame_interval_minutes": float(anim_stride * float(time_step_minutes)),
        "fps": float(anim_fps),
    }
