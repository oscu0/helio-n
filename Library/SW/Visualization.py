from pathlib import Path

from matplotlib import animation
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from Library.SW.Coords import find_axis_index

PREDICT_COLUMN_CANDIDATES = ("v_predict", "v_model_earth")
PREDICT_RAW_COLUMN_CANDIDATES = ("v_predict_raw", "v_model_earth_raw")
REAL_COLUMN_CANDIDATES = ("v_real", "v_ace")
SWX_COLUMN_CANDIDATES = ("v_swx", "forecast_sw_speed")
SAT_MARKER_SHAPES = ("o", "s", "^", "D", "P", "X", "v", "<", ">", "h", "*")
PREDICT_LINE_COLOR = "tab:red"
REAL_LINE_COLOR = "tab:green"
SWX_LINE_COLOR = "tab:orange"
SLOW_SW_LINE_COLOR = "tab:gray"


def resolve_slow_sw_speed(time_axis, slow_sw_speed):
    values = np.asarray(slow_sw_speed, dtype=float)
    if values.ndim == 0:
        values = np.full(len(time_axis), float(values))
    assert len(values) == len(time_axis)
    return pd.Series(values, index=pd.DatetimeIndex(time_axis), name="slow_sw_speed")


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


def wrap_phi_delta(phi_axis, phi_target):
    return np.abs(
        (np.asarray(phi_axis, dtype=float) - float(phi_target) + 180.0) % 360.0 - 180.0
    )


def find_phi_index(phi_axis, phi_target):
    return int(np.argmin(wrap_phi_delta(phi_axis, phi_target)))


def first_present_column(frame, candidates):
    for column in candidates:
        if column in frame.columns:
            return column
    return None


def resolve_comparison_frames(comparison_frames=None, earth_frame=None):
    if comparison_frames is None:
        assert earth_frame is not None, "Provide comparison_frames or earth_frame"
        if isinstance(earth_frame, pd.DataFrame):
            return {"earth": earth_frame}
        return dict(earth_frame)
    if isinstance(comparison_frames, pd.DataFrame):
        return {"earth": comparison_frames}
    return dict(comparison_frames)


def marker_radius_for_plot(r_value, r_axis):
    r_max = float(np.nanmax(r_axis))
    return float(np.clip(float(r_value), float(np.nanmin(r_axis)), r_max - 0.75))


def add_satellite_marker_legend(ax, sat_items):
    marker_handles = []
    for sat_item in sat_items:
        marker_handle = plt.Line2D(
            [],
            [],
            linestyle="None",
            marker=sat_item["marker"],
            color="black",
            markerfacecolor="black",
            markeredgecolor="black",
            markersize=8,
            label=sat_item["label"],
        )
        marker_handles.append(marker_handle)

    if marker_handles:
        sat_legend = ax.legend(
            handles=marker_handles,
            title="Satellites",
            loc="upper left",
            bbox_to_anchor=(0.0, 1.0),
            frameon=True,
            facecolor="white",
            edgecolor="gray",
            framealpha=0.72,
            borderaxespad=0.0,
            ncol=1,
            fontsize=8,
            title_fontsize=9,
        )
        ax.add_artist(sat_legend)


def build_satellite_plot_items(comparison_frames):
    sat_items = []
    for sat_idx, (sat_name, frame) in enumerate(comparison_frames.items()):
        sat_items.append(
            {
                "sat_name": sat_name,
                "frame": frame,
                "label": frame.attrs.get("label", sat_name),
                "predict_col": first_present_column(frame, PREDICT_COLUMN_CANDIDATES),
                "real_col": first_present_column(frame, REAL_COLUMN_CANDIDATES),
                "swx_col": first_present_column(frame, SWX_COLUMN_CANDIDATES),
                "marker": SAT_MARKER_SHAPES[sat_idx % len(SAT_MARKER_SHAPES)],
            }
        )
    return sat_items


def build_polar_speed_cmap():
    cmap = plt.cm.plasma.copy()
    cmap.set_bad((1, 1, 1, 0))
    return cmap


def resolve_polar_speed_vlims(
    post_vlims_raw,
    slow_sw_series,
    backfill_empty_with_300=False,
):
    if backfill_empty_with_300:
        return float(slow_sw_series.min()), float(post_vlims_raw[1])
    return float(post_vlims_raw[0]), float(post_vlims_raw[1])


def build_polar_speed_frame(
    grid_raw,
    slow_sw_pred_mask,
    slow_sw_series,
    t_idx,
    draw_slow_sw=True,
    backfill_empty_with_300=False,
    frame_buffer=None,
):
    if frame_buffer is None:
        frame_buffer = np.empty((grid_raw.shape[2], grid_raw.shape[1]), dtype=np.float32)

    ti = int(t_idx)
    np.copyto(frame_buffer, grid_raw[ti].T)
    if backfill_empty_with_300:
        np.nan_to_num(frame_buffer, copy=False, nan=float(slow_sw_series.iloc[ti]))
    if not draw_slow_sw:
        frame_buffer[slow_sw_pred_mask[ti].T] = np.nan
    return frame_buffer


def draw_polar_speed_mesh(
    ax,
    phi_axis,
    r_axis,
    frame_data,
    vmin,
    vmax,
    cmap=None,
):
    if cmap is None:
        cmap = build_polar_speed_cmap()
    theta = np.deg2rad(phi_axis.astype(np.float32))
    radius = r_axis.astype(np.float32)
    return ax.pcolormesh(
        theta,
        radius,
        frame_data,
        cmap=cmap,
        shading="nearest",
        vmin=vmin,
        vmax=vmax,
    )


def update_polar_speed_mesh(artist, frame_data):
    artist.set_array(frame_data.ravel())


def format_polar_speed_title(current_time):
    return f"Heliosphere at {pd.Timestamp(current_time).strftime('%Y-%m-%d %H:%M:%S')}"


def configure_polar_speed_axis(ax, r_axis, r0, title_text=""):
    title = ax.set_title(title_text, y=1.14, pad=0)
    ax.set_ylim(0.0, float(np.nanmax(r_axis)) + 1.0)
    ax.set_yticklabels([])
    return title


def add_polar_speed_colorbar(ax, artist):
    colorbar = plt.colorbar(artist, ax=ax, pad=0.14, fraction=0.05)
    colorbar.set_label("v (km/s)")
    return colorbar


def add_satellite_polar_markers(ax, sat_items):
    for sat_item in sat_items:
        (sat_item["polar_marker"],) = ax.plot(
            [],
            [],
            linestyle="None",
            marker=sat_item["marker"],
            color="black",
            markerfacecolor="black",
            markeredgecolor="black",
            markersize=9,
            clip_on=False,
            zorder=6,
        )


def update_satellite_polar_markers(sat_items, current_time, r_axis):
    current_time = pd.Timestamp(current_time)
    for sat_item in sat_items:
        sat_frame = sat_item["frame"]
        if current_time not in sat_frame.index:
            sat_item["polar_marker"].set_data([], [])
            continue
        phi_value = float(sat_frame.loc[current_time, "phi_target"])
        r_value = marker_radius_for_plot(
            sat_frame.loc[current_time, "r_target"], r_axis=r_axis
        )
        sat_item["polar_marker"].set_data([np.deg2rad(phi_value)], [r_value])


def resolve_satellite_panel_ylim(sat_items):
    y_parts = []
    for sat_item in sat_items:
        for column in (
            sat_item["predict_col"],
            sat_item["real_col"],
            sat_item["swx_col"],
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
        y_min = 250.0
        y_max = 850.0
    y_pad = max(10.0, 0.08 * (y_max - y_min + 1e-6))
    return y_min - y_pad, y_max + y_pad


def update_satellite_panel_windows(
    sat_items,
    slow_sw_series,
    current_time,
    window_before_days,
    window_after_days,
):
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
            sat_item["real_line"].set_data(
                compare_window.index,
                compare_window[sat_item["real_col"]].values,
            )
        if sat_item["swx_line"] is not None:
            sat_item["swx_line"].set_data(
                compare_window.index,
                compare_window[sat_item["swx_col"]].values,
            )
        slow_sw_window = slow_sw_series.loc[window_start:window_end]
        sat_item["slow_sw_line"].set_data(slow_sw_window.index, slow_sw_window.values)
        sat_item["time_marker"].set_xdata([current_time, current_time])
        sat_item["axis"].set_xlim(window_start, window_end)


def initialize_satellite_panels(
    sat_axes,
    sat_items,
    slow_sw_series,
    current_time,
    window_before_days,
    window_after_days,
    y_lim,
):
    for sat_idx, sat_item in enumerate(sat_items):
        sat_axis = sat_axes[sat_idx]
        sat_item["axis"] = sat_axis
        sat_item["predict_line"] = None
        sat_item["real_line"] = None
        sat_item["swx_line"] = None
        sat_item["slow_sw_line"] = None
        sat_item["time_marker"] = sat_axis.axvline(
            current_time,
            color="black",
            linestyle="--",
            linewidth=1.0,
            alpha=0.7,
        )

        if sat_item["predict_col"] is not None:
            (sat_item["predict_line"],) = sat_axis.plot(
                [],
                [],
                color=PREDICT_LINE_COLOR,
                linewidth=1.7,
                label="v_predict",
            )
        if sat_item["real_col"] is not None:
            (sat_item["real_line"],) = sat_axis.plot(
                [],
                [],
                color=REAL_LINE_COLOR,
                linewidth=1.4,
                alpha=0.95,
                linestyle="--",
                label="v_real",
            )
        if sat_item["swx_col"] is not None:
            (sat_item["swx_line"],) = sat_axis.plot(
                [],
                [],
                color=SWX_LINE_COLOR,
                linewidth=1.35,
                alpha=0.95,
                linestyle=":",
                label="v_swx",
            )
        (sat_item["slow_sw_line"],) = sat_axis.plot(
            slow_sw_series.index,
            slow_sw_series.values,
            color=SLOW_SW_LINE_COLOR,
            linestyle="-.",
            linewidth=1.0,
            alpha=0.75,
            label="slow SW",
        )
        sat_axis.set_ylabel("v (km/s)")
        sat_axis.set_title(
            f"   {sat_item['label']}: t-{float(window_before_days):.0f}d to "
            f"t+{float(window_after_days):.0f}d",
            fontsize=10,
            loc="left",
        )
        sat_axis.plot(
            [0.012],
            [1.035],
            transform=sat_axis.transAxes,
            linestyle="None",
            marker=sat_item["marker"],
            color="black",
            markerfacecolor="black",
            markeredgecolor="black",
            markersize=7,
            clip_on=False,
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
        update_satellite_panel_windows(
            sat_items=sat_items,
            slow_sw_series=slow_sw_series,
            current_time=current_time,
            window_before_days=window_before_days,
            window_after_days=window_after_days,
        )


def build_model_target_frame(time_axis, df_sat, phi_target, r_target):
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
    return target_frame


def build_satellite_comparison_frame(
    time_axis,
    phi_axis,
    r_axis,
    grid_raw,
    slow_sw_pred_mask,
    slow_sw_speed,
    df_sat=None,
    df_swx=None,
    phi_target=0.0,
    r_target=215.0,
    draw_slow_sw=True,
    backfill_empty_with_300=False,
):
    slow_sw_series = resolve_slow_sw_speed(time_axis, slow_sw_speed)
    target_frame = build_model_target_frame(
        time_axis=time_axis,
        df_sat=df_sat,
        phi_target=phi_target,
        r_target=r_target,
    )

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
        if backfill_empty_with_300 and np.isnan(model_value):
            model_value = float(slow_sw_series.iloc[time_idx])
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
    if df_sat is not None and "lat_hgs" in df_sat.columns:
        comparison_frame = comparison_frame.join(df_sat[["lat_hgs"]], how="left")
    if df_sat is not None and "lat_hge" in df_sat.columns:
        comparison_frame = comparison_frame.join(df_sat[["lat_hge"]], how="left")
    if df_swx is not None and "v_swx" in df_swx.columns:
        comparison_frame = comparison_frame.join(df_swx[["v_swx"]], how="outer")

    if df_sat is not None:
        comparison_frame.attrs.update(df_sat.attrs)
    elif df_swx is not None:
        comparison_frame.attrs.update(df_swx.attrs)
    return comparison_frame.sort_index()


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
    df_sat = None
    if df_ace_earth is not None:
        df_sat = df_ace_earth.rename(columns={"v_ace": "v"})
        df_sat.attrs.update(df_ace_earth.attrs)
    df_swx = None
    if df_forecast_earth is not None:
        df_swx = df_forecast_earth.rename(columns={"forecast_sw_speed": "v_swx"})
        df_swx.attrs.update(df_forecast_earth.attrs)

    earth_frame = build_satellite_comparison_frame(
        time_axis=time_axis,
        phi_axis=phi_axis,
        r_axis=r_axis,
        grid_raw=grid_raw,
        slow_sw_pred_mask=slow_sw_pred_mask,
        slow_sw_speed=slow_sw_speed,
        df_sat=df_sat,
        df_swx=df_swx,
        phi_target=phi_target,
        r_target=r_target,
        draw_slow_sw=draw_slow_sw,
        backfill_empty_with_300=backfill_empty_with_300,
    )
    return earth_frame.rename(
        columns={
            "v_predict_raw": "v_model_earth_raw",
            "v_predict": "v_model_earth",
            "v_real": "v_ace",
            "v_swx": "forecast_sw_speed",
        }
    )


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
    comparison_frames=None,
    earth_frame=None,
    draw_slow_sw=True,
    backfill_empty_with_300=False,
    cr_days=27.0,
):
    slow_sw_series = resolve_slow_sw_speed(time_axis, slow_sw_speed)
    comparison_frames = resolve_comparison_frames(
        comparison_frames=comparison_frames,
        earth_frame=earth_frame,
    )
    sat_items = build_satellite_plot_items(comparison_frames)
    current_time = pd.Timestamp(date_str)
    t_idx = int(pd.DatetimeIndex(time_axis).get_loc(current_time))
    vmin_mode, vmax_mode = resolve_polar_speed_vlims(
        post_vlims_raw=post_vlims_raw,
        slow_sw_series=slow_sw_series,
        backfill_empty_with_300=backfill_empty_with_300,
    )
    frame_data = build_polar_speed_frame(
        grid_raw=grid_raw,
        slow_sw_pred_mask=slow_sw_pred_mask,
        slow_sw_series=slow_sw_series,
        t_idx=t_idx,
        draw_slow_sw=draw_slow_sw,
        backfill_empty_with_300=backfill_empty_with_300,
    )
    if not np.isfinite(frame_data).any():
        print(f"No data for {date_str} with current display settings")
        return

    window_before_days = float(cr_days - 7.0)
    window_after_days = 7.0
    y_lim = resolve_satellite_panel_ylim(sat_items)
    n_sat_panels = max(1, len(sat_items))
    panel_height = 1.55
    fig_height = max(8.6, 6.4 + panel_height * n_sat_panels)
    fig = plt.figure(figsize=(8.4, fig_height))
    grid_spec = fig.add_gridspec(
        n_sat_panels + 1,
        1,
        height_ratios=[4.2] + [1.0] * n_sat_panels,
        hspace=0.5,
    )
    ax = fig.add_subplot(grid_spec[0, 0], projection="polar")
    sat_axes = []
    shared_axis = None
    for sat_idx in range(n_sat_panels):
        sat_axis = fig.add_subplot(
            grid_spec[sat_idx + 1, 0],
            sharex=shared_axis if shared_axis is not None else None,
        )
        sat_axes.append(sat_axis)
        if shared_axis is None:
            shared_axis = sat_axis
    fig.subplots_adjust(top=0.94, right=0.86, left=0.12, bottom=0.06)

    artist = draw_polar_speed_mesh(
        ax=ax,
        phi_axis=phi_axis,
        r_axis=r_axis,
        frame_data=frame_data,
        vmin=vmin_mode,
        vmax=vmax_mode,
    )
    add_polar_speed_colorbar(ax=ax, artist=artist)
    configure_polar_speed_axis(
        ax=ax,
        r_axis=r_axis,
        r0=r0,
        title_text=format_polar_speed_title(current_time),
    )
    add_satellite_polar_markers(ax=ax, sat_items=sat_items)
    update_satellite_polar_markers(
        sat_items=sat_items,
        current_time=current_time,
        r_axis=r_axis,
    )
    add_satellite_marker_legend(ax, sat_items)
    initialize_satellite_panels(
        sat_axes=sat_axes,
        sat_items=sat_items,
        slow_sw_series=slow_sw_series,
        current_time=current_time,
        window_before_days=window_before_days,
        window_after_days=window_after_days,
        y_lim=y_lim,
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
    superresolution_enabled,
    slow_sw_speed,
    r0,
    cr_days,
    comparison_frames=None,
    earth_frame=None,
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
    assert anim_plot_style == "mesh"
    slow_sw_series = resolve_slow_sw_speed(time_axis, slow_sw_speed)
    comparison_frames = resolve_comparison_frames(
        comparison_frames=comparison_frames,
        earth_frame=earth_frame,
    )

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

    anim_vmin, anim_vmax = resolve_polar_speed_vlims(
        post_vlims_raw=post_vlims_raw,
        slow_sw_series=slow_sw_series,
        backfill_empty_with_300=backfill_empty_with_300,
    )

    frame_idx = np.arange(0, len(time_axis), anim_stride, dtype=np.int32)
    window_before_days = float(cr_days - 7.0)
    window_after_days = 7.0

    sat_items = build_satellite_plot_items(comparison_frames)
    y_lim = resolve_satellite_panel_ylim(sat_items)

    n_sat_panels = max(1, len(sat_items))
    panel_height = 1.55
    fig_width = 14.2
    fig_height = max(6.8, panel_height * n_sat_panels + 0.9)
    fig = plt.figure(figsize=(fig_width, fig_height))
    height_ratios = [1.0] * n_sat_panels
    grid_spec = fig.add_gridspec(
        n_sat_panels,
        2,
        width_ratios=[1.35, 1.0],
        height_ratios=height_ratios,
        wspace=0.34,
        hspace=0.48,
    )
    ax = fig.add_subplot(grid_spec[:, 0], projection="polar")
    sat_axes = []
    shared_axis = None
    for sat_idx in range(n_sat_panels):
        sat_axis = fig.add_subplot(
            grid_spec[sat_idx, 1],
            sharex=shared_axis if shared_axis is not None else None,
        )
        sat_axes.append(sat_axis)
        if shared_axis is None:
            shared_axis = sat_axis
    fig.subplots_adjust(top=0.88, right=0.94, left=0.045, bottom=0.08)

    frame_buf_2d = np.empty((len(r_axis), len(phi_axis)), dtype=np.float32)
    init_t = int(frame_idx[0])
    build_polar_speed_frame(
        grid_raw=grid_raw,
        slow_sw_pred_mask=slow_sw_pred_mask,
        slow_sw_series=slow_sw_series,
        t_idx=init_t,
        draw_slow_sw=draw_slow_sw,
        backfill_empty_with_300=backfill_empty_with_300,
        frame_buffer=frame_buf_2d,
    )
    artist = draw_polar_speed_mesh(
        ax=ax,
        phi_axis=phi_axis,
        r_axis=r_axis,
        frame_data=frame_buf_2d,
        vmin=anim_vmin,
        vmax=anim_vmax,
    )
    title = configure_polar_speed_axis(
        ax=ax,
        r_axis=r_axis,
        r0=r0,
        title_text=format_polar_speed_title(time_axis[init_t]),
    )

    add_satellite_polar_markers(ax=ax, sat_items=sat_items)
    update_satellite_polar_markers(
        sat_items=sat_items,
        current_time=time_axis[init_t],
        r_axis=r_axis,
    )
    add_satellite_marker_legend(ax, sat_items)

    add_polar_speed_colorbar(ax=ax, artist=artist)

    initialize_satellite_panels(
        sat_axes=sat_axes,
        sat_items=sat_items,
        slow_sw_series=slow_sw_series,
        current_time=time_axis[init_t],
        window_before_days=window_before_days,
        window_after_days=window_after_days,
        y_lim=y_lim,
    )

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
            title.set_text(format_polar_speed_title(t_cur))

            build_polar_speed_frame(
                grid_raw=grid_raw,
                slow_sw_pred_mask=slow_sw_pred_mask,
                slow_sw_series=slow_sw_series,
                t_idx=ti,
                draw_slow_sw=draw_slow_sw,
                backfill_empty_with_300=backfill_empty_with_300,
                frame_buffer=frame_buf_2d,
            )
            update_polar_speed_mesh(artist=artist, frame_data=frame_buf_2d)

            update_satellite_panel_windows(
                sat_items=sat_items,
                slow_sw_series=slow_sw_series,
                current_time=t_cur,
                window_before_days=window_before_days,
                window_after_days=window_after_days,
            )
            update_satellite_polar_markers(
                sat_items=sat_items,
                current_time=t_cur,
                r_axis=r_axis,
            )
            writer.grab_frame()

    plt.close(fig)
    return {
        "frames": float(len(frame_idx)),
        "stride": float(anim_stride),
        "fps": float(anim_fps),
        "achieved_speedup": float(achieved_speedup),
    }
