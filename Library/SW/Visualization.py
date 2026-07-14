from pathlib import Path

from matplotlib import animation
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from Library.SW.Constants import CARRINGTON_ROTATION_DAYS
from Library.SW.Stats import build_icme_duration_by_cr, build_recurrent_series

PREDICT_COLUMN = "v_predict"
PREDICT_RAW_COLUMN = "v_predict_raw"
REAL_COLUMN = "v_real"
SWX_COLUMN = "v_swx"
MICROFORECAST_COLUMN = "v_1cr_ago"
NOAA_COLUMN = "v_noaa"
SAT_MARKER_SHAPES = ("o", "s", "^", "D", "P", "X", "v", "<", ">", "h", "*")
PREDICT_LINE_COLOR = "tab:red"
REAL_LINE_COLOR = "tab:green"
SWX_LINE_COLOR = "tab:orange"
MICROFORECAST_LINE_COLOR = "tab:blue"
NOAA_LINE_COLOR = "tab:purple"
AU_KM = 149597870.7
SECONDS_PER_DAY = 86400.0
AGE_SAMPLE_TOLERANCE = pd.Timedelta(hours=12)
EXPORT_PLOT_FREQ = "1h"


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
    df_noaa=None,
    phi_target=0.0,
    r_target=215.0,
    cr_days=CARRINGTON_ROTATION_DAYS,
    draw_slow_sw=True,
    slow_sw_speed=None,
    slow_sw_patch=True,
    prediction_time_offset_steps=0,
):
    prediction_time_offset_steps = int(prediction_time_offset_steps)
    assert (
        prediction_time_offset_steps >= 0
    ), "prediction_time_offset_steps must be non-negative"

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
    slow_sw_target_mask = np.zeros(len(time_axis), dtype=bool)
    for time_idx, (phi_value, r_value) in enumerate(
        zip(
            target_frame["phi_target"].to_numpy(dtype=float),
            target_frame["r_target"].to_numpy(dtype=float),
        )
    ):
        if not (np.isfinite(phi_value) and np.isfinite(r_value)):
            continue

        prediction_time_idx = time_idx + prediction_time_offset_steps
        if prediction_time_idx >= len(time_axis):
            continue

        phi_idx = find_phi_index(phi_axis, phi_value)
        r_idx = find_axis_index(r_axis, target=r_value)
        raw_value = float(grid_raw[prediction_time_idx, phi_idx, r_idx])
        v_predict_raw[time_idx] = raw_value

        is_slow_sw = bool(
            slow_sw_pred_mask[prediction_time_idx, phi_idx, r_idx]
        )
        slow_sw_target_mask[time_idx] = is_slow_sw

        model_value = raw_value
        if not draw_slow_sw and is_slow_sw:
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
        if slow_sw_patch and df_sat.attrs.get("sat") == "ace_earth":
            assert slow_sw_speed is not None, "slow_sw_patch requires slow_sw_speed for ACE @ Earth"
            slow_sw_series = resolve_slow_sw_speed(time_axis, slow_sw_speed)
            patch_mask = pd.Series(
                slow_sw_target_mask,
                index=pd.DatetimeIndex(time_axis),
                name="slow_sw_patch_mask",
            ).reindex(comparison_frame.index, fill_value=False)
            patch_values = (
                slow_sw_series.reindex(comparison_frame.index)
                .interpolate(method="time")
                .ffill()
                .bfill()
            )
            comparison_frame.loc[patch_mask, "v_predict"] = patch_values.loc[patch_mask]
            comparison_frame["slow_sw_patch_mask"] = patch_mask
        recurrent_index = pd.date_range(
            comparison_frame.index.min().floor(EXPORT_PLOT_FREQ),
            (
                pd.Timestamp(df_sat.index.max())
                + pd.Timedelta(days=float(cr_days))
            ).ceil(EXPORT_PLOT_FREQ),
            freq=EXPORT_PLOT_FREQ,
        )
        recurrent = build_recurrent_series(
            observed_series=df_sat["v"],
            target_index=recurrent_index,
            cr_days=cr_days,
        )
        comparison_frame = comparison_frame.reindex(
            comparison_frame.index.union(recurrent_index)
        )
        comparison_frame["v_1cr_ago"] = recurrent.reindex(comparison_frame.index)
    if df_sat is not None and "lat_hgs" in df_sat.columns:
        comparison_frame = comparison_frame.join(df_sat[["lat_hgs"]], how="left")
    if df_sat is not None and "lat_hge" in df_sat.columns:
        comparison_frame = comparison_frame.join(df_sat[["lat_hge"]], how="left")
    if df_swx is not None and "v_swx" in df_swx.columns:
        comparison_frame = comparison_frame.join(df_swx[["v_swx"]], how="outer")
    if df_noaa is not None and "v_noaa" in df_noaa.columns:
        comparison_frame = comparison_frame.join(df_noaa[["v_noaa"]], how="outer")

    if df_sat is not None:
        comparison_frame.attrs.update(df_sat.attrs)
    elif df_swx is not None:
        comparison_frame.attrs.update(df_swx.attrs)
    return comparison_frame.sort_index()


def first_finite_time(frame, column):
    if column not in frame.columns:
        return None
    series = pd.to_numeric(frame[column], errors="coerce").dropna()
    if series.empty:
        return None
    return pd.Timestamp(series.index.min())


def format_time_or_na(timestamp):
    if timestamp is None:
        return "n/a"
    return f"{timestamp:%Y-%m-%d %H:%M}"


def prepare_export_plot_series(frame, column, start_dt=None, end_dt=None, freq=EXPORT_PLOT_FREQ):
    series = pd.Series(
        pd.to_numeric(frame[column], errors="coerce").to_numpy(),
        index=pd.DatetimeIndex(frame.index),
        name=column,
    )
    series = series[~series.index.duplicated(keep="last")].sort_index()
    if start_dt is not None or end_dt is not None:
        series = series.loc[pd.Timestamp(start_dt) : pd.Timestamp(end_dt)]
    return series.resample(freq).mean()


def build_export_plot_frame(frame, columns, freq=EXPORT_PLOT_FREQ):
    plot_series = [
        prepare_export_plot_series(frame, column, freq=freq)
        for column in columns
        if column in frame.columns
    ]
    if not plot_series:
        return pd.DataFrame(index=pd.DatetimeIndex([], name=frame.index.name))
    return pd.concat(plot_series, axis=1)


def export_solar_wind_plot(
    plot_outfile,
    comparison_frames,
    start_dt,
    end_dt,
    sat_labels=None,
    predict_column=PREDICT_COLUMN,
    dpi=300,
):
    output_path = Path(plot_outfile)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plot_start = pd.Timestamp(start_dt)
    plot_end = pd.Timestamp(end_dt)
    sat_labels = {} if sat_labels is None else sat_labels
    sw_plot_columns = [
        (predict_column, PREDICT_LINE_COLOR, 1.7, "-", predict_column),
        (REAL_COLUMN, REAL_LINE_COLOR, 1.4, "-", "v_obs"),
        (SWX_COLUMN, SWX_LINE_COLOR, 1.35, "-", "v_swx"),
        (MICROFORECAST_COLUMN, MICROFORECAST_LINE_COLOR, 1.25, "--", "v_prev_cr"),
        (NOAA_COLUMN, NOAA_LINE_COLOR, 1.35, "-", "v_noaa"),
    ]
    sw_plot_frames = [
        (
            sat_labels.get(sat_name, frame.attrs.get("label", sat_name)),
            frame.loc[plot_start:plot_end].copy(),
            sat_name,
        )
        for sat_name, frame in comparison_frames.items()
    ]

    y_values = []
    for _label, frame, _sat_name in sw_plot_frames:
        for column, _color, _linewidth, _linestyle, _legend_label in sw_plot_columns:
            if column in frame.columns:
                values = prepare_export_plot_series(
                    frame,
                    column,
                    start_dt=plot_start,
                    end_dt=plot_end,
                ).to_numpy(dtype=float)
                finite_values = values[np.isfinite(values)]
                if finite_values.size > 0:
                    y_values.append(finite_values)

    if y_values:
        y_all = np.concatenate(y_values)
        y_min = float(np.min(y_all))
        y_max = float(np.max(y_all))
    else:
        y_min, y_max = 250.0, 850.0
    y_pad = max(10.0, 0.08 * (y_max - y_min + 1e-6))

    fig, axes = plt.subplots(
        len(sw_plot_frames),
        1,
        figsize=(11.0, max(3.0, 2.75 * len(sw_plot_frames))),
        sharex=True,
        constrained_layout=True,
    )
    axes = np.atleast_1d(axes)

    for ax_idx, (axis, (label, frame, sat_name)) in enumerate(zip(axes, sw_plot_frames)):
        for column, color, linewidth, linestyle, legend_label in sw_plot_columns:
            if column not in frame.columns:
                continue
            series = prepare_export_plot_series(
                frame,
                column,
                start_dt=plot_start,
                end_dt=plot_end,
            )
            if series.empty:
                continue
            axis.plot(
                series.index,
                series.values,
                color=color,
                linewidth=linewidth,
                alpha=0.95,
                linestyle=linestyle,
                label=legend_label,
            )

        title_text = f"{label}: {plot_start:%Y-%m-%d %H:%M} to {plot_end:%Y-%m-%d %H:%M}"
        if sat_name == "stereo_a":
            first_prediction_time = first_finite_time(frame, predict_column)
            title_text += (
                " (first reached by prediction: "
                f"{format_time_or_na(first_prediction_time)})"
            )
        axis.set_title(
            title_text,
            fontsize=10,
            loc="left",
        )
        axis.set_ylabel("v (km/s)")
        axis.grid(alpha=0.25)
        axis.set_ylim(y_min - y_pad, y_max + y_pad)
        axis.legend(loc="upper left", fontsize=8)
        axis.tick_params(axis="x", labelsize=8)
        if ax_idx < len(sw_plot_frames) - 1:
            axis.tick_params(labelbottom=False)

    axes[-1].set_xlim(plot_start, plot_end)
    date_locator = mdates.AutoDateLocator()
    axes[-1].xaxis.set_major_locator(date_locator)
    axes[-1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(date_locator))
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path


def plot_cr_icme_figure(cr_stats, events, sat, comparison_labels=None):
    """Plot per-CR correlations, ICME duration, and Earth-only Dst context."""
    labels = {
        "raw_vs_observed": "Raw model",
        "recurrent_vs_observed": "Recurrent forecast",
    }
    if comparison_labels is not None:
        labels.update(comparison_labels)

    comparison_styles = {
        "raw_vs_observed": "tab:red",
        "recurrent_vs_observed": "tab:blue",
    }
    regime_styles = {
        "all_sw": ("-", "o", "All solar wind"),
        "no_icme": ("--", "s", "ICME excluded"),
    }
    sat_stats = cr_stats.loc[cr_stats["sat"] == sat].copy()
    assert not sat_stats.empty, f"No per-rotation statistics found for sat={sat!r}"
    sat_stats["cr"] = pd.to_numeric(sat_stats["cr"], errors="coerce")
    sat_stats["r"] = pd.to_numeric(sat_stats["r"], errors="coerce")

    fig, (correlation_axis, event_axis) = plt.subplots(
        2,
        1,
        figsize=(10.0, 6.3),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"height_ratios": [2.0, 1.0]},
    )

    for comparison, color in comparison_styles.items():
        for regime, (linestyle, marker, regime_label) in regime_styles.items():
            line_data = sat_stats.loc[
                (sat_stats["comparison"] == comparison)
                & (sat_stats["regime"] == regime)
            ].sort_values("cr")
            if line_data.empty:
                continue
            correlation_axis.plot(
                line_data["cr"],
                line_data["r"],
                color=color,
                linestyle=linestyle,
                marker=marker,
                markersize=4.5,
                linewidth=1.4,
                label=f"{labels[comparison]}, {regime_label}",
            )

    correlation_axis.axhline(0.0, color="0.5", linewidth=0.8)
    correlation_axis.set_ylim(-1.0, 1.0)
    correlation_axis.set_ylabel("Pearson correlation, r")
    correlation_axis.set_title(sat, loc="left")
    correlation_axis.grid(alpha=0.25)
    correlation_axis.legend(loc="best", fontsize=8, ncols=2)

    duration_data = build_icme_duration_by_cr(
        cr_stats=cr_stats,
        events=events,
        sat=sat,
    )
    event_axis.hlines(
        duration_data["icme_duration_hours"],
        duration_data["cr"] - 0.42,
        duration_data["cr"] + 0.42,
        color="tab:orange",
        linewidth=3.0,
        zorder=3,
    )
    event_axis.set_ylabel("ICME duration per CR (h)")
    event_axis.set_xlabel("Carrington rotation")
    event_axis.set_ylim(bottom=0.0)
    event_axis.set_xlim(
        duration_data["cr"].min() - 0.5,
        duration_data["cr"].max() + 0.5,
    )
    event_axis.grid(alpha=0.25)

    sat_events = events.loc[events["sat"] == sat].copy()
    sat_events["event_cr"] = pd.to_numeric(
        sat_events["event_cr"], errors="coerce"
    )
    sat_events["dst_min"] = pd.to_numeric(
        sat_events["dst_min"], errors="coerce"
    )
    dst_events = sat_events.dropna(subset=["event_cr", "dst_min"])
    if not dst_events.empty:
        dst_axis = event_axis.twinx()
        dst_axis.scatter(
            dst_events["event_cr"],
            dst_events["dst_min"],
            color="tab:purple",
            marker="v",
            s=24,
            zorder=4,
        )
        dst_axis.set_ylabel("Minimum Dst per ICME (nT)", color="tab:purple")
        dst_axis.tick_params(axis="y", colors="tab:purple")
        dst_axis.axhline(0.0, color="tab:purple", linewidth=0.7, alpha=0.4)

    return fig


def _build_satellite_plot_items(comparison_frames):
    sat_items = []
    for sat_idx, (sat_name, frame) in enumerate(comparison_frames.items()):
        plot_columns = [
            column
            for column in (
                PREDICT_COLUMN,
                REAL_COLUMN,
                NOAA_COLUMN,
            )
            if column in frame.columns
        ]
        sat_items.append(
            {
                "sat_name": sat_name,
                "frame": frame,
                "panel_frame": build_export_plot_frame(frame, plot_columns),
                "label": frame.attrs.get("label", sat_name),
                "predict_col": PREDICT_COLUMN if PREDICT_COLUMN in frame.columns else None,
                "real_col": REAL_COLUMN if REAL_COLUMN in frame.columns else None,
                "swx_col": None,
                "microforecast_col": None,
                "noaa_col": NOAA_COLUMN if NOAA_COLUMN in frame.columns else None,
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


def _compute_ch_age_days(phi_deg, speed_km_s, cr_days):
    if not (np.isfinite(phi_deg) and np.isfinite(speed_km_s) and float(speed_km_s) > 0.0):
        return np.nan
    phi_fraction = np.mod(float(phi_deg), 360.0) / 360.0
    rotation_days = phi_fraction * float(cr_days)
    travel_days = AU_KM / float(speed_km_s) / SECONDS_PER_DAY
    return rotation_days + travel_days


def _resolve_age_inputs(frame, current_time):
    if not {"phi_target", "v_predict"}.issubset(frame.columns):
        return np.nan, np.nan
    age_frame = frame[["phi_target", "v_predict"]].dropna().sort_index()
    age_frame = age_frame[~age_frame.index.duplicated(keep="last")]
    if age_frame.empty:
        return np.nan, np.nan
    nearest_pos = age_frame.index.get_indexer(
        [pd.Timestamp(current_time)],
        method="nearest",
        tolerance=AGE_SAMPLE_TOLERANCE,
    )[0]
    if nearest_pos < 0:
        return np.nan, np.nan
    nearest_row = age_frame.iloc[int(nearest_pos)]
    return float(nearest_row["phi_target"]), float(nearest_row["v_predict"])


def _format_satellite_panel_title(label, window_before_days, window_after_days, age_days):
    if np.isfinite(age_days):
        age_text = f"age={age_days:.2f} d"
    else:
        age_text = "age=n/a"
    return (
        f"   {label} ({age_text}): "
        f"t-{float(window_before_days):.0f}d to t+{float(window_after_days):.0f}d"
    )


def _resolve_panel_ylim(sat_items):
    y_parts = []
    for sat_item in sat_items:
        panel_frame = sat_item["panel_frame"]
        for column in (
            sat_item["predict_col"],
            sat_item["real_col"],
            sat_item["swx_col"],
            sat_item["microforecast_col"],
            sat_item["noaa_col"],
        ):
            if column is None:
                continue
            values = panel_frame[column].to_numpy(dtype=float)
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


def _update_panel_titles(sat_items, current_time, window_before_days, window_after_days):
    current_time = pd.Timestamp(current_time)
    for sat_item in sat_items:
        sat_frame = sat_item["frame"]
        phi_value, speed_value = _resolve_age_inputs(sat_frame, current_time)
        age_days = _compute_ch_age_days(
            phi_deg=phi_value,
            speed_km_s=speed_value,
            cr_days=window_before_days + window_after_days,
        )
        sat_item["axis"].set_title(
            _format_satellite_panel_title(
                sat_item["label"],
                window_before_days=window_before_days,
                window_after_days=window_after_days,
                age_days=age_days,
            ),
            fontsize=10,
            loc="left",
        )


def _update_panel_windows(sat_items, current_time, window_before_days, window_after_days):
    current_time = pd.Timestamp(current_time)
    window_start = current_time - pd.Timedelta(days=float(window_before_days))
    window_end = current_time + pd.Timedelta(days=float(window_after_days))
    for sat_item in sat_items:
        compare_window = sat_item["panel_frame"].loc[window_start:window_end]
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
        if sat_item["noaa_line"] is not None:
            _noaa = compare_window[sat_item["noaa_col"]].dropna()
            sat_item["noaa_line"].set_data(_noaa.index, _noaa.values)
        sat_item["time_marker"].set_xdata([current_time, current_time])
        sat_item["axis"].set_xlim(window_start, window_end)


def _initialize_panels(sat_axes, sat_items, current_time, window_before_days, window_after_days, y_lim):
    for sat_idx, sat_item in enumerate(sat_items):
        sat_axis = sat_axes[sat_idx]
        sat_item["axis"] = sat_axis
        sat_item["predict_line"] = None
        sat_item["real_line"] = None
        sat_item["swx_line"] = None
        sat_item["microforecast_line"] = None
        sat_item["noaa_line"] = None
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
            (sat_item["microforecast_line"],) = sat_axis.plot([], [], color=MICROFORECAST_LINE_COLOR, linewidth=1.25, alpha=0.95, linestyle="--", label="v_prev_cr")
        if sat_item["noaa_col"] is not None:
            (sat_item["noaa_line"],) = sat_axis.plot([], [], color=NOAA_LINE_COLOR, linewidth=1.35, alpha=0.95, linestyle="-", label="v_noaa")
        sat_axis.set_ylabel("v (km/s)")
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
        _update_panel_titles(
            sat_items=sat_items,
            current_time=current_time,
            window_before_days=window_before_days,
            window_after_days=window_after_days,
        )
        _update_panel_windows(
            sat_items=sat_items,
            current_time=current_time,
            window_before_days=window_before_days,
            window_after_days=window_after_days,
        )


def _build_figure_axes(n_sat_panels, layout):
    """Create figure and (polar_ax, sat_axes) for a layout."""
    panel_height = 1.55
    if layout == "vertical":
        fig_height = max(10.0, 7.0 + 2.1 * n_sat_panels)
        fig = plt.figure(figsize=(8.8, fig_height))
        grid_spec = fig.add_gridspec(
            n_sat_panels + 1, 1,
            height_ratios=[4.2] + [1.35] * n_sat_panels,
            hspace=0.46,
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
        current_time=t_cur,
        window_before_days=state["window_before_days"],
        window_after_days=state["window_after_days"],
    )
    _update_panel_titles(
        sat_items=state["sat_items"],
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
    cr_days=CARRINGTON_ROTATION_DAYS,
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
    comparison_frames,
    draw_slow_sw=True,
    cr_days=CARRINGTON_ROTATION_DAYS,
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
