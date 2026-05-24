from pathlib import Path

import numpy as np
import pandas as pd

from Library.ICME import build_icme_mask

SCORE_FREQ = "1h"
FORECAST_ORDER = {"pred": 0, "prev_cr": 1, "swx": 2, "noaa": 3}
REGIME_ORDER = [
    "all_sw",
    "no_icme",
    "no_icme_no_slow_sw",
]
REGIME_SORT_ORDER = [
    "all_sw",
    "no_slow_sw",
    "no_icme",
    "no_icme_no_slow_sw",
]
REGIME_ORDER_BY_SAT = {
    "stereo_a": [
        "all_sw",
        "no_slow_sw",
    ],
}
FORECAST_COLUMNS_BY_SAT = {
    "ace_earth": [
        ("pred", "v_predict"),
        ("prev_cr", "v_1cr_ago"),
        ("swx", "v_swx"),
        ("noaa", "v_noaa"),
    ],
    "stereo_a": [
        ("pred", "v_predict"),
        ("prev_cr", "v_1cr_ago"),
        ("noaa", "v_noaa"),
    ],
}
DEFAULT_FORECAST_COLUMNS = [
    ("pred", "v_predict"),
    ("prev_cr", "v_1cr_ago"),
    ("swx", "v_swx"),
    ("noaa", "v_noaa"),
]


def build_eval_index(start_dt, end_dt, freq=SCORE_FREQ):
    return pd.date_range(pd.Timestamp(start_dt), pd.Timestamp(end_dt), freq=freq)


def prepare_eval_series(series, eval_index, output_name, freq=SCORE_FREQ):
    prepared = pd.Series(
        pd.to_numeric(series, errors="coerce").to_numpy(),
        index=pd.DatetimeIndex(series.index),
        name=output_name,
    )
    prepared = prepared[~prepared.index.duplicated(keep="last")].sort_index()
    prepared = prepared.loc[eval_index.min() : eval_index.max()]
    prepared = prepared.resample(freq).mean()
    return prepared.reindex(eval_index)


def prepare_eval_mask(series, eval_index, output_name, freq=SCORE_FREQ):
    prepared = pd.Series(
        pd.Series(series).fillna(False).astype(bool).to_numpy(),
        index=pd.DatetimeIndex(series.index),
        name=output_name,
    )
    prepared = prepared[~prepared.index.duplicated(keep="last")].sort_index()
    prepared = prepared.loc[eval_index.min() : eval_index.max()]
    prepared = prepared.resample(freq).max()
    return prepared.reindex(eval_index, fill_value=False).astype(bool)


def build_slow_sw_eval_mask(frame, eval_index, time_axis, slow_sw_speed):
    if "slow_sw_patch_mask" in frame.columns:
        return prepare_eval_mask(frame["slow_sw_patch_mask"], eval_index, "is_slow_sw")

    raw_column = "v_predict_raw" if "v_predict_raw" in frame.columns else "v_predict"
    raw_speed = pd.Series(
        pd.to_numeric(frame[raw_column], errors="coerce").to_numpy(),
        index=pd.DatetimeIndex(frame.index),
        name=raw_column,
    )
    slow_ref = pd.Series(
        slow_sw_speed,
        index=pd.DatetimeIndex(time_axis),
        name="slow_sw_speed",
    )
    slow_ref = (
        slow_ref.reindex(raw_speed.index).interpolate(method="time").ffill().bfill()
    )
    is_slow = pd.Series(
        np.isclose(raw_speed.to_numpy(dtype=float), slow_ref.to_numpy(dtype=float)),
        index=raw_speed.index,
        name="is_slow_sw",
    )
    return prepare_eval_mask(is_slow, eval_index, "is_slow_sw")


def build_sat_score_frame(
    sat_name,
    frame,
    eval_index,
    icme_mask,
    time_axis,
    slow_sw_speed,
    forecast_columns_by_sat=FORECAST_COLUMNS_BY_SAT,
):
    assert "v_real" in frame.columns, f"Missing observed speed column for {sat_name}"

    out = pd.DataFrame(index=eval_index)
    out["obs"] = prepare_eval_series(frame["v_real"], eval_index, "obs")
    out["phi_target"] = prepare_eval_series(
        frame["phi_target"], eval_index, "phi_target"
    )
    out["is_icme"] = icme_mask.reindex(eval_index, fill_value=False).astype(bool)
    out["is_slow_sw"] = build_slow_sw_eval_mask(
        frame=frame,
        eval_index=eval_index,
        time_axis=time_axis,
        slow_sw_speed=slow_sw_speed,
    )

    forecast_columns = forecast_columns_by_sat.get(sat_name, DEFAULT_FORECAST_COLUMNS)
    for forecast_name, column in forecast_columns:
        if column in frame.columns:
            out[forecast_name] = prepare_eval_series(
                frame[column], eval_index, forecast_name
            )

    return out


def build_regime_masks(eval_frame):
    return {
        "all_sw": pd.Series(True, index=eval_frame.index),
        "no_slow_sw": ~eval_frame["is_slow_sw"],
        "no_icme": ~eval_frame["is_icme"],
        "no_icme_no_slow_sw": ~eval_frame["is_icme"] & ~eval_frame["is_slow_sw"],
    }


def get_regime_order(sat_name):
    return REGIME_ORDER_BY_SAT.get(sat_name, REGIME_ORDER)


def compute_forecast_stats(actual, forecast, sample_mask):
    paired = pd.concat(
        [
            pd.to_numeric(actual, errors="coerce").rename("actual"),
            pd.to_numeric(forecast, errors="coerce").rename("forecast"),
            sample_mask.rename("sample_mask"),
        ],
        axis=1,
    )
    paired = paired.loc[paired["sample_mask"]].dropna(subset=["actual", "forecast"])
    if len(paired) == 0:
        return {
            "n_samples": 0,
            "r": np.nan,
            "rmse": np.nan,
            "mae": np.nan,
            "bias": np.nan,
        }

    err = paired["forecast"] - paired["actual"]
    r_value = np.nan
    if len(paired) >= 2:
        actual_std = float(paired["actual"].std())
        forecast_std = float(paired["forecast"].std())
        if actual_std > 0.0 and forecast_std > 0.0:
            r_value = float(paired["actual"].corr(paired["forecast"]))

    return {
        "n_samples": int(len(paired)),
        "r": r_value,
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "bias": float(np.mean(err)),
    }


def build_score_frames(
    comparison_frames,
    start_dt,
    end_dt,
    time_axis,
    slow_sw_speed,
    freq=SCORE_FREQ,
    forecast_columns_by_sat=FORECAST_COLUMNS_BY_SAT,
):
    eval_index = build_eval_index(start_dt, end_dt, freq=freq)
    icme_eval_mask = build_icme_mask(eval_index)
    return {
        sat_name: build_sat_score_frame(
            sat_name=sat_name,
            frame=frame,
            eval_index=eval_index,
            icme_mask=icme_eval_mask,
            time_axis=time_axis,
            slow_sw_speed=slow_sw_speed,
            forecast_columns_by_sat=forecast_columns_by_sat,
        )
        for sat_name, frame in comparison_frames.items()
    }


def build_forecast_skill_df(
    score_frames,
    sat_labels,
    forecast_columns_by_sat=FORECAST_COLUMNS_BY_SAT,
):
    score_rows = []
    for sat_name, eval_frame in score_frames.items():
        sat_label = sat_labels.get(sat_name, sat_name)
        regime_masks = build_regime_masks(eval_frame)
        for regime in get_regime_order(sat_name):
            sample_mask = regime_masks[regime]
            forecast_columns = forecast_columns_by_sat.get(
                sat_name, DEFAULT_FORECAST_COLUMNS
            )
            for forecast_name, _column in forecast_columns:
                if forecast_name not in eval_frame.columns:
                    continue
                score_rows.append(
                    {
                        "sat": sat_label,
                        "regime": regime,
                        "forecast": forecast_name,
                        **compute_forecast_stats(
                            actual=eval_frame["obs"],
                            forecast=eval_frame[forecast_name],
                            sample_mask=sample_mask,
                        ),
                    }
                )

    forecast_skill_df = pd.DataFrame(score_rows)
    if forecast_skill_df.empty:
        empty = pd.DataFrame(
            columns=["n_samples", "r", "rmse", "mae", "bias"],
            index=pd.MultiIndex.from_tuples(
                [],
                names=["sat", "regime", "forecast"],
            ),
        )
        return empty

    forecast_skill_df["sat_order"] = forecast_skill_df["sat"].map(
        {label: order for order, label in enumerate(sat_labels.values())}
    )
    forecast_skill_df["regime_order"] = forecast_skill_df["regime"].map(
        {regime: order for order, regime in enumerate(REGIME_SORT_ORDER)}
    )
    forecast_skill_df["forecast_order"] = forecast_skill_df["forecast"].map(
        FORECAST_ORDER
    )
    return (
        forecast_skill_df.sort_values(
            ["sat_order", "regime_order", "forecast_order"]
        )
        .drop(columns=["sat_order", "regime_order", "forecast_order"])
        .set_index(["sat", "regime", "forecast"])
        .round({"r": 3, "rmse": 1, "mae": 1, "bias": 1})
    )


def build_pred_vs_noaa_df(score_frames, sat_labels):
    pred_vs_noaa_rows = []
    for sat_name, eval_frame in score_frames.items():
        if "noaa" not in eval_frame.columns or "pred" not in eval_frame.columns:
            continue
        sat_label = sat_labels.get(sat_name, sat_name)
        regime_masks = build_regime_masks(eval_frame)
        for regime in get_regime_order(sat_name):
            pred_vs_noaa_rows.append(
                {
                    "sat": sat_label,
                    "regime": regime,
                    **compute_forecast_stats(
                        actual=eval_frame["noaa"],
                        forecast=eval_frame["pred"],
                        sample_mask=regime_masks[regime],
                    ),
                }
            )

    pred_vs_noaa_df = pd.DataFrame(pred_vs_noaa_rows)
    if pred_vs_noaa_df.empty:
        empty = pd.DataFrame(
            columns=["n_samples", "r", "rmse", "mae", "bias"],
            index=pd.MultiIndex.from_tuples([], names=["sat", "regime"]),
        )
        return empty

    pred_vs_noaa_df["sat_order"] = pred_vs_noaa_df["sat"].map(
        {label: order for order, label in enumerate(sat_labels.values())}
    )
    pred_vs_noaa_df["regime_order"] = pred_vs_noaa_df["regime"].map(
        {regime: order for order, regime in enumerate(REGIME_SORT_ORDER)}
    )
    return (
        pred_vs_noaa_df.sort_values(["sat_order", "regime_order"])
        .drop(columns=["sat_order", "regime_order"])
        .set_index(["sat", "regime"])
        .round({"r": 3, "rmse": 1, "mae": 1, "bias": 1})
    )


def build_sw_forecast_stats(
    comparison_frames,
    start_dt,
    end_dt,
    time_axis,
    slow_sw_speed,
    sat_labels,
    freq=SCORE_FREQ,
):
    score_frames = build_score_frames(
        comparison_frames=comparison_frames,
        start_dt=start_dt,
        end_dt=end_dt,
        time_axis=time_axis,
        slow_sw_speed=slow_sw_speed,
        freq=freq,
    )
    return {
        "forecast_skill": build_forecast_skill_df(
            score_frames=score_frames,
            sat_labels=sat_labels,
        ),
        "pred_vs_noaa": build_pred_vs_noaa_df(
            score_frames=score_frames,
            sat_labels=sat_labels,
        ),
    }


def build_sw_forecast_stats_csv_frame(stats_tables):
    forecast_skill_df = stats_tables["forecast_skill"].reset_index()
    forecast_skill_df.insert(0, "table", "forecast_skill")

    pred_vs_noaa_df = stats_tables["pred_vs_noaa"].reset_index()
    pred_vs_noaa_df.insert(0, "table", "pred_vs_noaa")
    pred_vs_noaa_df.insert(3, "forecast", "pred_vs_noaa")

    return pd.concat([forecast_skill_df, pred_vs_noaa_df], ignore_index=True)


def export_sw_forecast_stats_csv(csv_outfile, **stats_kwargs):
    output_path = Path(csv_outfile)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats_tables = build_sw_forecast_stats(**stats_kwargs)
    csv_frame = build_sw_forecast_stats_csv_frame(stats_tables)
    csv_frame.to_csv(output_path, index=False)
    return stats_tables, csv_frame
