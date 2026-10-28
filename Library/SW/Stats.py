from pathlib import Path

import numpy as np
import pandas as pd
from sunpy.coordinates.sun import carrington_rotation_number, carrington_rotation_time

from Library.ICME import (
    DEFAULT_POST_ICME_BODY_END_TOLERANCE,
    build_icme_mask,
    load_icmecat_windows,
    load_icme_windows,
)
from Library.SW.Constants import CARRINGTON_ROTATION_DAYS
from Library.SW.Inputs import (
    load_ace_swx_frame,
    load_satellite_frame,
)

SCORE_FREQ = "1h"
HSS_EVENT_REFERENCES = [
    ("Yermolaev checked - Earth", "earth", "yermolaev_hss.csv"),
    ("B1 - ACE", "ace", "ace_hss_b1.csv"),
    ("B1 - STEREO-A", "stereo_a", "stereo-a_hss_b1.csv"),
    ("B1 - STEREO-B", "stereo_b", "stereo-b_hss_b1.csv"),
    ("B1 - PSP", "psp", "psp_hss_b1.csv"),
    ("B1 - Solar Orbiter", "solo", "solo_hss_b1.csv"),
]
RECURRENT_MAX_SOURCE_GAP = pd.Timedelta(hours=6)
REGIME_ORDER = ["all_sw", "no_icme"]
COMPARISON_ORDER = [
    "raw_vs_observed",
    "recurrent_vs_observed",
    "slow_model_vs_swx",
]
PAPER_COMPARISONS_BY_SAT = {
    "ace": [
        {
            "comparison": "raw_vs_observed",
            "reference": "v_real",
            "candidate": "v_predict_raw",
            "regimes": REGIME_ORDER,
        },
        {
            "comparison": "recurrent_vs_observed",
            "reference": "v_real",
            "candidate": "v_1cr_ago",
            "regimes": REGIME_ORDER,
        },
        {
            "comparison": "slow_model_vs_swx",
            "reference": "v_swx",
            "candidate": "v_predict",
            "regimes": ["all_sw"],
        },
    ],
    "ace_earth": [
        {
            "comparison": "raw_vs_observed",
            "reference": "v_real",
            "candidate": "v_predict_raw",
            "regimes": REGIME_ORDER,
        },
        {
            "comparison": "recurrent_vs_observed",
            "reference": "v_real",
            "candidate": "v_1cr_ago",
            "regimes": REGIME_ORDER,
        },
        {
            "comparison": "slow_model_vs_swx",
            "reference": "v_swx",
            "candidate": "v_predict",
            "regimes": ["all_sw"],
        },
    ],
    "stereo_a": [
        {
            "comparison": "raw_vs_observed",
            "reference": "v_real",
            "candidate": "v_predict_raw",
            "regimes": REGIME_ORDER,
        },
        {
            "comparison": "recurrent_vs_observed",
            "reference": "v_real",
            "candidate": "v_1cr_ago",
            "regimes": REGIME_ORDER,
        },
    ],
    "stereo_b": [
        {
            "comparison": "raw_vs_observed",
            "reference": "v_real",
            "candidate": "v_predict_raw",
            "regimes": REGIME_ORDER,
        },
        {
            "comparison": "recurrent_vs_observed",
            "reference": "v_real",
            "candidate": "v_1cr_ago",
            "regimes": REGIME_ORDER,
        },
    ],
    "psp": [
        {
            "comparison": "raw_vs_observed",
            "reference": "v_real",
            "candidate": "v_predict_raw",
            "regimes": REGIME_ORDER,
        },
        {
            "comparison": "recurrent_vs_observed",
            "reference": "v_real",
            "candidate": "v_1cr_ago",
            "regimes": REGIME_ORDER,
        },
    ],
    "solo": [
        {
            "comparison": "raw_vs_observed",
            "reference": "v_real",
            "candidate": "v_predict_raw",
            "regimes": REGIME_ORDER,
        },
        {
            "comparison": "recurrent_vs_observed",
            "reference": "v_real",
            "candidate": "v_1cr_ago",
            "regimes": REGIME_ORDER,
        },
    ],
}


def build_recurrent_series(
    observed_series,
    target_index,
    cr_days=CARRINGTON_ROTATION_DAYS,
    max_source_gap=RECURRENT_MAX_SOURCE_GAP,
):
    """Shift observations by one rotation, bridging only short source-data gaps."""
    source = pd.Series(
        pd.to_numeric(observed_series, errors="coerce").to_numpy(),
        index=pd.DatetimeIndex(observed_series.index),
        name="v_1cr_ago",
    )
    source = (
        source[~source.index.duplicated(keep="last")]
        .sort_index()
        .resample(SCORE_FREQ)
        .mean()
        .dropna()
    )
    source.index = source.index + pd.Timedelta(days=float(cr_days))

    target_index = pd.DatetimeIndex(target_index)
    combined_index = source.index.union(target_index).sort_values()
    source_times = pd.Series(source.index, index=source.index)
    previous_time = source_times.reindex(combined_index).ffill()
    next_time = source_times.reindex(combined_index).bfill()
    bounded = (next_time - previous_time) <= pd.Timedelta(max_source_gap)
    interpolated = source.reindex(combined_index).interpolate(method="time")
    return interpolated.reindex(target_index).where(bounded.reindex(target_index))


def build_eval_index(start_dt, end_dt, freq=SCORE_FREQ):
    return pd.date_range(
        pd.Timestamp(start_dt), pd.Timestamp(end_dt), freq=freq, inclusive="left"
    )


def restore_observed_and_recurrent_series(
    comparison_frames,
    start_dt,
    end_dt,
    freq=SCORE_FREQ,
):
    """Replace gap-filled parquet series with native cached observations."""
    eval_index = build_eval_index(start_dt=start_dt, end_dt=end_dt, freq=freq)
    source_index = pd.date_range(
        (
            pd.Timestamp(start_dt)
            - pd.Timedelta(days=CARRINGTON_ROTATION_DAYS)
        ).floor(freq),
        pd.Timestamp(end_dt),
        freq=freq,
        inclusive="left",
    )
    observed_frames = {
        sat_name: load_satellite_frame(
            sat_id=sat_name,
            time_axis=source_index,
            time_freq=freq,
        )
        for sat_name in comparison_frames
    }

    restored_frames = {}
    for sat_name, frame in comparison_frames.items():
        assert sat_name in observed_frames, (
            f"No native observation loader configured for satellite: {sat_name}"
        )
        observed_frame = observed_frames[sat_name]
        observed = observed_frame["v"].resample(freq).mean().reindex(eval_index)
        recurrent = build_recurrent_series(
            observed_series=observed_frame["v"],
            target_index=eval_index,
        )

        restored = frame.copy()
        restored["v_real"] = observed.reindex(restored.index)
        restored["v_1cr_ago"] = recurrent.reindex(restored.index)
        observation_columns = [
            column
            for column in (
                "N",
                "t",
                "b",
                "b_x_gse",
                "b_y_gse",
                "b_z_gse",
                "b_r_rtn",
                "b_t_rtn",
                "b_n_rtn",
                "v_x_gse",
                "v_y_gse",
                "v_z_gse",
                "x_hee_au",
                "y_hee_au",
                "z_hee_au",
                "lat_hee",
                "phi_target",
                "r_target",
            )
            if column in observed_frame.columns
        ]
        if observation_columns:
            restored = restored.drop(
                columns=[column for column in observation_columns if column in restored.columns]
            )
            restored = restored.join(observed_frame[observation_columns], how="left")
        restored_frames[sat_name] = restored

    return restored_frames


def restore_swx_series(comparison_frames, swx_path=None):
    """Replace the old gap-filled SWX series with the frozen native forecast."""
    restored_frames = {
        sat_name: frame.copy() for sat_name, frame in comparison_frames.items()
    }
    ace_sat = "ace" if "ace" in restored_frames else "ace_earth"
    if ace_sat not in restored_frames:
        return restored_frames

    ace_frame = restored_frames[ace_sat]
    if "v_swx" in ace_frame.columns:
        ace_frame = ace_frame.drop(columns="v_swx")
    swx_frame = (
        load_ace_swx_frame()
        if swx_path is None
        else load_ace_swx_frame(swx_path)
    )
    restored_frames[ace_sat] = ace_frame.join(
        swx_frame[["v_swx"]], how="outer"
    ).sort_index()
    return restored_frames


def prepare_eval_series(series, eval_index, output_name, freq=SCORE_FREQ):
    prepared = pd.Series(
        pd.to_numeric(series, errors="coerce").to_numpy(),
        index=pd.DatetimeIndex(series.index),
        name=output_name,
    )
    prepared = prepared[~prepared.index.duplicated(keep="last")].sort_index()
    return prepared.resample(freq).mean().reindex(eval_index)


def prepare_eval_mask(series, eval_index, output_name, freq=SCORE_FREQ):
    mask_source = pd.Series(series)
    mask = mask_source.where(mask_source.notna(), False).astype(bool)
    prepared = pd.Series(
        mask.to_numpy(),
        index=pd.DatetimeIndex(series.index),
        name=output_name,
    )
    prepared = prepared[~prepared.index.duplicated(keep="last")].sort_index()
    return prepared.resample(freq).max().reindex(eval_index, fill_value=False)


def load_sat_icme_windows(sat_name):
    if sat_name in {"ace", "ace_earth"}:
        return load_icme_windows()
    if sat_name == "stereo_a":
        return load_icmecat_windows("STEREO-A")
    if sat_name in {"stereo_b", "psp", "solo"}:
        return pd.DataFrame(
            {
                "start": pd.Series(dtype="datetime64[ns]"),
                "end": pd.Series(dtype="datetime64[ns]"),
            }
        )
    raise ValueError(f"No ICME catalogue configured for satellite: {sat_name}")


def build_sat_score_frame(sat_name, frame, eval_index, freq=SCORE_FREQ):
    comparison_specs = PAPER_COMPARISONS_BY_SAT[sat_name]
    if any(
        spec["comparison"] == "slow_model_vs_swx" for spec in comparison_specs
    ):
        assert "slow_sw_patch_mask" in frame.columns, (
            "The slow_model_vs_swx comparison requires a reproduction built "
            "with the ACE slow-wind patch enabled"
        )
    required_columns = {
        column
        for spec in comparison_specs
        for column in [spec["reference"], spec["candidate"]]
    }
    missing_columns = required_columns.difference(frame.columns)
    assert not missing_columns, (
        f"Missing paper comparison columns for {sat_name}: "
        f"{sorted(missing_columns)}"
    )

    out = pd.DataFrame(index=eval_index)
    for column in sorted(required_columns):
        out[column] = prepare_eval_series(
            frame[column], eval_index, output_name=column, freq=freq
        )
    if "slow_sw_patch_mask" in frame.columns:
        out["slow_sw_patch_mask"] = prepare_eval_mask(
            frame["slow_sw_patch_mask"],
            eval_index,
            output_name="slow_sw_patch_mask",
            freq=freq,
        )

    out["is_icme"] = build_icme_mask(
        eval_index,
        icme_windows=load_sat_icme_windows(sat_name),
        inclusive_end=False,
        sample_width=pd.Timedelta(freq),
    ).to_numpy()
    return out


def build_score_frames(comparison_frames, start_dt, end_dt, freq=SCORE_FREQ):
    eval_index = build_eval_index(start_dt, end_dt, freq=freq)
    return {
        sat_name: build_sat_score_frame(
            sat_name=sat_name,
            frame=frame,
            eval_index=eval_index,
            freq=freq,
        )
        for sat_name, frame in comparison_frames.items()
    }


def build_regime_masks(eval_frame):
    return {
        "all_sw": pd.Series(True, index=eval_frame.index),
        "no_icme": ~eval_frame["is_icme"],
    }


def compute_comparison_stats(reference, candidate, sample_mask):
    """Compute metrics with bias defined as candidate minus reference."""
    paired = pd.concat(
        [
            pd.to_numeric(reference, errors="coerce").rename("reference"),
            pd.to_numeric(candidate, errors="coerce").rename("candidate"),
            sample_mask.rename("sample_mask"),
        ],
        axis=1,
    )
    paired = paired.loc[paired["sample_mask"]].dropna(
        subset=["reference", "candidate"]
    )
    if paired.empty:
        return {
            "n_samples": 0,
            "r": np.nan,
            "rmse": np.nan,
            "mae": np.nan,
            "bias": np.nan,
        }

    error = paired["candidate"] - paired["reference"]
    r_value = np.nan
    if len(paired) >= 2:
        reference_std = float(paired["reference"].std())
        candidate_std = float(paired["candidate"].std())
        if reference_std > 0.0 and candidate_std > 0.0:
            r_value = float(paired["reference"].corr(paired["candidate"]))

    return {
        "n_samples": int(len(paired)),
        "r": r_value,
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "bias": float(np.mean(error)),
    }


def build_sw_forecast_stats(
    comparison_frames,
    start_dt,
    end_dt,
    sat_labels,
    freq=SCORE_FREQ,
):
    score_frames = build_score_frames(
        comparison_frames=comparison_frames,
        start_dt=start_dt,
        end_dt=end_dt,
        freq=freq,
    )
    return _build_stats_from_score_frames(score_frames, sat_labels)


def _build_stats_from_score_frames(score_frames, sat_labels):
    """Score prepared frames with the shared comparisons and ICME regimes."""
    rows = []
    for sat_name, eval_frame in score_frames.items():
        regime_masks = build_regime_masks(eval_frame)
        for spec in PAPER_COMPARISONS_BY_SAT[sat_name]:
            for regime in spec["regimes"]:
                rows.append(
                    {
                        "sat": sat_labels.get(sat_name, sat_name),
                        "comparison": spec["comparison"],
                        "regime": regime,
                        "reference": spec["reference"],
                        "candidate": spec["candidate"],
                        **compute_comparison_stats(
                            reference=eval_frame[spec["reference"]],
                            candidate=eval_frame[spec["candidate"]],
                            sample_mask=regime_masks[regime],
                        ),
                    }
                )

    stats = pd.DataFrame(rows)
    stats["sat_order"] = stats["sat"].map(
        {label: order for order, label in enumerate(sat_labels.values())}
    )
    stats["comparison_order"] = stats["comparison"].map(
        {name: order for order, name in enumerate(COMPARISON_ORDER)}
    )
    stats["regime_order"] = stats["regime"].map(
        {name: order for order, name in enumerate(REGIME_ORDER)}
    )
    return (
        stats.sort_values(["sat_order", "comparison_order", "regime_order"])
        .drop(columns=["sat_order", "comparison_order", "regime_order"])
        .reset_index(drop=True)
        .round({"r": 3, "rmse": 1, "mae": 1, "bias": 1})
    )


def export_sw_forecast_stats_csv(csv_outfile, **stats_kwargs):
    output_path = Path(csv_outfile)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats = build_sw_forecast_stats(**stats_kwargs)
    stats.to_csv(output_path, index=False)
    return stats


def build_sw_forecast_yearly_stats(
    comparison_frames,
    start_dt,
    end_dt,
    sat_labels,
    freq=SCORE_FREQ,
):
    """Score calendar years from hourly pairs, clipping partial years to the run."""
    start, end = pd.Timestamp(start_dt), pd.Timestamp(end_dt)
    assert start < end, f"Expected start before end, got [{start}, {end})"
    score_frames = build_score_frames(
        comparison_frames=comparison_frames,
        start_dt=start,
        end_dt=end,
        freq=freq,
    )
    frames = []
    year_start = start
    while year_start < end:
        year_end = min(
            pd.Timestamp(year=year_start.year + 1, month=1, day=1, tz=start.tz), end
        )
        year_frames = {
            sat: frame.loc[(frame.index >= year_start) & (frame.index < year_end)]
            for sat, frame in score_frames.items()
        }
        stats = _build_stats_from_score_frames(year_frames, sat_labels)
        stats.insert(0, "end_utc", year_end)
        stats.insert(0, "start_utc", year_start)
        stats.insert(0, "year", year_start.year)
        frames.append(stats)
        year_start = year_end
    return pd.concat(frames, ignore_index=True)


def build_cr_windows(start_dt, end_dt):
    start = pd.Timestamp(start_dt)
    end = pd.Timestamp(end_dt)
    rows = []
    cr = int(np.floor(carrington_rotation_number(start)))
    while True:
        full_cr_start = pd.Timestamp(carrington_rotation_time(cr).datetime)
        full_cr_end = pd.Timestamp(carrington_rotation_time(cr + 1).datetime)
        cr_start = max(start, full_cr_start)
        cr_end = min(end, full_cr_end)
        if cr_start < cr_end:
            rows.append({"cr": cr, "cr_start": cr_start, "cr_end": cr_end})
        if full_cr_end >= end:
            break
        cr += 1
    return pd.DataFrame(rows)


def build_cr_sample_mask(eval_frame, cr_start, cr_end):
    mask = (eval_frame.index >= cr_start) & (eval_frame.index < cr_end)
    return pd.Series(mask, index=eval_frame.index)


def build_sw_forecast_cr_stats_csv_frame(
    comparison_frames,
    start_dt,
    end_dt,
    sat_labels,
    freq=SCORE_FREQ,
):
    score_frames = build_score_frames(
        comparison_frames=comparison_frames,
        start_dt=start_dt,
        end_dt=end_dt,
        freq=freq,
    )
    cr_windows = build_cr_windows(start_dt=start_dt, end_dt=end_dt)
    rows = []
    for cr_window in cr_windows.itertuples(index=False):
        for sat_name, eval_frame in score_frames.items():
            cr_mask = build_cr_sample_mask(
                eval_frame=eval_frame,
                cr_start=cr_window.cr_start,
                cr_end=cr_window.cr_end,
            )
            regime_masks = build_regime_masks(eval_frame)
            for spec in PAPER_COMPARISONS_BY_SAT[sat_name]:
                for regime in spec["regimes"]:
                    rows.append(
                        {
                            "cr": cr_window.cr,
                            "cr_start": cr_window.cr_start,
                            "cr_end": cr_window.cr_end,
                            "sat": sat_labels.get(sat_name, sat_name),
                            "comparison": spec["comparison"],
                            "regime": regime,
                            "reference": spec["reference"],
                            "candidate": spec["candidate"],
                            **compute_comparison_stats(
                                reference=eval_frame[spec["reference"]],
                                candidate=eval_frame[spec["candidate"]],
                                sample_mask=cr_mask & regime_masks[regime],
                            ),
                        }
                    )

    stats = pd.DataFrame(rows)
    stats["sat_order"] = stats["sat"].map(
        {label: order for order, label in enumerate(sat_labels.values())}
    )
    stats["comparison_order"] = stats["comparison"].map(
        {name: order for order, name in enumerate(COMPARISON_ORDER)}
    )
    stats["regime_order"] = stats["regime"].map(
        {name: order for order, name in enumerate(REGIME_ORDER)}
    )
    return (
        stats.sort_values(
            ["cr", "sat_order", "comparison_order", "regime_order"]
        )
        .drop(columns=["sat_order", "comparison_order", "regime_order"])
        .reset_index(drop=True)
        .round({"r": 3, "rmse": 1, "mae": 1, "bias": 1})
    )


def export_sw_forecast_cr_stats_csv(csv_outfile, **stats_kwargs):
    output_path = Path(csv_outfile)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    stats = build_sw_forecast_cr_stats_csv_frame(**stats_kwargs)
    stats.to_csv(output_path, index=False)
    return stats


def build_icme_duration_by_cr(cr_stats, events, sat):
    """Measure physical ICME occupancy in each CR without double-counting."""
    sat_stats = cr_stats.loc[cr_stats["sat"] == sat].copy()
    assert not sat_stats.empty, f"No per-rotation statistics found for sat={sat!r}"
    cr_windows = sat_stats[["cr", "cr_start", "cr_end"]].drop_duplicates()
    assert not cr_windows["cr"].duplicated().any(), (
        f"Multiple time windows found for one Carrington rotation at sat={sat!r}"
    )
    cr_windows["cr"] = pd.to_numeric(cr_windows["cr"], errors="coerce")
    cr_windows["cr_start"] = pd.to_datetime(cr_windows["cr_start"])
    cr_windows["cr_end"] = pd.to_datetime(cr_windows["cr_end"])
    cr_windows = cr_windows.dropna().sort_values("cr")

    sat_events = events.loc[events["sat"] == sat].copy()
    sat_events["event_start"] = pd.to_datetime(sat_events["event_start"])
    sat_events["event_end"] = pd.to_datetime(sat_events["event_end"])
    sat_events = sat_events.dropna(subset=["event_start", "event_end"])

    rows = []
    for cr_window in cr_windows.itertuples(index=False):
        clipped_intervals = []
        for event in sat_events.itertuples(index=False):
            overlap_start = max(cr_window.cr_start, event.event_start)
            overlap_end = min(cr_window.cr_end, event.event_end)
            if overlap_start < overlap_end:
                clipped_intervals.append((overlap_start, overlap_end))

        merged_intervals = []
        for interval_start, interval_end in sorted(clipped_intervals):
            if merged_intervals and interval_start <= merged_intervals[-1][1]:
                previous_start, previous_end = merged_intervals[-1]
                merged_intervals[-1] = (
                    previous_start,
                    max(previous_end, interval_end),
                )
            else:
                merged_intervals.append((interval_start, interval_end))

        rows.append(
            {
                "cr": cr_window.cr,
                "cr_start": cr_window.cr_start,
                "cr_end": cr_window.cr_end,
                "icme_duration_hours": sum(
                    (interval_end - interval_start) / pd.Timedelta(hours=1)
                    for interval_start, interval_end in merged_intervals
                ),
            }
        )

    return pd.DataFrame(rows)


def build_icme_event_df(start_dt, end_dt, sat_labels, satellite_ids=None):
    """Build event-level ICME context, with Dst populated only at ACE/Earth."""
    start = pd.Timestamp(start_dt)
    end = pd.Timestamp(end_dt)
    output_columns = [
        "event_id",
        "catalog",
        "sat",
        "event_start",
        "mo_start",
        "event_end",
        "exclusion_end",
        "event_midpoint",
        "event_cr",
        "icme_duration_hours",
        "dst_min",
    ]

    rows = []
    if satellite_ids is None:
        satellite_ids = list(sat_labels)
    for sat_name in satellite_ids:
        icme_windows = load_sat_icme_windows(sat_name)
        overlapping_icmes = icme_windows.loc[
            (icme_windows["start"] < end) & (icme_windows["end"] > start)
        ]
        for event in overlapping_icmes.itertuples(index=False):
            event_start = pd.Timestamp(event.start)
            event_end = pd.Timestamp(event.end)
            event_midpoint = event_start + (event_end - event_start) / 2
            if sat_name in {"ace", "ace_earth"}:
                event_id = f"earth_{event_start:%Y%m%d_%H%M}"
                mo_start = pd.Timestamp(event.T_start)
                dst_min = pd.to_numeric(event.dst_min_omni_body, errors="coerce")
                catalog = "unified_earth"
            else:
                event_id = event.icmecat_id
                mo_start = pd.Timestamp(event.mo_start_time)
                dst_min = np.nan
                catalog = "icmecat_v2.3"

            rows.append(
                {
                    "event_id": event_id,
                    "catalog": catalog,
                    "sat": sat_labels.get(sat_name, sat_name),
                    "event_start": event_start,
                    "mo_start": mo_start,
                    "event_end": event_end,
                    "exclusion_end": (
                        event_end + DEFAULT_POST_ICME_BODY_END_TOLERANCE
                    ),
                    "event_midpoint": event_midpoint,
                    "event_cr": float(carrington_rotation_number(event_midpoint)),
                    "icme_duration_hours": (
                        event_end - event_start
                    ) / pd.Timedelta(hours=1),
                    "dst_min": dst_min,
                }
            )

    return (
        pd.DataFrame(rows, columns=output_columns)
        .sort_values(["sat", "event_start"])
        .reset_index(drop=True)
    )


def export_icme_event_csv(csv_outfile, **event_kwargs):
    output_path = Path(csv_outfile)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    events = build_icme_event_df(**event_kwargs)
    events.to_csv(output_path, index=False)
    return events


def load_hss_event_catalogs(hss_list_dir, start_dt, end_dt):
    """Select reference events by their original UTC start, ignoring ICME flags."""
    catalogs = {}
    for sheet_name, _satellite, filename in HSS_EVENT_REFERENCES:
        events = pd.read_csv(
            Path(hss_list_dir) / filename, usecols=["T_start", "T_end", "v_max"]
        )
        for column in ["T_start", "T_end"]:
            events[column] = pd.to_datetime(events[column], utc=True).dt.tz_localize(None)
        catalogs[sheet_name] = events.loc[
            (events.T_start >= pd.Timestamp(start_dt))
            & (events.T_start < pd.Timestamp(end_dt))
        ].reset_index(drop=True)
    return catalogs


def score_hss_events(events, predicted_speed):
    """Score padded observed intervals against raw hourly model speed."""
    prediction = pd.Series(
        pd.to_numeric(predicted_speed).to_numpy(dtype=float),
        index=pd.to_datetime(predicted_speed.index, utc=True).tz_localize(None),
    ).sort_index().resample(SCORE_FREQ).mean()
    scored = events.copy()
    scored["n_scored"] = False
    scored["hit"] = False
    scored["miss"] = False
    scored["miss_no_ch"] = False
    scored["v_pred_peak"] = np.nan
    scored["t_pred_peak"] = pd.NaT
    scored["underreporting_km_s"] = np.nan
    padding = pd.Timedelta(hours=12)
    for event in events.itertuples():
        index = event.Index
        hours = pd.date_range(
            event.T_start - padding, event.T_end + padding, freq=SCORE_FREQ
        )
        window = prediction.reindex(hours)
        if not np.isfinite(window.to_numpy()).all():
            continue
        peak = float(window.max())
        scored.loc[index, "n_scored"] = True
        scored.loc[index, "hit"] = peak > 400.0
        scored.loc[index, "miss"] = peak <= 400.0
        scored.loc[index, "miss_no_ch"] = window.eq(300.0).all()
        scored.loc[index, "v_pred_peak"] = peak
        scored.loc[index, "t_pred_peak"] = window.idxmax()
        scored.loc[index, "underreporting_km_s"] = event.v_max - peak
    return scored


def build_hss_yearly_stats(catalogs, prediction_series, start_dt, end_dt):
    """Aggregate event counts and signed peak shortfalls by event-start year."""
    last_year = (pd.Timestamp(end_dt) - pd.Timedelta(nanoseconds=1)).year
    years = pd.Index(
        range(pd.Timestamp(start_dt).year, last_year + 1),
        name="year",
    )
    count_columns = [
        "n_events", "n_scored", "hit", "miss", "miss_no_ch", "n_underreporting"
    ]
    tables = {}
    for sheet_name, satellite, _filename in HSS_EVENT_REFERENCES:
        events = score_hss_events(
            catalogs[sheet_name], prediction_series.get(satellite, pd.Series(dtype=float))
        )
        events["year"] = events.T_start.dt.year
        yearly = events.groupby("year").agg(
            n_events=("T_start", "size"),
            n_scored=("n_scored", "sum"),
            hit=("hit", "sum"),
            miss=("miss", "sum"),
            miss_no_ch=("miss_no_ch", "sum"),
            n_underreporting=("underreporting_km_s", "count"),
            mean_underreporting_km_s=("underreporting_km_s", "mean"),
        ).reindex(years)
        yearly[count_columns] = yearly[count_columns].fillna(0).astype(int)
        yearly.insert(2, "n_unscored", yearly.n_events - yearly.n_scored)
        for column in ["hit", "miss", "miss_no_ch"]:
            yearly[f"{column}_rate"] = yearly[column] / yearly.n_scored.replace(0, np.nan)
        tables[sheet_name] = yearly.reset_index()[
            [
                "year", "n_events", "n_scored", "n_unscored", "hit", "miss",
                "miss_no_ch", "hit_rate", "miss_rate", "miss_no_ch_rate",
                "n_underreporting", "mean_underreporting_km_s",
            ]
        ]
    return tables
