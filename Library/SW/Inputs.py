from pathlib import Path

import numpy as np
import pandas as pd
import psycopg

import userpwd
from Library.Paths import data_path, resolve_repo_path

DEFAULT_SQL_QUERY = """
SELECT
  f.dt AS dt,
  f.forecast_dt,
  f.forecast_sw_speed,
  c.ch_relative_correct_sphere_area AS ch_relative_area
FROM SDO_PREFIX.sdo_sw_forecast_0193p AS f
LEFT JOIN SDO_PREFIX.sdo_fill_sw_193(
  48,
  '1h',
  %(start_dt)s - interval '5 days',
  %(end_dt)s
) AS c
  ON c.dt = f.dt
WHERE f.forecast_dt BETWEEN %(start_dt)s AND %(end_dt)s
ORDER BY f.forecast_dt, f.dt;
"""

SDO_V2_HANDOFF = "2019-01-01"
DEFAULT_SQL_CONNECTION = {
    "host": "213.131.1.41",
    "user": "selector",
    "dbname": "smdc",
}

DEFAULT_INPUT_PARQUET_PATH = data_path("CH Area.parquet")
DEFAULT_ACE_PARQUET_PATH = data_path("ACE At Earth 1h.parquet")
DEFAULT_STEREO_A_PARQUET_PATH = data_path("STEREO-A Plastic.parquet")
DEFAULT_ENLIL_PARQUET_PATH = data_path("ENLIL 2018-02-01 2018-07-01.parquet")
DEFAULT_ACE_EARTH_SAT = "ace_earth"
DEFAULT_ACE_EARTH_LABEL = "ACE @ Earth"
DEFAULT_STEREO_A_SAT = "stereo_a"
DEFAULT_STEREO_A_LABEL = "STEREO-A"
EARTH_RADII_PER_SOLAR_RADIUS = 109.0763707060096
SATELLITE_FRAME_COLUMNS = [
    "v",
    "phi_target",
    "r_target",
    "lat_hgs",
    "lat_hge",
    "v_swx",
]


def iter_year_windows(start_dt, end_dt):
    chunk_start = pd.Timestamp(start_dt)
    chunk_end_limit = pd.Timestamp(end_dt)
    while chunk_start < chunk_end_limit:
        next_year = pd.Timestamp(year=chunk_start.year + 1, month=1, day=1)
        chunk_end = min(next_year, chunk_end_limit)
        yield chunk_start, chunk_end
        chunk_start = chunk_end


def load_sw_input_from_parquet(input_parquet_path=DEFAULT_INPUT_PARQUET_PATH):
    parquet_path = resolve_repo_path(input_parquet_path)
    assert parquet_path.exists(), f"Missing input parquet: {parquet_path}"
    return pd.read_parquet(parquet_path).copy()


def load_sw_input_from_sql(
    start_dt,
    end_dt,
    query=None,
    connection_kwargs=None,
    password=None,
):
    conn_kwargs = dict(DEFAULT_SQL_CONNECTION)
    if connection_kwargs is not None:
        conn_kwargs.update(connection_kwargs)

    conn = psycopg.connect(
        password=userpwd.userpwd_postgre if password is None else password,
        **conn_kwargs,
    )
    frames = []
    columns = None
    with conn:
        with conn.cursor() as cur:
            for chunk_start, chunk_end in iter_year_windows(start_dt, end_dt):
                sdo_prefix = (
                    "sdo" if chunk_start < pd.Timestamp(SDO_V2_HANDOFF) else "sdo_v2"
                )
                chunk_query = query if query is not None else DEFAULT_SQL_QUERY
                chunk_query = chunk_query.replace("SDO_PREFIX", sdo_prefix)
                cur.execute(
                    chunk_query,
                    {"start_dt": chunk_start, "end_dt": chunk_end},
                )
                rows = cur.fetchall()
                if columns is None:
                    columns = [desc.name for desc in cur.description]
                frames.append(pd.DataFrame(rows, columns=columns))
    conn.close()

    if not frames:
        return pd.DataFrame(columns=columns)
    return pd.concat(frames, ignore_index=True)


def normalize_sw_input_frame(df_input_raw, start_dt, end_dt):
    df_sdo_sw = df_input_raw.copy()
    df_sdo_sw["dt"] = pd.to_datetime(df_sdo_sw["dt"])

    if "forecast_dt" in df_sdo_sw.columns:
        df_sdo_sw["forecast_dt"] = pd.to_datetime(df_sdo_sw["forecast_dt"])

    if "ch_relative_area" not in df_sdo_sw.columns:
        if "ch_area_1" in df_sdo_sw.columns:
            df_sdo_sw["ch_relative_area"] = pd.to_numeric(
                df_sdo_sw["ch_area_1"], errors="coerce"
            )
        elif "ch_area" in df_sdo_sw.columns:
            df_sdo_sw["ch_relative_area"] = pd.to_numeric(
                df_sdo_sw["ch_area"], errors="coerce"
            )
    else:
        df_sdo_sw["ch_relative_area"] = pd.to_numeric(
            df_sdo_sw["ch_relative_area"], errors="coerce"
        )

    if "forecast_sw_speed" not in df_sdo_sw.columns:
        if {"sw_speed_1", "sw_speed_2"}.issubset(df_sdo_sw.columns):
            sw_speed_1 = pd.to_numeric(df_sdo_sw["sw_speed_1"], errors="coerce")
            sw_speed_2 = pd.to_numeric(df_sdo_sw["sw_speed_2"], errors="coerce")
            df_sdo_sw["forecast_sw_speed"] = sw_speed_1.fillna(sw_speed_2)
        elif "sw_speed_1" in df_sdo_sw.columns:
            df_sdo_sw["forecast_sw_speed"] = pd.to_numeric(
                df_sdo_sw["sw_speed_1"], errors="coerce"
            )
    else:
        df_sdo_sw["forecast_sw_speed"] = pd.to_numeric(
            df_sdo_sw["forecast_sw_speed"], errors="coerce"
        )

    window_column = "forecast_dt" if "forecast_dt" in df_sdo_sw.columns else "dt"
    df_sdo_sw = df_sdo_sw[
        (df_sdo_sw[window_column] >= start_dt) & (df_sdo_sw[window_column] < end_dt)
    ].copy()
    return df_sdo_sw.sort_values("dt").reset_index(drop=True)


def load_sw_input_frame(
    start_dt,
    end_dt,
    source="parquet",
    input_parquet_path=DEFAULT_INPUT_PARQUET_PATH,
    query=None,
    connection_kwargs=None,
    password=None,
):
    if source == "parquet":
        df_input_raw = load_sw_input_from_parquet(input_parquet_path)
    elif source == "sql":
        df_input_raw = load_sw_input_from_sql(
            start_dt=start_dt,
            end_dt=end_dt,
            query=query,
            connection_kwargs=connection_kwargs,
            password=password,
        )
    else:
        raise ValueError(f"Unsupported SW input source: {source}")

    return normalize_sw_input_frame(df_input_raw, start_dt=start_dt, end_dt=end_dt)


def normalize_satellite_frame(df_sat_raw, sat, label=None):
    df_sat = df_sat_raw.copy()

    if not isinstance(df_sat.index, pd.DatetimeIndex):
        time_column = next(
            (column for column in ("time", "date", "dt") if column in df_sat.columns),
            None,
        )
        assert time_column is not None, "Satellite frame must be time-indexed"
        df_sat = df_sat.set_index(time_column)

    df_sat.index = pd.to_datetime(df_sat.index)
    rename_map = {}
    if "v" not in df_sat.columns:
        for candidate in ("speed", "v_ace", "v_real"):
            if candidate in df_sat.columns:
                rename_map[candidate] = "v"
                break
    if "v_swx" not in df_sat.columns and "forecast_sw_speed" in df_sat.columns:
        rename_map["forecast_sw_speed"] = "v_swx"
    df_sat = df_sat.rename(columns=rename_map)

    keep_columns = [
        column for column in SATELLITE_FRAME_COLUMNS if column in df_sat.columns
    ]
    df_sat = df_sat[keep_columns].sort_index()
    df_sat = df_sat[~df_sat.index.duplicated(keep="last")]

    for column in keep_columns:
        df_sat[column] = pd.to_numeric(df_sat[column], errors="coerce")

    df_sat.attrs["sat"] = str(sat)
    df_sat.attrs["label"] = str(label) if label is not None else str(sat)
    return df_sat


def load_cached_satellite_frame(path, sat, label=None):
    sat_path = resolve_repo_path(path)
    df_sat_raw = pd.read_parquet(sat_path).copy()
    return normalize_satellite_frame(df_sat_raw, sat=sat, label=label)


def load_ace_earth_frame(ace_path=DEFAULT_ACE_PARQUET_PATH):
    return load_cached_satellite_frame(
        ace_path,
        sat=DEFAULT_ACE_EARTH_SAT,
        label=DEFAULT_ACE_EARTH_LABEL,
    )


def load_stereo_a_frame(
    time_axis,
    time_freq,
    stereo_path=DEFAULT_STEREO_A_PARQUET_PATH,
):
    stereo_path = resolve_repo_path(stereo_path)
    stereo_a_df = pd.read_parquet(
        stereo_path,
        columns=[
            "V",
            "radialDistance",
            "heliographicLatitude",
            "heliographicLongitude",
        ],
    ).copy()
    stereo_a_df.index = pd.to_datetime(stereo_a_df.index, utc=True).tz_convert(None)
    stereo_a_df = stereo_a_df.loc[
        pd.Timestamp(time_axis.min()) : pd.Timestamp(time_axis.max())
    ]
    stereo_a_df = stereo_a_df.rename(
        columns={
            "V": "v",
            "radialDistance": "r_target",
            "heliographicLatitude": "lat_hge",
            "heliographicLongitude": "phi_target",
        }
    )
    for column in stereo_a_df.columns:
        stereo_a_df[column] = pd.to_numeric(stereo_a_df[column], errors="coerce")
    stereo_a_df["r_target"] = (
        stereo_a_df["r_target"] / EARTH_RADII_PER_SOLAR_RADIUS
    )
    stereo_a_df = stereo_a_df.resample(time_freq).mean().reindex(time_axis)
    stereo_a_df.attrs["sat"] = DEFAULT_STEREO_A_SAT
    stereo_a_df.attrs["label"] = DEFAULT_STEREO_A_LABEL
    stereo_a_df.attrs["coord_frame"] = "HGE"
    return stereo_a_df


def load_enlil_prediction_frames(
    time_axis,
    time_freq,
    enlil_path=DEFAULT_ENLIL_PARQUET_PATH,
    lead_days=5.0,
    lead_tolerance=pd.Timedelta(hours=12),
):
    if enlil_path is None:
        enlil_path = DEFAULT_ENLIL_PARQUET_PATH
    enlil_path = resolve_repo_path(enlil_path)
    enlil_raw = pd.read_parquet(
        enlil_path,
        columns=["time", "run_id", "Earth_V1", "STEREO_A_V1"],
    )
    enlil_raw["time"] = pd.to_datetime(
        enlil_raw["time"], utc=True
    ).dt.tz_convert(None)
    enlil_raw["issue_dt"] = pd.to_datetime(
        enlil_raw["run_id"].str.extract(r"_(\d{8})_\d{4}$", expand=False),
        format="%Y%m%d",
    )
    enlil_raw["lead"] = enlil_raw["time"] - enlil_raw["issue_dt"]

    target_lead = pd.Timedelta(days=float(lead_days))
    enlil_selected = enlil_raw.loc[
        (enlil_raw["lead"] >= target_lead - lead_tolerance)
        & (enlil_raw["lead"] <= target_lead + lead_tolerance)
    ].copy()
    enlil_selected["lead_err"] = (enlil_selected["lead"] - target_lead).abs()
    enlil_selected = (
        enlil_selected.sort_values(["time", "lead_err"])
        .drop_duplicates("time", keep="first")
        .set_index("time")
        .sort_index()
    )

    def build_enlil_frame(velocity_column):
        series = (
            pd.to_numeric(enlil_selected[velocity_column], errors="coerce") / 1000.0
        )
        series = series.loc[
            pd.Timestamp(time_axis.min()) : pd.Timestamp(time_axis.max())
        ]
        series = (
            series.resample(time_freq)
            .mean()
            .reindex(time_axis)
            .interpolate(method="time")
        )
        return pd.DataFrame({"v_noaa": series})

    return {
        DEFAULT_ACE_EARTH_SAT: build_enlil_frame("Earth_V1"),
        DEFAULT_STEREO_A_SAT: build_enlil_frame("STEREO_A_V1"),
    }


def build_model_input_series(
    sdo_input_df,
    empirical,
    superresolution_enabled,
    time_freq,
    simulation_pad_days,
):
    required_cols = {"dt", "ch_relative_area"}
    assert required_cols.issubset(
        sdo_input_df.columns
    ), "Expected SDO input dataframe to include dt and ch_relative_area columns"

    prepared_input = sdo_input_df.copy()
    prepared_input["dt"] = pd.to_datetime(prepared_input["dt"])
    prepared_input["ch_relative_area"] = pd.to_numeric(
        prepared_input["ch_relative_area"], errors="coerce"
    )
    prepared_input = prepared_input.dropna(subset=["dt", "ch_relative_area"])
    prepared_input = prepared_input.sort_values("dt")
    assert (
        len(prepared_input) > 0
    ), "No valid SW input rows remain after filtering and CH-area normalization"

    launch_time = (prepared_input["dt"] + pd.Timedelta(minutes=30)).dt.floor("1h")
    prepared_input["v_empirical"] = empirical.v_from_area(
        prepared_input["ch_relative_area"].to_numpy(dtype=float),
        t=launch_time,
    )
    df_v = (
        pd.DataFrame({"time": launch_time, "v": prepared_input["v_empirical"]})
        .groupby("time", as_index=True)["v"]
        .mean()
        .to_frame()
        .sort_index()
    )

    df_ch_area = (
        pd.DataFrame(
            {
                "time": launch_time,
                "ch_relative_area": prepared_input["ch_relative_area"],
            }
        )
        .groupby("time", as_index=True)["ch_relative_area"]
        .mean()
        .to_frame()
        .sort_index()
    )

    if superresolution_enabled:
        sr_index = pd.date_range(
            df_v.index.min().floor(time_freq),
            df_v.index.max().ceil(time_freq),
            freq=time_freq,
        )
        df_v = df_v.reindex(sr_index).interpolate(method="time").ffill().bfill()
        df_ch_area = df_ch_area.reindex(sr_index).ffill().bfill()
        df_v.index.name = "time"
        df_ch_area.index.name = "time"

    sim_start = df_v.index.min().floor(time_freq)
    sim_end = (df_v.index.max() + pd.Timedelta(days=float(simulation_pad_days))).ceil(
        time_freq
    )

    return {
        "sdo_input_df": prepared_input,
        "df_v": df_v,
        "df_ch_area": df_ch_area,
        "sim_start": sim_start,
        "sim_end": sim_end,
    }


def load_ace_at_earth(ace_path=DEFAULT_ACE_PARQUET_PATH):
    df_ace_earth = load_ace_earth_frame(ace_path)
    return df_ace_earth[["v"]].rename(columns={"v": "v_ace"})


def build_ace_earth_swx_frame(sdo_input_df):
    time_column = "forecast_dt" if "forecast_dt" in sdo_input_df.columns else "dt"
    df_swx = pd.DataFrame(index=pd.to_datetime(sdo_input_df[time_column]))
    if "forecast_sw_speed" in sdo_input_df.columns:
        df_swx["v_swx"] = pd.to_numeric(
            sdo_input_df["forecast_sw_speed"], errors="coerce"
        ).to_numpy()
    df_swx.attrs["sat"] = DEFAULT_ACE_EARTH_SAT
    df_swx.attrs["label"] = DEFAULT_ACE_EARTH_LABEL
    return df_swx.sort_index()
