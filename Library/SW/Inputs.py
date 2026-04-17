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
DEFAULT_ACE_EARTH_SAT = "ace_earth"
DEFAULT_ACE_EARTH_LABEL = "ACE @ Earth"
SATELLITE_FRAME_COLUMNS = [
    "v",
    "v_gse_x",
    "v_gse_y",
    "v_gse_z",
    "x_gse",
    "y_gse",
    "z_gse",
    "phi_target",
    "r_target",
    "lat_hgs",
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
                chunk_query = query
                if chunk_query is None:
                    chunk_query = DEFAULT_SQL_QUERY.replace(
                        "SDO_PREFIX",
                        (
                            "sdo"
                            if chunk_start < pd.Timestamp(SDO_V2_HANDOFF)
                            else "sdo_v2"
                        ),
                    )
                cur.execute(
                    chunk_query,
                    {"start_dt": chunk_start, "end_dt": chunk_end},
                )
                rows = cur.fetchall()
                if columns is None:
                    columns = [desc.name for desc in cur.description]
                frames.append(pd.DataFrame(rows, columns=columns))
    conn.close()

    if columns is None:
        return pd.DataFrame()
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

    df_sdo_sw = df_sdo_sw[
        (df_sdo_sw["dt"] >= start_dt) & (df_sdo_sw["dt"] < end_dt)
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

    numeric_columns = [
        column for column in SATELLITE_FRAME_COLUMNS if column in df_sat.columns
    ]
    for column in numeric_columns:
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


def v_from_area(area, empirical, t=None):
    return empirical.v_from_area(area, t=t)


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
    prepared_input["v_empirical"] = v_from_area(
        prepared_input["ch_relative_area"].to_numpy(dtype=float),
        empirical=empirical,
        t=launch_time,
    )
    df_v = (
        pd.DataFrame({"time": launch_time, "v": prepared_input["v_empirical"]})
        .groupby("time", as_index=True)["v"]
        .mean()
        .to_frame()
        .sort_index()
    )

    df_ch_area_hourly = (
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
        df_ch_area = df_ch_area_hourly.reindex(sr_index).ffill().bfill()
        df_v.index.name = "time"
        df_ch_area_hourly.index.name = "time"
        df_ch_area.index.name = "time"
    else:
        df_ch_area = df_ch_area_hourly.copy()

    sim_start = df_v.index.min().floor(time_freq)
    sim_end = (df_v.index.max() + pd.Timedelta(days=float(simulation_pad_days))).ceil(
        time_freq
    )

    return {
        "sdo_input_df": prepared_input,
        "df_v": df_v,
        "df_ch_area_hourly": df_ch_area_hourly,
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
    elif "sw_speed_1" in sdo_input_df.columns:
        df_swx["v_swx"] = pd.to_numeric(
            sdo_input_df["sw_speed_1"], errors="coerce"
        ).to_numpy()
    df_swx.attrs["sat"] = DEFAULT_ACE_EARTH_SAT
    df_swx.attrs["label"] = DEFAULT_ACE_EARTH_LABEL
    return df_swx.sort_index()


def build_forecast_earth_frame(sdo_input_df):
    return build_ace_earth_swx_frame(sdo_input_df).rename(
        columns={"v_swx": "forecast_sw_speed"}
    )
