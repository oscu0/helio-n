from pathlib import Path

import numpy as np
import pandas as pd
import psycopg
from tqdm.auto import tqdm

import userpwd

DEFAULT_SQL_QUERY = """
SELECT
    dt,
    ch_relative_area,
    forecast_sw_speed,
    forecast_dt
FROM sdo.sdo_sw_forecast_0193
WHERE dt >= %(start_dt)s
  AND dt < %(end_dt)s
ORDER BY dt
"""

DEFAULT_SQL_CONNECTION = {
    "host": "213.131.1.41",
    "user": "selector",
    "dbname": "smdc",
}

DEFAULT_INPUT_PARQUET_PATH = Path("Data/CH Area.parquet")
DEFAULT_ACE_PARQUET_PATH = Path("Data/ACE At Earth 1h.parquet")


def load_sw_input_from_parquet(input_parquet_path=DEFAULT_INPUT_PARQUET_PATH):
    parquet_path = Path(input_parquet_path)
    assert parquet_path.exists(), f"Missing input parquet: {parquet_path}"
    return pd.read_parquet(parquet_path).copy()


def load_sw_input_from_sql(
    start_dt,
    end_dt,
    query=DEFAULT_SQL_QUERY,
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
    with conn.cursor() as cur:
        cur.execute(query, {"start_dt": start_dt, "end_dt": end_dt})
        rows = cur.fetchall()
        columns = [desc.name for desc in cur.description]
    conn.close()

    return pd.DataFrame(rows, columns=columns)


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

    if "forecast_sw_speed" not in df_sdo_sw.columns:
        if {"sw_speed_1", "sw_speed_2"}.issubset(df_sdo_sw.columns):
            sw_speed_1 = pd.to_numeric(df_sdo_sw["sw_speed_1"], errors="coerce")
            sw_speed_2 = pd.to_numeric(df_sdo_sw["sw_speed_2"], errors="coerce")
            df_sdo_sw["forecast_sw_speed"] = sw_speed_1.fillna(sw_speed_2)
        elif "sw_speed_1" in df_sdo_sw.columns:
            df_sdo_sw["forecast_sw_speed"] = pd.to_numeric(
                df_sdo_sw["sw_speed_1"], errors="coerce"
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
    query=DEFAULT_SQL_QUERY,
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


def v_from_area(area, empirical):
    values = np.asarray(area, dtype=float)
    speed = empirical.v_min + empirical.a * (values**empirical.alpha)
    if isinstance(area, pd.Series):
        return pd.Series(speed, index=area.index, name="v_empirical")
    return speed


def build_model_input_series(
    sdo_input_df,
    empirical,
    time_controls,
    simulation_pad_days,
    input_chunk_rows,
    show_progress=True,
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

    area_values = prepared_input["ch_relative_area"].to_numpy(dtype=float)
    v_empirical = np.empty_like(area_values, dtype=float)
    chunk_rows = int(input_chunk_rows)
    iterator = range(0, len(area_values), chunk_rows)
    if show_progress:
        iterator = tqdm(iterator, desc="empirical v(area)", unit="rows")
    for start in iterator:
        stop = min(start + chunk_rows, len(area_values))
        v_empirical[start:stop] = v_from_area(
            area_values[start:stop], empirical=empirical
        )

    prepared_input["v_empirical"] = v_empirical
    launch_time = (prepared_input["dt"] + pd.Timedelta(minutes=30)).dt.floor("1h")
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

    if time_controls.superresolution_enabled:
        sr_index = pd.date_range(
            df_v.index.min().floor(time_controls.time_freq),
            df_v.index.max().ceil(time_controls.time_freq),
            freq=time_controls.time_freq,
        )
        df_v = df_v.reindex(sr_index).interpolate(method="time").ffill().bfill()
        df_ch_area = df_ch_area_hourly.reindex(sr_index).ffill().bfill()
        df_v.index.name = "time"
        df_ch_area_hourly.index.name = "time"
        df_ch_area.index.name = "time"
    else:
        df_ch_area = df_ch_area_hourly.copy()

    sim_start = df_v.index.min().floor(time_controls.time_freq)
    sim_end = (df_v.index.max() + pd.Timedelta(days=float(simulation_pad_days))).ceil(
        time_controls.time_freq
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
    df_ace_earth = pd.read_parquet(ace_path).copy()
    if not isinstance(df_ace_earth.index, pd.DatetimeIndex):
        df_ace_earth.index = pd.to_datetime(df_ace_earth.index)
    if "speed" in df_ace_earth.columns:
        df_ace_earth = df_ace_earth.rename(columns={"speed": "v_ace"})
    return df_ace_earth[["v_ace"]].sort_index()


def build_forecast_earth_frame(sdo_input_df):
    df_forecast_earth = pd.DataFrame(index=pd.to_datetime(sdo_input_df["dt"]))
    if "forecast_sw_speed" in sdo_input_df.columns:
        df_forecast_earth["forecast_sw_speed"] = pd.to_numeric(
            sdo_input_df["forecast_sw_speed"], errors="coerce"
        ).to_numpy()
    elif "sw_speed_1" in sdo_input_df.columns:
        df_forecast_earth["forecast_sw_speed"] = pd.to_numeric(
            sdo_input_df["sw_speed_1"], errors="coerce"
        ).to_numpy()
    return df_forecast_earth.sort_index()
