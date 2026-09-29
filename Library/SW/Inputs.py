from pathlib import Path

from astropy import units as u
from astropy.coordinates import SkyCoord
import numpy as np
import pandas as pd
import psycopg
from sunpy.coordinates.frames import (
    GeocentricSolarEcliptic,
    HeliocentricEarthEcliptic,
    HeliographicStonyhurst,
)

import userpwd
from Library.Paths import data_path, resolve_repo_path
from Library.SW.Constants import SW_MODEL_V2_HANDOFF
from Library.SW.Constants import SOLAR_RADIUS_KM
from Library.SW.Config import get_satellite_config

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
WHERE f.forecast_dt >= %(start_dt)s
  AND f.forecast_dt < %(end_dt)s
ORDER BY f.forecast_dt, f.dt;
"""

DEFAULT_SQL_CONNECTION = {
    "host": "213.131.1.41",
    "user": "selector",
    "dbname": "smdc",
}

DEFAULT_INPUT_PARQUET_PATH = data_path("CH Area.parquet")
DEFAULT_ACE_PARQUET_PATH = data_path("ACE 1h.parquet")
DEFAULT_ACE_AT_EARTH_PARQUET_PATH = data_path("ACE At Earth 1h.parquet")
# Daily GSE positions published by the ACE Science Center:
# https://izw1.caltech.edu/ACE/ASC/DATA/pos_att/ACE_GSE_position.txt
DEFAULT_ACE_EPHEMERIS_PATH = data_path("ACE GSE position.txt")
DEFAULT_STEREO_A_PARQUET_PATH = data_path("STEREO-A PLASTIC.parquet")
# 5,087 native forecast_dt/forecast_sw_speed points from
# sdo.sdo_sw_forecast_0193p over [2018-01-01, 2019-01-01), without interpolation.
DEFAULT_SWX_PARQUET_PATH = data_path("SWX Forecast 2018.parquet")
DEFAULT_ENLIL_PARQUET_PATH = data_path("ENLIL 2018-02-01 2018-07-01.parquet")
DEFAULT_ACE_SAT = "ace"
DEFAULT_ACE_LABEL = "ACE"
DEFAULT_ACE_EARTH_SAT = "ace_earth"
DEFAULT_ACE_EARTH_LABEL = "ACE @ Earth"
DEFAULT_STEREO_A_SAT = "stereo_a"
DEFAULT_STEREO_A_LABEL = "STEREO-A"
EARTH_RADII_PER_SOLAR_RADIUS = 109.0763707060096
SATELLITE_MAX_SOURCE_GAP = pd.Timedelta(hours=6)
SATELLITE_FRAME_COLUMNS = [
    "x_hee_au",
    "y_hee_au",
    "z_hee_au",
    "v",
    "v_x",
    "v_y",
    "v_z",
    "N",
    "t",
    "b",
    "b_x_gse",
    "b_y_gse",
    "b_z_gse",
    "b_r_rtn",
    "b_t_rtn",
    "b_n_rtn",
    "v_swx",
]
ACE_GSE_COLUMNS = ("x_gse_km", "y_gse_km", "z_gse_km")
AU_KM = 149597870.7


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
                    "sdo"
                    if chunk_start < pd.Timestamp(SW_MODEL_V2_HANDOFF)
                    else "sdo_v2"
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


def _hee_cartesian_from_hgs(index, longitude_deg, latitude_deg, radius_au):
    """Convert Heliographic Stonyhurst positions to Heliocentric Earth Ecliptic."""
    observation_times = pd.DatetimeIndex(index)
    source = SkyCoord(
        lon=np.asarray(longitude_deg, dtype=float) * u.deg,
        lat=np.asarray(latitude_deg, dtype=float) * u.deg,
        radius=np.asarray(radius_au, dtype=float) * u.AU,
        frame=HeliographicStonyhurst(obstime=observation_times),
    )
    target = source.transform_to(
        HeliocentricEarthEcliptic(obstime=observation_times)
    )
    xyz_au = target.cartesian.xyz.to_value(u.AU).T
    return pd.DataFrame(
        xyz_au,
        index=observation_times,
        columns=["x_hee_au", "y_hee_au", "z_hee_au"],
    )


def _hee_cartesian_from_gse(index, x_km, y_km, z_km):
    """Convert geocentric solar-ecliptic ACE positions to HEE."""
    observation_times = pd.DatetimeIndex(index)
    source = SkyCoord(
        np.asarray(x_km, dtype=float) * u.km,
        np.asarray(y_km, dtype=float) * u.km,
        np.asarray(z_km, dtype=float) * u.km,
        frame=GeocentricSolarEcliptic(obstime=observation_times),
        representation_type="cartesian",
    )
    target = source.transform_to(
        HeliocentricEarthEcliptic(obstime=observation_times)
    )
    xyz_au = target.cartesian.xyz.to_value(u.AU).T
    return pd.DataFrame(
        xyz_au,
        index=observation_times,
        columns=["x_hee_au", "y_hee_au", "z_hee_au"],
    )


def _load_ace_hee_ephemeris(path):
    ephemeris_path = resolve_repo_path(path)
    assert ephemeris_path.exists(), f"Missing ACE ephemeris: {ephemeris_path}"
    raw = pd.read_csv(ephemeris_path, sep=r"\s+", engine="python")
    required = {"Year", "DOY", "Secofday", "GSE_X(km)", "GSE_y(km)", "GSE_z(km)"}
    assert required.issubset(raw.columns), (
        f"ACE ephemeris must contain {sorted(required)}; got {list(raw.columns)}"
    )
    dates = pd.to_datetime(
        raw["Year"].astype(str) + "-" + raw["DOY"].astype(str),
        format="%Y-%j",
        utc=True,
    ).dt.tz_convert(None) + pd.to_timedelta(raw["Secofday"], unit="s")
    source = pd.DataFrame(
        {
            "x_gse_km": pd.to_numeric(raw["GSE_X(km)"], errors="coerce").to_numpy(),
            "y_gse_km": pd.to_numeric(raw["GSE_y(km)"], errors="coerce").to_numpy(),
            "z_gse_km": pd.to_numeric(raw["GSE_z(km)"], errors="coerce").to_numpy(),
        },
        index=pd.DatetimeIndex(dates),
    )
    source = source.dropna().sort_index()
    source = source[~source.index.duplicated(keep="last")]
    return source


def _convert_ace_gse_columns_to_hee(frame):
    gse = frame[list(ACE_GSE_COLUMNS)].dropna()
    if gse.empty:
        return pd.DataFrame(
            index=frame.index,
            columns=["x_hee_au", "y_hee_au", "z_hee_au"],
            dtype=float,
        )
    converted = _hee_cartesian_from_gse(
        gse.index,
        gse["x_gse_km"],
        gse["y_gse_km"],
        gse["z_gse_km"],
    )
    return converted.reindex(frame.index).interpolate(method="time")


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
        for candidate in ("speed", "V", "v_ace", "v_real"):
            if candidate in df_sat.columns:
                rename_map[candidate] = "v"
                break
    for canonical, candidates in {
        "N": ("density", "n"),
        "t": ("temperature", "temp"),
        "b": ("B", "b_total", "magnetic_field_magnitude"),
        "b_x_gse": ("B_X_GSE",),
        "b_y_gse": ("B_Y_GSE",),
        "b_z_gse": ("B_Z_GSE",),
        "b_r_rtn": ("b_r", "Br"),
        "b_t_rtn": ("b_t", "Bt"),
        "b_n_rtn": ("b_n", "Bn"),
    }.items():
        if canonical not in df_sat.columns:
            for candidate in candidates:
                if candidate in df_sat.columns:
                    rename_map[candidate] = canonical
                    break
    if "v_swx" not in df_sat.columns and "forecast_sw_speed" in df_sat.columns:
        rename_map["forecast_sw_speed"] = "v_swx"
    for canonical, candidates in {
        "x_gse_km": ("pos_gse_x", "gse_x_km"),
        "y_gse_km": ("pos_gse_y", "gse_y_km"),
        "z_gse_km": ("pos_gse_z", "gse_z_km"),
    }.items():
        if canonical not in df_sat.columns:
            for candidate in candidates:
                if candidate in df_sat.columns:
                    rename_map[candidate] = canonical
                    break
    df_sat = df_sat.rename(columns=rename_map)
    ambiguous_components = {
        "b_x", "b_y", "b_z", "Bx", "By", "Bz",
        "magnetic_field_x", "magnetic_field_y", "magnetic_field_z",
    }.intersection(df_sat.columns)
    assert not ambiguous_components, (
        "Magnetic components require an explicit frame in their column names; "
        f"resolve {sorted(ambiguous_components)} from the source metadata"
    )
    for components in (
        ("b_x_gse", "b_y_gse", "b_z_gse"),
        ("b_r_rtn", "b_t_rtn", "b_n_rtn"),
    ):
        if "b" not in df_sat.columns and set(components).issubset(df_sat.columns):
            vector = df_sat[list(components)].apply(pd.to_numeric, errors="coerce")
            df_sat["b"] = np.sqrt((vector ** 2).sum(axis=1, min_count=3))

    keep_columns = [
        column for column in SATELLITE_FRAME_COLUMNS if column in df_sat.columns
    ] + [column for column in ACE_GSE_COLUMNS if column in df_sat.columns]
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


def load_ace_frame(
    ace_path=DEFAULT_ACE_PARQUET_PATH,
    ephemeris_path=DEFAULT_ACE_EPHEMERIS_PATH,
):
    frame = load_cached_satellite_frame(
        ace_path,
        sat=DEFAULT_ACE_SAT,
        label=DEFAULT_ACE_LABEL,
    )
    if set(("x_hee_au", "y_hee_au", "z_hee_au")).issubset(frame.columns):
        positions = frame[["x_hee_au", "y_hee_au", "z_hee_au"]]
        position_source = "native ACE HEE columns"
    elif set(ACE_GSE_COLUMNS).issubset(frame.columns):
        positions = _convert_ace_gse_columns_to_hee(frame)
        position_source = "native ACE GSE columns converted to HEE"
    else:
        ephemeris = _load_ace_hee_ephemeris(ephemeris_path)
        positions = _hee_cartesian_from_gse(
            ephemeris.index,
            ephemeris["x_gse_km"],
            ephemeris["y_gse_km"],
            ephemeris["z_gse_km"],
        ).reindex(frame.index).interpolate(method="time")
        position_source = str(ephemeris_path)
    missing_positions = positions.isna().any(axis=1)
    assert not missing_positions.any(), (
        "Native ACE observations need an HEE position for every sample; "
        f"missing {int(missing_positions.sum())} rows"
    )
    frame[["x_hee_au", "y_hee_au", "z_hee_au"]] = positions
    frame.attrs["coord_frame"] = "HEE"
    frame.attrs["position_static"] = False
    frame.attrs["source"] = "ACE"
    frame.attrs["position_source"] = position_source
    return frame


def load_ace_earth_frame(ace_path=DEFAULT_ACE_AT_EARTH_PARQUET_PATH):
    """Load the propagated ACE-at-Earth reference at the fixed HEE Earth point."""
    frame = load_cached_satellite_frame(
        ace_path,
        sat=DEFAULT_ACE_EARTH_SAT,
        label=DEFAULT_ACE_EARTH_LABEL,
    )
    frame[["x_hee_au", "y_hee_au", "z_hee_au"]] = (1.0, 0.0, 0.0)
    frame.attrs["coord_frame"] = "HEE"
    frame.attrs["position_static"] = True
    frame.attrs["source"] = "ACE at Earth"
    frame.attrs["position_source"] = "fixed Earth point"
    return frame


def load_ace_swx_frame(swx_path=DEFAULT_SWX_PARQUET_PATH):
    frame = load_cached_satellite_frame(
        swx_path,
        sat=DEFAULT_ACE_SAT,
        label=DEFAULT_ACE_LABEL,
    )
    assert "v_swx" in frame.columns, f"Missing v_swx column in {swx_path}"
    return frame


def interpolate_short_gaps(frame, target_index, max_source_gap=SATELLITE_MAX_SOURCE_GAP):
    """Interpolate each column only between observations at most max_source_gap apart."""
    max_source_gap = pd.Timedelta(max_source_gap)
    assert max_source_gap > pd.Timedelta(0)
    source = frame.sort_index()
    assert source.index.is_unique, "Satellite source timestamps must be unique"
    target_index = pd.DatetimeIndex(target_index)
    interpolation_index = source.index.union(target_index).sort_values()
    source = source.reindex(interpolation_index)
    interpolated = source.interpolate(method="time", limit_area="inside")
    for column in source.columns:
        observed_times = source.index[source[column].notna()]
        source_times = pd.Series(observed_times, index=observed_times)
        previous_time = source_times.reindex(interpolation_index).ffill()
        next_time = source_times.reindex(interpolation_index).bfill()
        bounded = (next_time - previous_time) <= max_source_gap
        interpolated[column] = interpolated[column].where(bounded)
    return interpolated.reindex(target_index)


def load_stereo_a_frame(
    time_axis,
    time_freq,
    stereo_path=DEFAULT_STEREO_A_PARQUET_PATH,
    max_source_gap=SATELLITE_MAX_SOURCE_GAP,
):
    stereo_path = resolve_repo_path(stereo_path)
    stereo_a_df = pd.read_parquet(stereo_path).copy()
    stereo_a_df.index = pd.to_datetime(stereo_a_df.index, utc=True).tz_convert(None)
    sampling_margin = max(pd.Timedelta(time_freq), pd.Timedelta(max_source_gap))
    stereo_a_df = stereo_a_df.loc[
        pd.Timestamp(time_axis.min())
        - sampling_margin : pd.Timestamp(time_axis.max())
        + sampling_margin
    ]
    stereo_a_df = normalize_satellite_frame(
        stereo_a_df,
        sat=DEFAULT_STEREO_A_SAT,
        label=DEFAULT_STEREO_A_LABEL,
    )
    stereo_a_df = stereo_a_df.resample(time_freq).mean()
    stereo_a_df = interpolate_short_gaps(stereo_a_df, time_axis, max_source_gap)
    required_coordinates = {"heliographicLatitude", "heliographicLongitude", "radialDistance"}
    raw_coordinates = pd.read_parquet(stereo_path, columns=list(required_coordinates)).copy()
    raw_coordinates.index = pd.to_datetime(raw_coordinates.index, utc=True).tz_convert(None)
    raw_coordinates = raw_coordinates.loc[
        pd.Timestamp(time_axis.min()) - sampling_margin : pd.Timestamp(time_axis.max()) + sampling_margin
    ]
    raw_coordinates = raw_coordinates.apply(pd.to_numeric, errors="coerce")
    raw_coordinates = raw_coordinates.resample(time_freq).mean()
    raw_coordinates = interpolate_short_gaps(raw_coordinates, time_axis, max_source_gap)
    radial_distance_solar = (
        raw_coordinates["radialDistance"] / EARTH_RADII_PER_SOLAR_RADIUS
    )
    radius_au = radial_distance_solar * SOLAR_RADIUS_KM / AU_KM
    positions = _hee_cartesian_from_hgs(
        raw_coordinates.index,
        raw_coordinates["heliographicLongitude"],
        raw_coordinates["heliographicLatitude"],
        radius_au,
    )
    stereo_a_df = stereo_a_df.reindex(time_axis)
    stereo_a_df[["x_hee_au", "y_hee_au", "z_hee_au"]] = positions
    stereo_a_df["phi_target"] = np.mod(
        np.degrees(np.arctan2(positions["y_hee_au"], positions["x_hee_au"])),
        360.0,
    )
    stereo_a_df["lat_hee"] = np.degrees(
        np.arcsin(
            np.divide(
                positions["z_hee_au"],
                np.sqrt((positions ** 2).sum(axis=1)),
                out=np.full(len(positions), np.nan),
                where=(positions ** 2).sum(axis=1).to_numpy() > 0.0,
            )
        )
    )
    stereo_a_df["r_target"] = (
        radial_distance_solar
    )
    stereo_a_df["lat_hgs"] = raw_coordinates["heliographicLatitude"]
    stereo_a_df.attrs["sat"] = DEFAULT_STEREO_A_SAT
    stereo_a_df.attrs["label"] = DEFAULT_STEREO_A_LABEL
    stereo_a_df.attrs["coord_frame"] = "HEE"
    stereo_a_df.attrs["source_coord_frame"] = "HGS"
    stereo_a_df.attrs["source"] = "STEREO-A PLASTIC"
    return stereo_a_df


def load_satellite_frame(
    sat_id,
    time_axis=None,
    time_freq=None,
    ace_path=DEFAULT_ACE_PARQUET_PATH,
    ace_at_earth_path=DEFAULT_ACE_AT_EARTH_PARQUET_PATH,
    stereo_a_path=DEFAULT_STEREO_A_PARQUET_PATH,
):
    """Load one configured source in the common satellite-frame schema."""
    config = get_satellite_config(sat_id)
    if config.loader == "ace":
        return load_ace_frame(ace_path=ace_path)
    if config.loader == "ace_earth":
        return load_ace_earth_frame(ace_path=ace_at_earth_path)
    if config.loader == "stereo_a":
        assert time_axis is not None and time_freq is not None, (
            "STEREO-A loading requires time_axis and time_freq"
        )
        return load_stereo_a_frame(
            time_axis=time_axis,
            time_freq=time_freq,
            stereo_path=stereo_a_path,
        )
    raise ValueError(f"No data loader configured for satellite: {sat_id}")


def load_satellite_frames(satellite_ids, time_axis, time_freq):
    return {
        sat_id: load_satellite_frame(
            sat_id=sat_id,
            time_axis=time_axis,
            time_freq=time_freq,
        )
        for sat_id in satellite_ids
    }


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

    ace_enlil = build_enlil_frame("Earth_V1")
    return {
        DEFAULT_ACE_SAT: ace_enlil,
        DEFAULT_ACE_EARTH_SAT: ace_enlil.copy(),
        DEFAULT_STEREO_A_SAT: build_enlil_frame("STEREO_A_V1"),
    }


def build_model_input_series(
    sdo_input_df,
    empirical,
    output_step_minutes,
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
    if "forecast_dt" in prepared_input.columns:
        parameter_time = pd.to_datetime(prepared_input["forecast_dt"])
        assert parameter_time.notna().all(), (
            "Expected forecast_dt to be populated when selecting time-dependent "
            "CH-SW parameters"
        )
    else:
        parameter_time = launch_time
    prepared_input["v_empirical"] = empirical.v_from_area(
        prepared_input["ch_relative_area"].to_numpy(dtype=float),
        t=launch_time,
        parameter_time=parameter_time,
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

    df_v.index.name = "time"
    df_ch_area.index.name = "time"

    output_frequency = f"{int(output_step_minutes)}min"
    sim_start = df_v.index.min().floor(output_frequency)
    sim_end = (df_v.index.max() + pd.Timedelta(days=float(simulation_pad_days))).ceil(
        output_frequency
    )

    return {
        "sdo_input_df": prepared_input,
        "df_v": df_v,
        "df_ch_area": df_ch_area,
        "sim_start": sim_start,
        "sim_end": sim_end,
    }


def load_ace_at_earth(ace_path=DEFAULT_ACE_AT_EARTH_PARQUET_PATH):
    df_ace_earth = load_ace_earth_frame(ace_path)
    return df_ace_earth[["v"]].rename(columns={"v": "v_ace"})


def build_ace_earth_swx_frame(sdo_input_df):
    time_column = "forecast_dt" if "forecast_dt" in sdo_input_df.columns else "dt"
    df_swx = pd.DataFrame(index=pd.to_datetime(sdo_input_df[time_column]))
    if "forecast_sw_speed" in sdo_input_df.columns:
        df_swx["v_swx"] = pd.to_numeric(
            sdo_input_df["forecast_sw_speed"], errors="coerce"
        ).to_numpy()
    df_swx.attrs["sat"] = DEFAULT_ACE_SAT
    df_swx.attrs["label"] = DEFAULT_ACE_LABEL
    return df_swx.sort_index()
