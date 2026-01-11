from __future__ import annotations

import numpy as np
import pandas as pd

from Models._base import DateRangeSpec, ModelSpec

YEARS = (2011, 2015, 2018, 2020)
VAL_FRACTION = 0.1
RANDOM_SEED = 0


def _to_dt_index(idx) -> pd.DatetimeIndex:
    raw = pd.Series(idx.astype(str))
    dt_direct = pd.to_datetime(raw, format="%Y%m%d_%H%M", errors="coerce")
    dt_direct_s = pd.to_datetime(raw, format="%Y%m%d_%H%M%S", errors="coerce")
    cleaned = raw.str.replace(r"\D", "", regex=True)
    with_seconds = cleaned.where(cleaned.str.len() == 14)
    with_minutes = cleaned.where(cleaned.str.len() == 12)
    dt = pd.to_datetime(with_seconds, format="%Y%m%d%H%M%S", errors="coerce")
    dt_minutes = pd.to_datetime(with_minutes, format="%Y%m%d%H%M", errors="coerce")
    return dt_direct.fillna(dt_direct_s).fillna(dt).fillna(dt_minutes)


def _with_dt(df: pd.DataFrame) -> pd.DataFrame:
    dt = _to_dt_index(df.index)
    df = df.copy()
    df["_dt"] = pd.to_datetime(dt).to_numpy()
    return df.dropna(subset=["_dt"])


def _year_slice(df: pd.DataFrame, year: int) -> pd.DataFrame:
    if "_dt" not in df.columns:
        df = _with_dt(df)
    return df[df["_dt"].dt.year == year]


def _shuffle_split_year(
    df: pd.DataFrame, year: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    year_df = _year_slice(df, year)
    year_df = _one_per_day(year_df, RANDOM_SEED + year)
    if len(year_df) == 0:
        empty = year_df.drop(columns=["_dt"], errors="ignore")
        return empty, empty

    shuffled = year_df.sample(frac=1.0, random_state=RANDOM_SEED + year)
    if len(shuffled) == 1:
        return (
            shuffled.drop(columns=["_dt"], errors="ignore"),
            shuffled.iloc[0:0].drop(columns=["_dt"], errors="ignore"),
        )

    n_val = max(1, int(len(shuffled) * VAL_FRACTION))
    n_val = min(n_val, len(shuffled) - 1)
    return (
        shuffled.iloc[:-n_val].drop(columns=["_dt"], errors="ignore"),
        shuffled.iloc[-n_val:].drop(columns=["_dt"], errors="ignore"),
    )


def _one_per_day(df: pd.DataFrame, seed: int) -> pd.DataFrame:
    if len(df) == 0:
        return df
    if "_dt" not in df.columns:
        df = _with_dt(df)
    if len(df) == 0:
        return df
    tmp = df.copy()
    tmp["_day"] = tmp["_dt"].dt.normalize()
    rng = np.random.RandomState(seed)
    tmp["_rand"] = rng.rand(len(tmp))
    tmp = tmp.sort_values(["_day", "_rand"])
    tmp = tmp.groupby("_day", sort=False).head(1)
    return tmp.drop(columns=["_day", "_rand", "_dt"])


def _a2_selector(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = _with_dt(df)
    if len(df) == 0:
        return df, df
    df = df.sort_values("_dt")

    train_parts = []
    val_parts = []
    for year in YEARS:
        train_year, val_year = _shuffle_split_year(df, year)
        if len(train_year):
            train_parts.append(train_year)
        if len(val_year):
            val_parts.append(val_year)

    train_df = pd.concat(train_parts) if train_parts else df.iloc[0:0]
    val_df = pd.concat(val_parts) if val_parts else df.iloc[0:0]
    return (
        train_df.drop(columns=["_dt"], errors="ignore"),
        val_df.drop(columns=["_dt"], errors="ignore"),
    )


MODEL = ModelSpec(
    model_id="A2",
    params={
        "base_filters": 80,
        "img_size": 256,
        "num_epochs": 30,
        "learning_rate": 0.0001,
        "correct_steps_by_n": True,
        "ceil_based_steps_per_epoch": False,
        "shuffle_df": False,
        "avoid_requantization": False,
    },
)

D1 = DateRangeSpec(
    range_id="D1",
    start=None,
    end=None,
    keep_every=1,
    selector=_a2_selector,
)

DATE_RANGES = {
    "D1": D1,
}


def get_architecture() -> dict:
    return dict(MODEL.params)


def get_date_range(date_range_id: str) -> DateRangeSpec:
    return DATE_RANGES[date_range_id]
