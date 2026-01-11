from __future__ import annotations

import pandas as pd

from Models._base import DateRangeSpec, ModelSpec

YEARS = (2011, 2015, 2018, 2020)
VAL_FRACTION = 0.1
RANDOM_SEED = 0


def _year_slice(df: pd.DataFrame, year: int) -> pd.DataFrame:
    start = f"{year}0101"
    end = f"{year + 1}0101"
    return df[start:end]


def _shuffle_split_year(
    df: pd.DataFrame, year: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    year_df = _year_slice(df, year)
    if len(year_df) == 0:
        return year_df, year_df

    shuffled = year_df.sample(frac=1.0, random_state=RANDOM_SEED + year)
    if len(shuffled) == 1:
        return shuffled, shuffled.iloc[0:0]

    n_val = max(1, int(len(shuffled) * VAL_FRACTION))
    n_val = min(n_val, len(shuffled) - 1)
    return shuffled.iloc[:-n_val], shuffled.iloc[-n_val:]


def _a2_selector(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not df.index.is_monotonic_increasing:
        df = df.sort_index()

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
    return train_df, val_df


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
    },
)

D1 = DateRangeSpec(
    range_id="D1",
    start=None,
    end=None,
    keep_every=4,
    selector=_a2_selector,
)

DATE_RANGES = {
    "D1": D1,
}


def get_architecture() -> dict:
    return dict(MODEL.params)


def get_date_range(date_range_id: str) -> DateRangeSpec:
    return DATE_RANGES[date_range_id]
