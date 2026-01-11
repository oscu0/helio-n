from __future__ import annotations

import pandas as pd

from Models._base import DateRangeSpec, ModelSpec


def _split_selector(start: str, end: str):
    def _select(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        subset = df[start:end]
        if len(subset) == 0:
            return subset, subset
        n_val = max(1, int(0.1 * len(subset)))
        n_train = len(subset) - n_val
        return subset.iloc[:n_train], subset.iloc[n_train:]

    return _select

MODEL = ModelSpec(
    model_id="A0",
    params={
        "base_filters": 32,
        "img_size": 64,
        "num_epochs": 20,
        "learning_rate": 0.0001,
        "avoid_requantization": False,
        "correct_steps_by_n": True,
        "batch_size": 4,
    },
)

D1 = DateRangeSpec(
    range_id="D1",
    start="20170101",
    end="20180101",
    keep_every=1,
    selector=_split_selector("20170101", "20180101"),
)

DATE_RANGES = {
    "D1": D1,
}


def get_architecture() -> dict:
    return dict(MODEL.params)


def get_date_range(date_range_id: str) -> DateRangeSpec:
    return DATE_RANGES[date_range_id]
