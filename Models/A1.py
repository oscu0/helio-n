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
    model_id="A1",
    params={
        "base_filters": 80,
        "img_size": 256,
        "num_epochs": 30,
        "learning_rate": 0.0001,
        "avoid_requantization": False,
        "correct_steps_by_n": False,
        "batch_size": 4,
    },
)

D1 = DateRangeSpec(
    range_id="D1",
    start="20170501",
    end="20170801",
    keep_every=3,
    selector=_split_selector("20170501", "20170801"),
)


D2 = DateRangeSpec(
    range_id="D1",
    start="20220101",
    end="20230101",
    keep_every=1,
    selector=_split_selector("20170501", "20170801"),
)


DATE_RANGES = {
    "D1": D1,
}


def get_architecture() -> dict:
    return dict(MODEL.params)


def get_date_range(date_range_id: str) -> DateRangeSpec:
    return DATE_RANGES[date_range_id]
