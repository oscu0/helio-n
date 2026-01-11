from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import pandas as pd


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    params: dict


@dataclass(frozen=True)
class DateRangeSpec:
    range_id: str
    start: Optional[str] = None
    end: Optional[str] = None
    keep_every: int = 1
    selector: Optional[Callable[[pd.DataFrame], tuple[pd.DataFrame, pd.DataFrame]]] = None

    def select_pairs(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        if self.selector is not None:
            return self.selector(df)

        if self.start is None or self.end is None:
            raise ValueError(
                f"DateRangeSpec {self.range_id} requires start/end or a selector."
            )

        return df[self.start : self.end], None
