from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


def load_ch_sw_model():
    from Models.CH_SW_Correspondence.Shugay import MODEL
    return MODEL


@dataclass(frozen=True)
class EmpiricalCHSWModel:
    source_path: Path
    v_min: object  # float or callable
    a: float
    alpha: float

    def slow_sw_speed(self, t=None):
        if not callable(self.v_min):
            v_min_const = float(self.v_min)
            if t is None or np.isscalar(t) or isinstance(t, pd.Timestamp):
                return v_min_const
            return np.full(len(pd.to_datetime(t)), v_min_const, dtype=float)
        return self.v_min(t)

    def v_from_area(self, area, t=None):
        values = np.asarray(area, dtype=float)
        baseline = self.slow_sw_speed(t=t)
        speed = baseline + self.a * (values**self.alpha)
        if isinstance(area, pd.Series):
            return pd.Series(speed, index=area.index, name="v_empirical")
        return speed
