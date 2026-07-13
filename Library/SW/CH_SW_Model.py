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
    a_before_handoff: float | None = None
    alpha_before_handoff: float | None = None
    parameter_handoff: object = None

    def slow_sw_speed(self, t=None):
        if not callable(self.v_min):
            v_min_const = float(self.v_min)
            if t is None or np.isscalar(t) or isinstance(t, pd.Timestamp):
                return v_min_const
            return np.full(len(pd.to_datetime(t)), v_min_const, dtype=float)
        return self.v_min(t)

    def v_from_area(self, area, t=None, parameter_time=None):
        values = np.asarray(area, dtype=float)
        baseline = self.slow_sw_speed(t=t)
        if parameter_time is None:
            parameter_time = t

        a = self.a
        alpha = self.alpha
        if self.parameter_handoff is not None:
            assert parameter_time is not None, (
                "Time-dependent CH-SW parameters require timestamps"
            )
            assert self.a_before_handoff is not None, (
                "a_before_handoff is required when parameter_handoff is set"
            )
            assert self.alpha_before_handoff is not None, (
                "alpha_before_handoff is required when parameter_handoff is set"
            )
            before_handoff = np.asarray(
                pd.to_datetime(parameter_time) < pd.Timestamp(self.parameter_handoff)
            )
            a = np.where(before_handoff, self.a_before_handoff, self.a)
            alpha = np.where(
                before_handoff,
                self.alpha_before_handoff,
                self.alpha,
            )

        speed = baseline + a * (values**alpha)
        if isinstance(area, pd.Series):
            return pd.Series(speed, index=area.index, name="v_empirical")
        return speed
