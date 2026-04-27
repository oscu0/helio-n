from dataclasses import dataclass
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pandas as pd

MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parent.parent
DEFAULT_SHUGAY_PATH = PROJECT_ROOT / "Models" / "CH_SW_Correspondence" / "Shugay.py"


def load_ch_sw_model(path=None):
    source_path = Path(path) if path is not None else DEFAULT_SHUGAY_PATH
    spec = spec_from_file_location("helio_n_ch_sw_model", source_path)
    module = module_from_spec(spec)
    assert spec.loader is not None, f"Unable to load CH-SW model module: {source_path}"
    spec.loader.exec_module(module)
    loaded = module.load()
    if not hasattr(loaded, "v_from_area"):
        raise AssertionError(
            f"{source_path} load() must return an object with v_from_area()"
        )
    return loaded


@dataclass(frozen=True)
class EmpiricalCHSWModel:
    source_path: Path
    v_min: object  # float or callable
    a: float
    alpha: float

    @classmethod
    def from_fields(cls, source_path=None, v_min=None, a=180.0, alpha=0.6):
        resolved_source_path = (
            Path(source_path) if source_path is not None else DEFAULT_SHUGAY_PATH
        )
        return cls(
            source_path=resolved_source_path,
            v_min=v_min,
            a=float(a),
            alpha=float(alpha),
        )

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
