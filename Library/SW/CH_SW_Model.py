from dataclasses import dataclass, replace
from functools import cached_property
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pandas as pd

MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parent.parent
DEFAULT_SHUGAY_PATH = PROJECT_ROOT / "Models" / "CH_SW_Correspondence" / "Shugay.py"


@dataclass(frozen=True)
class CHSWModel:
    source_path: Path

    @classmethod
    def load(cls, path=None):
        source_path = Path(path) if path is not None else DEFAULT_SHUGAY_PATH
        spec = spec_from_file_location("helio_n_ch_sw_model", source_path)
        module = module_from_spec(spec)
        assert (
            spec.loader is not None
        ), f"Unable to load CH-SW model module: {source_path}"
        spec.loader.exec_module(module)
        loaded = module.load()
        assert isinstance(
            loaded, cls
        ), f"{source_path} load() must return {cls.__name__}"
        return loaded

    def v_from_area(self, area, t=None):
        raise NotImplementedError

    def slow_sw_speed(self, t=None):
        raise NotImplementedError


@dataclass(frozen=True)
class EmpiricalCHSWModel(CHSWModel):
    # v_min can be func or float
    v_min: object
    a: float
    alpha: float

    @classmethod
    def from_fields(cls, source_path, v_min, a, alpha):
        return cls(
            source_path=Path(source_path),
            v_min=v_min,
            a=float(a),
            alpha=float(alpha),
        )

    def with_overrides(self, **overrides):
        if "source_path" in overrides:
            overrides["source_path"] = Path(overrides["source_path"])
        if "a" in overrides:
            overrides["a"] = float(overrides["a"])
        if "alpha" in overrides:
            overrides["alpha"] = float(overrides["alpha"])
        return replace(self, **overrides)

    @cached_property
    def constant_v_min(self):
        if callable(self.v_min):
            return None
        return float(self.v_min)

    def v_min_value(self, t=None):
        if self.constant_v_min is not None:
            return self.constant_v_min
        return self.v_min(t)

    def slow_sw_speed(self, t=None):
        if self.constant_v_min is not None:
            if t is None or np.isscalar(t) or isinstance(t, pd.Timestamp):
                return self.constant_v_min
            return np.full(len(pd.to_datetime(t)), self.constant_v_min, dtype=float)
        return self.v_min_value(t=t)

    def v_from_area(self, area, t=None):
        values = np.asarray(area, dtype=float)
        baseline = self.slow_sw_speed(t=t)
        speed = baseline + self.a * (values**self.alpha)
        if isinstance(area, pd.Series):
            return pd.Series(speed, index=area.index, name="v_empirical")
        return speed
