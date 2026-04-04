from dataclasses import dataclass, replace
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

    def v_min_value(self, t=None):
        if callable(self.v_min):
            return self.v_min(t)
        return float(self.v_min)

    def v_from_area(self, area, t=None):
        values = np.asarray(area, dtype=float)
        baseline = self.v_min_value(t=t)
        speed = baseline + self.a * (values**self.alpha)
        if isinstance(area, pd.Series):
            return pd.Series(speed, index=area.index, name="v_empirical")
        return speed
