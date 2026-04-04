from pathlib import Path

import numpy as np
import pandas as pd

from Library.SW.CH_SW_Model import EmpiricalCHSWModel

ACE_PATH = Path("Data/ACE At Earth 1h.parquet")


def load_ace_speed_series(path=ACE_PATH):
    df = pd.read_parquet(path).copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    speed_column = "v_ace" if "v_ace" in df.columns else "speed"
    series = pd.to_numeric(df[speed_column], errors="coerce").sort_index()
    series = series[~series.index.duplicated(keep="last")]
    series = series.resample("1h").mean().interpolate("time").ffill().bfill()
    series.name = "v_ace"
    return series


def shugay_slow_sw_vmin_factory(v_ace):
    def shugay_slow_sw_vmin(t):
        assert t is not None, "Shugay_Slow_SW v_min requires timestamps"

        timestamps = pd.to_datetime(t)
        if isinstance(timestamps, pd.Timestamp):
            timestamps = pd.DatetimeIndex([timestamps])
            scalar = True
        else:
            timestamps = pd.DatetimeIndex(timestamps)
            scalar = False

        values = np.empty(len(timestamps), dtype=float)
        for i, ts in enumerate(timestamps):
            anchor_672 = ts - pd.Timedelta(hours=672)
            min_end = ts - pd.Timedelta(hours=1)
            min_start = ts - pd.Timedelta(hours=336)
            left = float(v_ace.asof(anchor_672))
            right = float(v_ace.loc[min_start:min_end].min())
            values[i] = 0.5 * (left + right)

        if scalar:
            return float(values[0])
        return values

    return shugay_slow_sw_vmin


MODEL = EmpiricalCHSWModel.from_fields(
    source_path=Path(__file__).resolve(),
    v_min=shugay_slow_sw_vmin_factory(load_ace_speed_series()),
    a=180.0,
    alpha=0.6,
)


def load():
    return MODEL
