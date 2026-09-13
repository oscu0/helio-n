import unittest
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from Library.SW.Inputs import build_model_input_series, interpolate_short_gaps, load_stereo_a_frame


class LinearEmpiricalModel:
    def v_from_area(self, area, t, parameter_time):
        return 300.0 + np.asarray(area, dtype=float)


class InputContractTests(unittest.TestCase):
    def test_satellite_interpolation_includes_six_hours_without_extrapolation(self):
        start = pd.Timestamp("2018-01-01 00:00")
        source = pd.DataFrame(
            {"v": [400.0, 500.0, 600.0]},
            index=pd.DatetimeIndex(
                [start, start + pd.Timedelta(hours=6), start + pd.Timedelta(hours=13)]
            ),
        )
        target = pd.date_range(start - pd.Timedelta(hours=1), periods=16, freq="1h")
        result = interpolate_short_gaps(source, target)
        self.assertEqual(result.loc[start + pd.Timedelta(hours=3), "v"], 450.0)
        self.assertTrue(np.isnan(result.loc[start + pd.Timedelta(hours=7), "v"]))
        self.assertTrue(np.isnan(result.loc[start - pd.Timedelta(hours=1), "v"]))
        self.assertTrue(np.isnan(result.loc[start + pd.Timedelta(hours=14), "v"]))

    def test_production_input_keeps_native_hourly_knots(self):
        start = pd.Timestamp("2020-01-01 00:00:00")
        source = pd.DataFrame(
            {
                "dt": pd.date_range(start, periods=3, freq="1h"),
                "ch_relative_area": [0.0, 100.0, 200.0],
            }
        )
        prepared = build_model_input_series(
            sdo_input_df=source,
            empirical=LinearEmpiricalModel(),
            output_step_minutes=60,
            simulation_pad_days=0.0,
        )
        self.assertEqual(len(prepared["df_v"]), 3)
        self.assertTrue(
            (
                prepared["df_v"].index.to_series().diff().dropna()
                == pd.Timedelta(hours=1)
            ).all()
        )

    def test_stereo_interpolates_short_gaps_but_leaves_long_gaps_missing(self):
        start = pd.Timestamp("2018-01-01 00:00")
        source = pd.DataFrame(
            {
                "V": [400.0, 500.0, 600.0],
                "radialDistance": [23451.0] * 3,
                "heliographicLatitude": [0.0, 10.0, 20.0],
                "heliographicLongitude": [0.0, 20.0, 40.0],
            },
            index=pd.DatetimeIndex(
                [start, start + pd.Timedelta(hours=2), start + pd.Timedelta(hours=10)],
                tz="UTC",
            ),
        )
        target = pd.date_range(start, periods=11, freq="1h")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stereo.parquet"
            source.to_parquet(path)
            result = load_stereo_a_frame(target, "1h", stereo_path=path)
        self.assertEqual(result.loc[start + pd.Timedelta(hours=1), "v"], 450.0)
        self.assertEqual(result.loc[start + pd.Timedelta(hours=1), "phi_target"], 10.0)
        self.assertTrue(result.loc[start + pd.Timedelta(hours=3):start + pd.Timedelta(hours=9), "v"].isna().all())
        self.assertTrue(result.loc[start + pd.Timedelta(hours=3):start + pd.Timedelta(hours=9), "phi_target"].isna().all())

    def test_stereo_gap_limit_is_independent_per_column(self):
        start = pd.Timestamp("2018-01-01 00:00")
        source = pd.DataFrame(
            {
                "V": [400.0, np.nan, 600.0],
                "radialDistance": [23451.0] * 3,
                "heliographicLatitude": [0.0, 10.0, 20.0],
                "heliographicLongitude": [0.0, 20.0, 40.0],
            },
            index=pd.DatetimeIndex(
                [start, start + pd.Timedelta(hours=4), start + pd.Timedelta(hours=8)],
                tz="UTC",
            ),
        )
        target = pd.date_range(start, periods=9, freq="1h")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stereo.parquet"
            source.to_parquet(path)
            result = load_stereo_a_frame(target, "1h", stereo_path=path)
        self.assertTrue(np.isnan(result.loc[start + pd.Timedelta(hours=2), "v"]))
        self.assertEqual(result.loc[start + pd.Timedelta(hours=2), "phi_target"], 10.0)


if __name__ == "__main__":
    unittest.main()
