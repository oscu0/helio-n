import unittest

import numpy as np
import pandas as pd

from Library.SW.Inputs import build_model_input_series


class LinearEmpiricalModel:
    def v_from_area(self, area, t, parameter_time):
        return 300.0 + np.asarray(area, dtype=float)


class InputContractTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
