import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from Library.SW.Stats import build_sw_forecast_stats, build_sw_forecast_yearly_stats


class SummaryStatsTests(unittest.TestCase):
    def setUp(self):
        self.start = pd.Timestamp("2019-12-31 22:00")
        self.end = pd.Timestamp("2020-01-01 04:00")
        index = pd.date_range(self.start, self.end, freq="1h", inclusive="left")
        ace = pd.DataFrame(
            {
                "v_real": [300., 400., 500., 600., np.nan, 800.],
                "v_predict_raw": [320., 380., 700., 550., 900., np.nan],
                "v_1cr_ago": [310., 390., 510., 590., 900., 790.],
                "v_predict": [320., 380., 700., 550., 900., np.nan],
                "v_swx": [300., 400., 500., 600., np.nan, 800.],
                "slow_sw_patch_mask": False,
            }, index=index,
        )
        self.frames = {"ace": ace, "psp": ace * np.nan}
        self.labels = {"ace": "ACE", "psp": "PSP"}
        self.windows = pd.DataFrame(
            {"start": [pd.Timestamp("2019-12-31 23:00")],
             "end": [pd.Timestamp("2020-01-01 00:00")]}
        )

    def test_global_and_yearly_scores_use_pairs_and_preserve_icme_and_missingness(self):
        with patch("Library.SW.Stats.load_sat_icme_windows", return_value=self.windows):
            global_stats = build_sw_forecast_stats(
                self.frames, self.start, self.end, self.labels
            )
            yearly = build_sw_forecast_yearly_stats(
                self.frames, self.start, self.end, self.labels
            )

        keys = ["sat", "comparison", "regime"]
        pd.testing.assert_series_equal(
            yearly.groupby(keys).n_samples.sum().sort_index(),
            global_stats.set_index(keys).n_samples.sort_index(),
        )
        raw = yearly.loc[
            (yearly.sat == "ACE") & (yearly.comparison == "raw_vs_observed")
            & (yearly.regime == "all_sw")
        ].set_index("year")
        self.assertEqual(raw.index.tolist(), [2019, 2020])
        self.assertEqual(raw.loc[2019, "start_utc"], self.start)
        self.assertEqual(raw.loc[2019, "end_utc"], pd.Timestamp("2020-01-01"))
        self.assertEqual(raw.loc[2020, "start_utc"], pd.Timestamp("2020-01-01"))
        self.assertEqual(raw.loc[2020, "end_utc"], self.end)
        self.assertEqual(raw.n_samples.tolist(), [2, 2])
        self.assertEqual(raw.r.tolist(), [1., -1.])
        self.assertEqual(raw.rmse.tolist(), [20., round(np.sqrt(21250.), 1)])
        self.assertEqual(raw.mae.tolist(), [20., 125.])

        global_raw = global_stats.loc[
            (global_stats.sat == "ACE")
            & (global_stats.comparison == "raw_vs_observed")
            & (global_stats.regime == "all_sw")
        ].iloc[0]
        self.assertEqual(global_raw.n_samples, 4)
        self.assertAlmostEqual(
            global_raw.r,
            round(np.corrcoef([300., 400., 500., 600.], [320., 380., 700., 550.])[0, 1], 3),
        )
        self.assertNotEqual(global_raw.r, raw.r.mean())
        self.assertEqual(global_raw.rmse, round(np.sqrt(10825.), 1))
        self.assertEqual(global_raw.mae, 72.5)
        self.assertEqual(global_raw.bias, 37.5)

        no_icme = yearly.loc[
            (yearly.sat == "ACE") & (yearly.comparison == "raw_vs_observed")
            & (yearly.regime == "no_icme")
        ].set_index("year")
        self.assertEqual(no_icme.n_samples.tolist(), [1, 0])
        self.assertTrue(np.isnan(no_icme.loc[2019, "r"]))
        self.assertTrue(no_icme.loc[2020, ["r", "rmse", "mae", "bias"]].isna().all())
        empty = yearly.loc[yearly.sat == "PSP"]
        self.assertTrue(empty.n_samples.eq(0).all())
        self.assertTrue(empty[["r", "rmse", "mae", "bias"]].isna().all().all())

    def test_exclusive_january_first_end_does_not_create_an_extra_year(self):
        with patch("Library.SW.Stats.load_sat_icme_windows", return_value=self.windows):
            yearly = build_sw_forecast_yearly_stats(
                self.frames, self.start, pd.Timestamp("2020-01-01"), self.labels
            )
        self.assertEqual(yearly.year.unique().tolist(), [2019])


if __name__ == "__main__":
    unittest.main()
