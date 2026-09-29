import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from Library.SW.Report import CSV_COLUMNS, build_hourly_csv_frame, build_hourly_report_frame, write_csv


class CsvReportTests(unittest.TestCase):
    def test_hourly_csv_contract_contains_native_fields_and_relative_hee_angles(self):
        index = pd.date_range("2020-01-01", periods=2, freq="1h")
        reproduction = pd.DataFrame(
            {
                ("input", "ch_area", "ch_relative_area"): [12.0, 14.0],
            },
            index=index,
        )
        ace = pd.DataFrame(
            {
                "v_predict": [350.0, 351.0],
                "v_real": [400.0, 401.0],
                "N": [5.0, 5.1],
                "t": [100000.0, 100100.0],
                "b": [5.0, 5.1],
                "b_x_gse": [1.0, 1.1],
                "b_y_gse": [2.0, 2.1],
                "b_z_gse": [3.0, 3.1],
                "x_hee_au": [1.0, 1.0],
                "y_hee_au": [0.0, 0.0],
                "z_hee_au": [0.0, 0.0],
            },
            index=index,
        )
        stereo = pd.DataFrame(
            {
                "v_predict": [450.0, 451.0],
                "v_real": [500.0, 501.0],
                "b_r_rtn": [7.0, 8.0],
                "b_t_rtn": [2.0, 3.0],
                "b_n_rtn": [-1.0, -2.0],
                "x_hee_au": [0.0, 0.0],
                "y_hee_au": [1.0, 1.0],
                "z_hee_au": [1.0, 1.0],
            },
            index=index,
        )
        result = build_hourly_csv_frame(
            reproduction_frame=reproduction,
            comparison_frames={"ace": ace, "stereo_a": stereo},
            start_dt=index[0],
            end_dt=index[-1] + pd.Timedelta(hours=1),
        )
        self.assertEqual(list(result.columns), CSV_COLUMNS)
        self.assertEqual(result.loc[index[0], "ace_speed_km_s"], 400.0)
        self.assertEqual(
            result.loc[index[0], "stereo_hee_relative_to_ace_longitude_deg"],
            135.0,
        )
        self.assertAlmostEqual(
            result.loc[index[0], "stereo_hee_relative_to_ace_latitude_deg"],
            np.degrees(np.arcsin(1.0 / np.sqrt(3.0))),
        )
        self.assertTrue(np.isnan(result.loc[index[0], "stereo_density_cm3"]))
        self.assertEqual(result.loc[index[0], "ace_bx_gse_nT"], 1.0)
        self.assertEqual(result.loc[index[0], "stereo_br_rtn_nT"], 7.0)
        self.assertTrue(result["stereo_bx_gse_nT"].isna().all())
        self.assertTrue(result["ace_br_rtn_nT"].isna().all())

    def test_reports_do_not_relabel_ace_at_earth(self):
        for builder in (build_hourly_csv_frame, build_hourly_report_frame):
            with self.subTest(builder=builder.__name__):
                with self.assertRaisesRegex(AssertionError, "native ACE"):
                    builder(
                        pd.DataFrame(), {"ace_earth": pd.DataFrame()},
                        "2020-01-01", "2020-01-02",
                    )

    def test_csv_rejects_unframed_archived_magnetic_components(self):
        index = pd.date_range("2020-01-01", periods=1, freq="1h")
        reproduction = pd.DataFrame(
            {("input", "ch_area", "ch_relative_area"): [1.0]}, index=index,
        )
        with self.assertRaisesRegex(AssertionError, "lack a frame"):
            build_hourly_csv_frame(
                reproduction, {"ace": pd.DataFrame({"b_x": [1.0]}, index=index)},
                "2020-01-01", "2020-01-02",
            )

    def test_write_csv_round_trip(self):
        frame = pd.DataFrame({column: [1.0] for column in CSV_COLUMNS})
        frame["datetime_utc"] = pd.Timestamp("2020-01-01")
        with tempfile.TemporaryDirectory() as directory:
            path = write_csv(frame, Path(directory) / "export.csv")
            loaded = pd.read_csv(path)
        self.assertEqual(list(loaded.columns), CSV_COLUMNS)
        self.assertEqual(len(loaded), 1)


if __name__ == "__main__":
    unittest.main()
