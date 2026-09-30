import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook

from Library.SW.Report import (
    CSV_COLUMNS,
    VARIABLE_DEFINITIONS_SHEET,
    build_hourly_csv_frame,
    build_hourly_report_frame,
    build_satellite_data_frames,
    write_csv,
    write_satellite_data_workbook,
)


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

    def test_minimal_satellite_workbook_has_one_flagged_sheet_per_satellite(self):
        index = pd.date_range("2020-01-01", periods=2, freq="1h")
        reproduction = pd.DataFrame(
            {("input", "ch_area", "ch_relative_area"): [12.0, 14.0]},
            index=index,
        )
        ace = pd.DataFrame(
            {
                "v_predict": [350.0, 351.0], "v_predict_raw": [340.0, 341.0],
                "v_swx": [330.0, 331.0],
                "v_real": [400.0, 401.0], "N": [5.0, 5.1], "t": [1e5, 1.1e5],
                "b": [5.0, 5.1], "b_x_gse": [1.0, 1.0], "b_y_gse": [2.0, 2.0],
                "b_z_gse": [3.0, 3.0], "x_hee_au": [1.0, 1.0],
                "y_hee_au": [0.0, 0.0], "z_hee_au": [0.0, 0.0],
                "hee_beta_deg": [0.0, 0.0], "sdo_observation_age_days": [0.0, 11.0],
            }, index=index,
        )
        stereo = pd.DataFrame(
            {
                "v_predict": [450.0, 451.0], "v_predict_raw": [440.0, 441.0],
                "v_real": [500.0, 501.0], "N": [6.0, 6.1], "t": [2e5, 2.1e5],
                "b": [6.0, 6.1], "b_x_gse": [1.0, 1.0], "b_y_gse": [2.0, 2.0],
                "b_z_gse": [3.0, 3.0], "x_hee_au": [0.0, 0.0],
                "y_hee_au": [1.0, 1.0], "z_hee_au": [0.0, 0.0],
                "hee_beta_deg": [0.0, 11.0], "sdo_observation_age_days": [0.0, 0.0],
            }, index=index,
        )
        frames = build_satellite_data_frames(
            reproduction, {"ace": ace, "stereo_a": stereo}, index[0], index[-1] + pd.Timedelta(hours=1)
        )
        self.assertEqual(list(frames), ["ace", "stereo_a"])
        self.assertEqual(frames["ace"].loc[index[0], "relative_latitude_to_ace_deg"], 0.0)
        self.assertEqual(frames["ace"].loc[index[0], "relative_longitude_to_ace_deg"], 0.0)
        self.assertEqual(frames["stereo_a"].loc[index[0], "relative_longitude_to_ace_deg"], 135.0)
        self.assertEqual(frames["ace"].loc[index[0], "v_swx"], 330.0)
        self.assertNotIn("v_swx", frames["stereo_a"].columns)
        self.assertTrue(frames["ace"]["sdo_age_over_10_days"].iloc[1])
        self.assertTrue(frames["stereo_a"]["hee_beta_over_10_deg"].iloc[1])

        with tempfile.TemporaryDirectory() as directory:
            path = write_satellite_data_workbook(frames, Path(directory) / "satellites.xlsx")
            workbook = load_workbook(path)
        self.assertEqual(workbook.sheetnames, ["ace", "stereo_a", VARIABLE_DEFINITIONS_SHEET])
        ace_sheet = workbook["ace"]
        stereo_sheet = workbook["stereo_a"]
        ace_headers = [cell.value for cell in ace_sheet[1]]
        stereo_headers = [cell.value for cell in stereo_sheet[1]]
        self.assertIn("v_forecast", ace_headers)
        self.assertIn("v_forecast_slow_sw_model", ace_headers)
        self.assertIn("v_forecast_swx", ace_headers)
        self.assertNotIn("v_forecast_swx", stereo_headers)
        self.assertEqual(ace_sheet.cell(2, ace_headers.index("v_forecast") + 1).value, 340.0)
        self.assertEqual(
            ace_sheet.cell(2, ace_headers.index("v_forecast_slow_sw_model") + 1).value,
            350.0,
        )
        swx_col = ace_headers.index("v_forecast_swx") + 1
        self.assertEqual(ace_sheet.cell(2, swx_col).value, 330.0)
        definitions = workbook[VARIABLE_DEFINITIONS_SHEET]
        definition_rows = {
            definitions.cell(row, 1).value: definitions.cell(row, 3).value
            for row in range(2, definitions.max_row + 1)
        }
        self.assertIn("v_forecast_swx", definition_rows)
        self.assertIn("SWX", definition_rows["v_forecast_swx"])
        age_col = ace_headers.index("sdo_age_over_10_days") + 1
        beta_col = stereo_headers.index("hee_beta_over_10_deg") + 1
        self.assertEqual(ace_sheet.cell(3, 1).fill.fgColor.rgb[-6:], "FFFF00")
        self.assertEqual(stereo_sheet.cell(3, 1).fill.fgColor.rgb[-6:], "F4CCCC")
        self.assertEqual(ace_sheet.cell(3, age_col).fill.fgColor.rgb[-6:], "FFFF00")
        self.assertEqual(stereo_sheet.cell(3, beta_col).fill.fgColor.rgb[-6:], "F4CCCC")


if __name__ == "__main__":
    unittest.main()
