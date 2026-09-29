import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from Library.SW.Config import (
    DEFAULT_ENABLED_SATELLITES,
    SATELLITE_CONFIGS,
    parse_satellite_ids,
)
from Library.SW.Inputs import load_ace_earth_frame, load_ace_frame, load_stereo_a_frame, normalize_satellite_frame
from Library.SW.Visualization import _build_hee_target_frame


class SatelliteConfigTests(unittest.TestCase):
    def test_registry_contains_planned_sources(self):
        self.assertEqual(
            set(SATELLITE_CONFIGS),
            {"ace", "ace_earth", "earth", "psp", "solo", "stereo_a", "stereo_b"},
        )
        self.assertEqual(parse_satellite_ids(None), list(DEFAULT_ENABLED_SATELLITES))
        self.assertEqual(list(DEFAULT_ENABLED_SATELLITES), ["ace", "stereo_a"])
        self.assertEqual(parse_satellite_ids("stereo_a,ace_earth"), ["stereo_a", "ace_earth"])
        self.assertEqual(SATELLITE_CONFIGS["ace"].label, "ACE")
        self.assertEqual(SATELLITE_CONFIGS["ace_earth"].label, "ACE @ Earth")

    def test_magnetic_frames_remain_distinct_and_missing_components_stay_missing(self):
        index = pd.date_range("2020-01-01", periods=2, freq="1h")
        for source_names, expected_names in (
            (("B_X_GSE", "B_Y_GSE", "B_Z_GSE"), ("b_x_gse", "b_y_gse", "b_z_gse")),
            (("Br", "Bt", "Bn"), ("b_r_rtn", "b_t_rtn", "b_n_rtn")),
        ):
            with self.subTest(source=source_names):
                source = pd.DataFrame(
                    {source_names[0]: [3., 3.], source_names[1]: [4., np.nan],
                     source_names[2]: [0., 0.]}, index=index,
                )
                result = normalize_satellite_frame(source, "sample")
                np.testing.assert_array_equal(result[list(expected_names)].values, source.values)
                self.assertEqual(result["b"].iloc[0], 5.)
                self.assertTrue(np.isnan(result["b"].iloc[1]))
                self.assertNotIn("b_x", result.columns)

    def test_ambiguous_magnetic_components_are_not_guessed(self):
        source = pd.DataFrame(
            {"Bx": [3.], "By": [4.], "Bz": [0.]},
            index=pd.date_range("2020-01-01", periods=1),
        )
        with self.assertRaisesRegex(AssertionError, "explicit frame"):
            normalize_satellite_frame(source, "ace")

    def test_native_ace_uses_common_fields_and_known_hee_position(self):
        start = pd.Timestamp("2018-11-13 00:00")
        source = pd.DataFrame(
            {
                "speed": [400.0],
                "temperature": [100000.0],
                "density": [5.0],
                "x_hee_au": [0.98],
                "y_hee_au": [0.01],
                "z_hee_au": [0.002],
            },
            index=pd.DatetimeIndex([start], name="date"),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ace.parquet"
            source.to_parquet(path)
            result = load_ace_frame(path)
        self.assertEqual(list(result[["v", "N", "t"]].iloc[0]), [400.0, 5.0, 100000.0])
        np.testing.assert_allclose(
            result[["x_hee_au", "y_hee_au", "z_hee_au"]].iloc[0].to_numpy(),
            [0.98, 0.01, 0.002],
        )
        self.assertEqual(result.attrs["coord_frame"], "HEE")
        self.assertEqual(result.attrs["label"], "ACE")

    def test_ace_fallback_converts_public_gse_ephemeris(self):
        start = pd.Timestamp("2018-11-13 00:00")
        source = pd.DataFrame(
            {
                "speed": [400.0],
                "temperature": [100000.0],
                "density": [5.0],
            },
            index=pd.DatetimeIndex([start], name="date"),
        )
        ephemeris = "Year DOY Secofday GSE_X(km) GSE_y(km) GSE_z(km)\n2018 317 0 1000 2000 3000\n"
        with tempfile.TemporaryDirectory() as directory:
            data_path = Path(directory) / "ace.parquet"
            ephemeris_path = Path(directory) / "ace_gse.txt"
            source.to_parquet(data_path)
            ephemeris_path.write_text(ephemeris)
            result = load_ace_frame(data_path, ephemeris_path)
        self.assertFalse(np.allclose(
            result[["x_hee_au", "y_hee_au", "z_hee_au"]].iloc[0].to_numpy(),
            [1.0, 0.0, 0.0],
        ))
        self.assertEqual(result.attrs["position_source"], str(ephemeris_path))

    def test_ace_at_earth_keeps_fixed_hee_position(self):
        start = pd.Timestamp("2018-11-13 00:00")
        source = pd.DataFrame(
            {"speed": [400.0], "temperature": [100000.0], "density": [5.0]},
            index=pd.DatetimeIndex([start], name="date"),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ace_at_earth.parquet"
            source.to_parquet(path)
            result = load_ace_earth_frame(path)
        np.testing.assert_allclose(
            result[["x_hee_au", "y_hee_au", "z_hee_au"]].iloc[0].to_numpy(),
            [1.0, 0.0, 0.0],
        )
        self.assertEqual(result.attrs["label"], "ACE @ Earth")

    def test_stereo_source_converts_hgs_coordinates_to_hee(self):
        start = pd.Timestamp("2018-11-13 00:00")
        source = pd.DataFrame(
            {
                "V": [400.0, 500.0],
                "radialDistance": [22428.0, 22428.0],
                "heliographicLatitude": [5.0, 5.0],
                "heliographicLongitude": [-100.0, -99.0],
            },
            index=pd.DatetimeIndex(
                [start, start + pd.Timedelta(hours=1)], tz="UTC", name="Epoch"
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stereo.parquet"
            source.to_parquet(path)
            result = load_stereo_a_frame(
                time_axis=pd.date_range(start, periods=2, freq="1h"),
                time_freq="1h",
                stereo_path=path,
            )
        self.assertEqual(result.attrs["coord_frame"], "HEE")
        self.assertEqual(result.attrs["source_coord_frame"], "HGS")
        self.assertNotEqual(result.loc[start, "phi_target"], -100.0)
        self.assertEqual(result.loc[start, "lat_hgs"], 5.0)

    def test_hgs_lookup_uses_source_longitude_not_hee_position(self):
        start = pd.Timestamp("2018-11-13 00:00")
        source = pd.DataFrame(
            {
                "phi_target": [-100.0],
                "r_target": [200.0],
                "x_hee_au": [0.5],
                "y_hee_au": [-0.5],
                "z_hee_au": [0.0],
            },
            index=pd.DatetimeIndex([start]),
        )
        source.attrs["coord_frame"] = "HGS"
        result = _build_hee_target_frame(
            df_sat=source,
            target_index=pd.DatetimeIndex([start]),
            phi_target=0.0,
            r_target=215.0,
        )
        self.assertEqual(result.loc[start, "phi_target"], -100.0)
        self.assertEqual(result.loc[start, "r_target"], 200.0)


if __name__ == "__main__":
    unittest.main()
