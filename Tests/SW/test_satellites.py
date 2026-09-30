import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from Library.SW.Config import (
    ALL_VALIDATION_SATELLITES,
    DEFAULT_ENABLED_SATELLITES,
    SATELLITE_CONFIGS,
    parse_satellite_ids,
)
from Library.SW.Constants import CARRINGTON_ROTATION_DAYS
from Library.SW.Inputs import (
    _hee_cartesian_from_hgs,
    add_sdo_observation_geometry,
    find_cdaweb_merged_product,
    load_ace_earth_frame,
    load_ace_frame,
    load_cdaweb_ace_frame,
    load_satellite_frames,
    load_stereo_a_frame,
    normalize_satellite_frame,
)
from Library.SW.Visualization import _build_hee_target_frame


class SatelliteConfigTests(unittest.TestCase):
    def test_registry_contains_planned_sources(self):
        self.assertEqual(
            set(SATELLITE_CONFIGS),
            {"ace", "ace_earth", "earth", "psp", "solo", "stereo_a", "stereo_b"},
        )
        self.assertEqual(parse_satellite_ids(None), list(DEFAULT_ENABLED_SATELLITES))
        self.assertEqual(list(DEFAULT_ENABLED_SATELLITES), ["ace", "stereo_a"])
        self.assertEqual(parse_satellite_ids("all"), list(ALL_VALIDATION_SATELLITES))
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

    def test_cdaweb_merged_products_are_the_validation_source(self):
        start = pd.Timestamp("2021-01-01 12:00")
        time_axis = pd.date_range(start, periods=2, freq="1h")
        source_index = pd.date_range(
            start - pd.Timedelta(hours=8),
            periods=18,
            freq="1h",
            tz="UTC",
            name="Epoch",
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "CDAWeb Archive"
            merged = root / "merged"
            merged.mkdir(parents=True)
            ace_path = merged / "ace_20210101T000000_20210103T000000.parquet"
            stereo_path = merged / "stereo-a_20210101T000000_20210103T000000.parquet"
            ace = pd.DataFrame(
                {
                    "B_X_GSE": 3.0,
                    "B_Y_GSE": 4.0,
                    "B_Z_GSE": 0.0,
                    "B": 5.0,
                    "V": 410.0,
                    "N_p": 6.0,
                    "X_GSE": 30.0,
                    "Y_GSE": 2.0,
                    "Z_GSE": -1.0,
                },
                index=source_index,
            )
            stereo = pd.DataFrame(
                {
                    "B_X_GSE": 1.0,
                    "B_Y_GSE": 2.0,
                    "B_Z_GSE": 2.0,
                    "B": 3.0,
                    "N_p": 7.0,
                    "V": 420.0,
                    "T_p": 90000.0,
                    "X_GSE": 20000.0,
                    "Y_GSE": -4000.0,
                    "Z_GSE": 300.0,
                    "radialDistance": 23000.0,
                },
                index=source_index,
            )
            stereo_b_index = pd.date_range(
                "2013-01-01", periods=18, freq="1h", tz="UTC", name="Epoch"
            )
            stereo_b = stereo.set_axis(stereo_b_index)
            psp = stereo.assign(V_X_GSE=10.0, V_Y_GSE=20.0, V_Z_GSE=30.0)
            solo = psp.copy()
            ace.to_parquet(ace_path)
            stereo.to_parquet(stereo_path)
            stereo_b.to_parquet(merged / "stereo-b_20100101T000000_20140927T170000.parquet")
            psp.to_parquet(merged / "psp_20180812T000000_20260101T000000.parquet")
            solo.to_parquet(merged / "solo_20200210T000000_20260101T000000.parquet")
            (root / "chunks" / "ace").mkdir(parents=True)

            frames = load_satellite_frames(
                list(ALL_VALIDATION_SATELLITES),
                time_axis=time_axis,
                time_freq="1h",
                validation_archive_root=root,
            )

            stereo_b_frame = load_satellite_frames(
                ["stereo_b"],
                time_axis=pd.date_range("2013-01-01 08:00", periods=2, freq="1h"),
                time_freq="1h",
                validation_archive_root=root,
            )["stereo_b"]

        ace_frame = frames["ace"]
        stereo_frame = frames["stereo_a"]
        self.assertEqual(ace_frame.loc[start, "v"], 410.0)
        self.assertEqual(ace_frame.loc[start, "N"], 6.0)
        self.assertEqual(ace_frame.loc[start, "b_x_gse"], 3.0)
        self.assertNotEqual(ace_frame.loc[start, "x_hee_au"], 1.0)
        self.assertEqual(ace_frame.attrs["source"], "CDAWeb merged ACE")
        self.assertEqual(ace_frame.attrs["position_source"], "merged GSE position in Earth radii")
        self.assertEqual(stereo_frame.loc[start, "v"], 420.0)
        self.assertEqual(stereo_frame.loc[start, "t"], 90000.0)
        self.assertEqual(stereo_frame.attrs["source_coord_frame"], "GSE")
        self.assertEqual(stereo_frame.attrs["source"], "CDAWeb merged STEREO-A")
        self.assertTrue(stereo_frame.attrs["archive_path"].endswith(stereo_path.name))
        for sat_id in ("psp", "solo"):
            self.assertEqual(frames[sat_id].loc[start, "v"], 420.0)
            self.assertEqual(frames[sat_id].loc[start, "v_x_gse"], 10.0)
            self.assertEqual(frames[sat_id].attrs["coord_frame"], "HEE")
        self.assertTrue(frames["stereo_b"]["v"].isna().all())
        self.assertEqual(stereo_b_frame.loc[pd.Timestamp("2013-01-01 08:00"), "v"], 420.0)
        self.assertEqual(stereo_b_frame.attrs["source"], "CDAWeb merged STEREO-B")

    def test_cdaweb_discovery_does_not_read_intermediate_chunks(self):
        start = pd.Timestamp("2018-01-01", tz="UTC")
        end = pd.Timestamp("2018-01-02", tz="UTC")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "CDAWeb Archive"
            chunk_dir = root / "chunks" / "ace"
            chunk_dir.mkdir(parents=True)
            (chunk_dir / "20180101T000000_20180103T000000.parquet").touch()
            with self.assertRaisesRegex(FileNotFoundError, "merged CDAWeb ace product"):
                find_cdaweb_merged_product(root, "ace", start, end)

    def test_merged_ace_interpolates_only_gaps_up_to_six_hours(self):
        start = pd.Timestamp("2021-01-01 00:00")
        time_axis = pd.date_range(start, periods=13, freq="1h")
        source_index = pd.date_range(
            start, periods=13, freq="1h", tz="UTC", name="Epoch"
        )
        speed = np.full(len(source_index), 500.0)
        speed[0] = 300.0
        speed[1:6] = np.nan
        speed[6] = 600.0
        density = np.full(len(source_index), 5.0)
        density[1:7] = np.nan
        x_gse = np.full(len(source_index), 30.0)
        x_gse[1:7] = np.nan
        source = pd.DataFrame(
            {
                "B_X_GSE": 3.0,
                "B_Y_GSE": 4.0,
                "B_Z_GSE": 0.0,
                "B": 5.0,
                "V": speed,
                "N_p": density,
                "X_GSE": x_gse,
                "Y_GSE": 2.0,
                "Z_GSE": -1.0,
            },
            index=source_index,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "CDAWeb Archive"
            merged = root / "merged"
            merged.mkdir(parents=True)
            path = merged / "ace_20201231T000000_20210102T000000.parquet"
            source.to_parquet(path)

            frame = load_cdaweb_ace_frame(root, time_axis, "1h")

        self.assertEqual(frame.index.tolist(), time_axis.tolist())
        self.assertEqual(frame.loc[start + pd.Timedelta(hours=3), "v"], 450.0)
        self.assertTrue(
            frame.loc[start + pd.Timedelta(hours=1):start + pd.Timedelta(hours=6), "N"].isna().all()
        )
        self.assertEqual(frame.loc[start + pd.Timedelta(hours=7), "N"], 5.0)
        self.assertTrue(
            np.isnan(frame.loc[start + pd.Timedelta(hours=3), "x_hee_au"])
        )
        self.assertTrue(np.isfinite(frame.loc[start + pd.Timedelta(hours=7), "x_hee_au"]))

    def test_sdo_age_and_beta_use_their_separate_geometries(self):
        index = pd.date_range("2020-01-01", periods=3, freq="1h")
        longitudes = np.array([0.0, 120.0, -120.0])
        age_positions = _hee_cartesian_from_hgs(
            index,
            longitudes,
            np.zeros(len(index)),
            np.ones(len(index)),
        )
        age_result = add_sdo_observation_geometry(age_positions)
        beta_positions = pd.DataFrame(
            {
                "x_hee_au": [1.0, 1.0, 1.0],
                "y_hee_au": [0.0, 0.0, 0.0],
                "z_hee_au": [0.0, np.tan(np.deg2rad(12.0)), np.tan(np.deg2rad(-11.0))],
            },
            index=index,
        )
        beta_result = add_sdo_observation_geometry(beta_positions)
        self.assertAlmostEqual(beta_result["hee_beta_deg"].iloc[0], 0.0, places=6)
        self.assertAlmostEqual(beta_result["hee_beta_deg"].iloc[1], 12.0, places=6)
        self.assertAlmostEqual(beta_result["hee_beta_deg"].iloc[2], -11.0, places=6)
        self.assertEqual(age_result["sdo_observation_age_days"].iloc[0], 0.0)
        rotation_days = CARRINGTON_ROTATION_DAYS
        self.assertAlmostEqual(
            age_result["sdo_observation_age_days"].iloc[1],
            30.0 / (360.0 / rotation_days),
            places=3,
        )
        self.assertAlmostEqual(
            age_result["sdo_observation_age_days"].iloc[2],
            150.0 / (360.0 / rotation_days),
            places=3,
        )

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
