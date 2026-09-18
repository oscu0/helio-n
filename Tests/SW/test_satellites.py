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
from Library.SW.Inputs import load_ace_earth_frame, load_stereo_a_frame
from Library.SW.Visualization import _build_hee_target_frame


class SatelliteConfigTests(unittest.TestCase):
    def test_registry_contains_planned_sources(self):
        self.assertEqual(
            set(SATELLITE_CONFIGS),
            {"ace", "ace_earth", "earth", "psp", "solo", "stereo_a", "stereo_b"},
        )
        self.assertEqual(parse_satellite_ids(None), list(DEFAULT_ENABLED_SATELLITES))
        self.assertEqual(parse_satellite_ids("stereo_a,ace_earth"), ["stereo_a", "ace_earth"])

    def test_ace_earth_uses_common_fields_and_known_hee_position(self):
        start = pd.Timestamp("2018-11-13 00:00")
        source = pd.DataFrame(
            {
                "speed": [400.0],
                "temperature": [100000.0],
                "density": [5.0],
            },
            index=pd.DatetimeIndex([start], name="date"),
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ace.parquet"
            source.to_parquet(path)
            result = load_ace_earth_frame(path)
        self.assertEqual(list(result[["v", "N", "t"]].iloc[0]), [400.0, 5.0, 100000.0])
        np.testing.assert_allclose(
            result[["x_hee_au", "y_hee_au", "z_hee_au"]].iloc[0].to_numpy(),
            [1.0, 0.0, 0.0],
        )
        self.assertEqual(result.attrs["coord_frame"], "HEE")

    def test_stereo_source_exposes_hgs_coordinates(self):
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
        self.assertEqual(result.attrs["coord_frame"], "HGS")
        self.assertEqual(result.loc[start, "phi_target"], -100.0)
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
