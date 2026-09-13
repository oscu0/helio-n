import json
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("SUNPY_CONFIGDIR", "/tmp/helio_n_sunpy")

import numpy as np
import pandas as pd

from Library.SW.Archive import cr_bounds, iter_crs, load_cr, load_cube, load_inputs, load_series, write_cr
from Library.SW.Coords import build_centered_time_axis
from Library.SW.Visualization import _format_title, build_satellite_comparison_frame, select_satellite_frames


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.cr = 2203
        self.boundary = cr_bounds(self.cr)[1]
        self.phi = np.array([0.0, 180.0], dtype=np.float32)
        self.radius = np.array([20.0, 215.0], dtype=np.float32)

    def tearDown(self):
        self.temp.cleanup()

    def write_sample(self, cr, value):
        start, end = cr_bounds(cr)
        times = build_centered_time_axis(start, end, 60)
        speed = np.full((len(times), 2, 2), value, dtype=np.float32)
        speed[0, 0, 0] = np.nan
        slow = np.zeros_like(speed, dtype=bool)
        slow[-1, 1, 1] = True
        inputs = pd.DataFrame({"dt": times, "ch_relative_area": 0.0})
        prepared = inputs.assign(v_empirical=value)
        series = pd.DataFrame({"speed": np.arange(len(times))}, index=times)
        return write_cr(
            self.root, cr, times, self.phi, self.radius,
            speed, slow, inputs, prepared, series,
            {"test": True, "output_step_minutes": 60},
        )

    def test_round_trip_and_no_overwrite(self):
        directory = self.write_sample(self.cr, 410.0)
        cube = load_cr(self.root, self.cr)
        self.assertEqual(cube.speed.dtype, np.float32)
        self.assertEqual(cube.is_slow_wind.dtype, np.bool_)
        self.assertEqual(cube.speed.dims, ("time", "phi", "r"))
        self.assertTrue(np.isnan(cube.speed.values[0, 0, 0]))
        self.assertTrue(cube.is_slow_wind.values[-1, 1, 1])
        self.assertEqual(json.loads((directory / "manifest.json").read_text())["cr"], self.cr)
        self.assertEqual(len(load_inputs(self.root, self.cr)), len(cube.time))
        self.assertTrue((load_inputs(self.root, self.cr, prepared=True)["v_empirical"] == 410.0).all())
        self.assertEqual(len(pd.read_parquet(self.root / "index.parquet")), 1)
        trajectory = pd.DataFrame(
            {"phi_target": 0.0, "r_target": 215.0},
            index=pd.DatetimeIndex(cube.time.values),
        )
        sampled = build_satellite_comparison_frame(
            time_axis=trajectory.index,
            phi_axis=cube.phi.values,
            r_axis=cube.r.values,
            grid_raw=cube.speed.values,
            slow_sw_pred_mask=cube.is_slow_wind.values,
            df_sat=trajectory,
        )
        self.assertTrue((sampled["v_predict"] == 410.0).all())
        with self.assertRaises(AssertionError):
            self.write_sample(self.cr, 420.0)

    def test_partial_rotation_is_rejected(self):
        start, end = cr_bounds(self.cr)
        times = build_centered_time_axis(start, end, 60)[:2]
        values = np.full((len(times), 2, 2), 400.0, dtype=np.float32)
        inputs = pd.DataFrame({"dt": times})
        with self.assertRaisesRegex(AssertionError, "every rounded-core time centre"):
            write_cr(
                self.root, self.cr, times, self.phi, self.radius,
                values, np.zeros_like(values, dtype=bool),
                inputs, inputs, pd.DataFrame(index=times),
                {"output_step_minutes": 60},
            )

    def test_rounded_half_open_boundary_and_context(self):
        self.write_sample(self.cr, 410.0)
        self.write_sample(self.cr + 1, 420.0)
        start = self.boundary - pd.Timedelta(hours=2)
        end = self.boundary + pd.Timedelta(hours=2)
        self.assertEqual(list(iter_crs(start, end)), [self.cr, self.cr + 1])
        cube = load_cube(self.root, start, end)
        series = load_series(self.root, start, end)
        self.assertEqual(len(cube.time), 4)
        self.assertEqual(len(series), 4)
        self.assertTrue(pd.DatetimeIndex(cube.time.values).is_unique)
        self.assertTrue(series.index.is_unique)
        self.assertTrue((pd.DatetimeIndex(cube.time.values) < end).all())
        self.assertTrue((pd.DatetimeIndex(cube.time.values) >= start).all())

        first_next_center = pd.Timestamp(load_cr(self.root, self.cr + 1).time.values[0])
        next_cube = load_cube(self.root, first_next_center, end, context_frames=1)
        next_series = load_series(self.root, first_next_center, end)
        self.assertEqual(pd.Timestamp(next_cube.time.values[0]), pd.Timestamp(cube.time.values[1]))
        self.assertEqual(next_series.index[0], first_next_center)
        self.assertLess(pd.Timestamp(next_cube.time.values[0]), self.boundary)

    def test_rounded_cr_bounds_and_utc_input(self):
        start, end = cr_bounds(self.cr)
        self.assertEqual(start, pd.Timestamp("2018-04-19 05:00"))
        self.assertEqual(end, pd.Timestamp("2018-05-16 11:00"))
        self.assertEqual(end, cr_bounds(self.cr + 1)[0])
        self.assertEqual(list(iter_crs(end, end + pd.Timedelta(hours=1))), [self.cr + 1])
        self.assertEqual(list(iter_crs(start, end)), [self.cr])
        self.assertIn("CR2203", _format_title(end - pd.Timedelta(hours=1)))
        self.assertIn("CR2204", _format_title(end))
        self.assertEqual(
            list(iter_crs(start.tz_localize("UTC"), end.tz_localize("UTC"))),
            [self.cr],
        )

    def test_satellite_selection_is_generic_and_ordered(self):
        frames = {name: pd.DataFrame() for name in ("ace_earth", "stereo_a", "solar_orbiter", "psp")}
        self.assertEqual(
            list(select_satellite_frames(frames, ["psp", "ace_earth"])),
            ["psp", "ace_earth"],
        )
        self.assertEqual(select_satellite_frames(frames, []), {})
        with self.assertRaisesRegex(AssertionError, "absent"):
            select_satellite_frames(frames, ["unknown"])

    def test_satellite_target_does_not_bridge_long_position_gaps(self):
        start = pd.Timestamp("2018-01-01 00:00")
        times = pd.date_range(start, periods=11, freq="1h")
        positions = pd.DataFrame(
            {"phi_target": [0.0, 20.0, 40.0], "r_target": [215.0] * 3},
            index=times[[0, 2, 10]],
        )
        speed = np.full((len(times), 2, 2), 400.0, dtype=np.float32)
        sampled = build_satellite_comparison_frame(
            time_axis=times,
            phi_axis=self.phi,
            r_axis=self.radius,
            grid_raw=speed,
            slow_sw_pred_mask=np.zeros_like(speed, dtype=bool),
            df_sat=positions,
        )
        self.assertTrue(np.isfinite(sampled.loc[times[1], "v_predict"]))
        self.assertTrue(np.isnan(sampled.loc[times[3], "v_predict"]))
        self.assertTrue(np.isnan(sampled.loc[times[3], "phi_target"]))


if __name__ == "__main__":
    unittest.main()
