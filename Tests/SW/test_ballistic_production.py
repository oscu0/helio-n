import os
import unittest

import numpy as np
import pandas as pd

os.environ.setdefault("SUNPY_CONFIGDIR", "/tmp/helio_n_sunpy")

from Library.SW.Ballistic import cube_stats, propagate_continuous_boundary
from Library.SW.Constants import SOLAR_RADIUS_KM
from Library.SW.Coords import build_centered_time_axis, compute_rotation_state
from Tests.SW.reference import propagate_continuous_reference


OUTPUT_STEP_MINUTES = 60
ROTATION = compute_rotation_state(phi_step_minutes=120)


def propagate(source, time_axis, phi_axis, radius_axis, maximum_gap_hours=12):
    return propagate_continuous_boundary(
        df_v_run=source.to_frame(name="v"),
        time_axis=time_axis,
        phi_axis=phi_axis,
        r_axis=radius_axis,
        rotation_state=ROTATION,
        r0=20.0,
        output_step_minutes=OUTPUT_STEP_MINUTES,
        maximum_source_gap_hours=maximum_gap_hours,
        show_progress=False,
    )[0]


class BallisticProductionTests(unittest.TestCase):
    def test_cube_stats_does_not_replace_the_speed_cube(self):
        speed = np.array(
            [[[300.0, 350.0, np.nan]], [[310.0, 300.0, 450.0]]],
            dtype=np.float32,
        )
        summary = cube_stats(speed, slow_sw_speed=[300.0, 310.0])
        np.testing.assert_array_equal(
            summary.slow_wind_mask,
            [[[True, False, False]], [[True, False, False]]],
        )
        self.assertEqual(summary.speed_range, (300.0, 450.0))
        self.assertEqual(summary.filled_cells, 5)
        self.assertEqual(summary.slow_cells, 2)
        self.assertEqual(summary.non_slow_cells, 3)
        self.assertAlmostEqual(summary.non_slow_fraction_filled, 0.6)
        self.assertFalse(hasattr(summary, "V_grid"))
        self.assertTrue(np.isnan(speed[0, 0, 2]))

    def test_adjacent_intervals_share_one_center_lattice(self):
        start = pd.Timestamp("2020-01-01 00:10:00")
        boundary = start + pd.Timedelta(hours=3, minutes=10)
        end = boundary + pd.Timedelta(hours=3, minutes=20)
        left = build_centered_time_axis(
            start,
            boundary,
            60,
        )
        right = build_centered_time_axis(boundary, end, 60)
        actual = left.append(right)
        expected = pd.DatetimeIndex(
            [
                pd.Timestamp("2020-01-01 01:00:00"),
                pd.Timestamp("2020-01-01 02:00:00"),
                pd.Timestamp("2020-01-01 03:00:00"),
                pd.Timestamp("2020-01-01 04:00:00"),
                pd.Timestamp("2020-01-01 05:00:00"),
                pd.Timestamp("2020-01-01 06:00:00"),
            ]
        )
        pd.testing.assert_index_equal(actual, expected)

    def test_hourly_knots_fill_hourly_inner_boundary(self):
        start = pd.Timestamp("2020-01-01 00:00:00")
        source = pd.Series(
            [300.0, 600.0, 450.0],
            index=pd.date_range(start, periods=3, freq="1h"),
        )
        time_axis = pd.date_range(start, periods=3, freq="1h")
        cube = propagate(source, time_axis, [0.0], [20.0])
        self.assertEqual(cube.dtype, np.float32)
        self.assertTrue(np.isfinite(cube[:, 0, 0]).all())
        self.assertGreater(float(cube[1, 0, 0]), float(cube[0, 0, 0]))
        self.assertLess(float(cube[2, 0, 0]), float(cube[1, 0, 0]))

    def test_rarefaction_is_filled_without_white_bins(self):
        start = pd.Timestamp("2020-01-01 00:00:00")
        source = pd.Series(
            [600.0, 300.0],
            index=[start, start + pd.Timedelta(hours=1)],
        )
        time_axis = pd.date_range(start, periods=2, freq="1h")
        cube = propagate(source, time_axis, [0.0], [21.0])
        finite_indices = np.flatnonzero(np.isfinite(cube[:, 0, 0]))
        filled_span = cube[finite_indices[0] : finite_indices[-1] + 1, 0, 0]
        self.assertTrue(np.isfinite(filled_span).all())
        self.assertTrue(np.all(np.diff(filled_span) <= 0.0))

    def test_compression_chooses_fastest_continuous_root(self):
        start = pd.Timestamp("2020-01-01 00:00:00")
        source = pd.Series(
            [300.0, 800.0],
            index=[start, start + pd.Timedelta(hours=1)],
        )
        time_axis = pd.date_range(start, periods=5, freq="1h")
        radius_axis = [23.0]
        actual = propagate(source, time_axis, [0.0], radius_axis)
        expected = propagate_continuous_reference(
            source=source,
            time_axis=time_axis,
            phi_axis=[0.0],
            radius_axis=radius_axis,
            omega_degrees_s=ROTATION.omega,
            launch_radius_rsun=20.0,
            solar_radius_km=SOLAR_RADIUS_KM,
            output_step="60min",
        )
        np.testing.assert_array_equal(
            np.isfinite(actual),
            np.isfinite(expected),
        )
        np.testing.assert_allclose(
            actual,
            expected,
            atol=0.05,
            equal_nan=True,
        )
        target_index = 2
        self.assertGreater(float(actual[target_index, 0, 0]), 600.0)

    def test_exactly_twelve_hour_source_segment_is_filled(self):
        start = pd.Timestamp("2020-01-01 00:00:00")
        source = pd.Series(
            [300.0, 600.0],
            index=[start, start + pd.Timedelta(hours=12)],
        )
        time_axis = pd.date_range(start, periods=13, freq="1h")
        forward = propagate(source, time_axis, [0.0], [20.0])
        reverse = propagate(source.iloc[::-1], time_axis, [0.0], [20.0])
        self.assertTrue(np.isfinite(forward[:, 0, 0]).all())
        np.testing.assert_array_equal(forward, reverse)

    def test_over_limit_source_gap_remains_unfilled(self):
        start = pd.Timestamp("2020-01-01 00:00:00")
        source = pd.Series(
            [400.0, 500.0, 600.0],
            index=[
                start,
                start + pd.Timedelta(hours=13),
                start + pd.Timedelta(hours=14),
            ],
        )
        time_axis = pd.date_range(start, periods=15, freq="1h")
        cube = propagate(source, time_axis, [0.0], [20.0])
        missing = cube[1:13, 0, 0]
        self.assertTrue(np.isnan(missing).all())
        self.assertTrue(np.isfinite(cube[13:, 0, 0]).all())

    def test_longitude_is_an_exact_time_shift(self):
        start = pd.Timestamp("2020-01-01 00:00:00")
        source = pd.Series(
            [300.0, 600.0, 450.0],
            index=pd.date_range(start, periods=3, freq="1h"),
        )
        time_axis = pd.date_range(start, periods=5, freq="1h")
        cube = propagate(
            source,
            time_axis,
            [0.0, ROTATION.phi_step],
            [20.0],
        )
        delay_steps = 2
        np.testing.assert_array_equal(
            cube[delay_steps:, 1, 0],
            cube[:-delay_steps, 0, 0],
        )


if __name__ == "__main__":
    unittest.main()
