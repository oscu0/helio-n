import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("SUNPY_CONFIGDIR", "/tmp/helio_n_sunpy")

from Library.SW.Constants import SOLAR_RADIUS_KM
from Tests.SW.reference import (
    ballistic_arrival_time,
    centered_bin_index,
    propagate_analytic_reference,
)


OMEGA_DEGREES_S = 360.0 / (27.2753 * 24.0 * 3600.0)


class BallisticContractTests(unittest.TestCase):
    def test_center_labeled_time_bins(self):
        center = pd.Timestamp("2020-01-01 00:00:00")
        left_edge = center - pd.Timedelta(minutes=1)
        right_edge = center + pd.Timedelta(minutes=1)
        self.assertEqual(centered_bin_index(left_edge, center, "2min"), 0)
        self.assertEqual(centered_bin_index(center, center, "2min"), 0)
        self.assertEqual(
            centered_bin_index(
                right_edge - pd.Timedelta(nanoseconds=1),
                center,
                "2min",
            ),
            0,
        )
        self.assertEqual(
            centered_bin_index(right_edge, center, "2min"),
            1,
        )
        self.assertEqual(
            centered_bin_index(
                left_edge - pd.Timedelta(nanoseconds=1),
                center,
                "2min",
            ),
            -1,
        )

    def test_arrival_time_matches_ballistic_equation(self):
        launch = pd.Timestamp("2020-01-01 00:00:00")
        speed = 500.0
        radius = 20.0 + speed * 600.0 / SOLAR_RADIUS_KM
        phi = OMEGA_DEGREES_S * 300.0
        arrival = ballistic_arrival_time(
            launch_time=launch,
            speed_km_s=speed,
            phi_degrees=phi,
            omega_degrees_s=OMEGA_DEGREES_S,
            radius_rsun=radius,
            launch_radius_rsun=20.0,
            solar_radius_km=SOLAR_RADIUS_KM,
        )
        self.assertLess(
            abs(arrival - (launch + pd.Timedelta(minutes=15))),
            pd.Timedelta(microseconds=1),
        )

    def test_2000_km_s_parcel_reaches_every_requested_shell(self):
        start = pd.Timestamp("2020-01-01 00:00:00")
        radius_axis = np.arange(20.0, 216.0, 1.5)
        cube = propagate_analytic_reference(
            launch_times=[start],
            speeds=[2000.0],
            phi_axis=[0.0],
            radius_axis=radius_axis,
            output_first_center=start,
            output_bins=700,
            bin_width="2min",
            omega_degrees_s=OMEGA_DEGREES_S,
            launch_radius_rsun=20.0,
            solar_radius_km=SOLAR_RADIUS_KM,
        )
        self.assertTrue(np.isfinite(cube[:, 0, :]).any(axis=0).all())

    def test_inner_boundary_reproduces_launch_series(self):
        start = pd.Timestamp("2020-01-01 00:00:00")
        launch_times = pd.date_range(start, periods=4, freq="2min")
        speeds = np.array([300.0, 450.0, np.nan, 600.0])
        cube = propagate_analytic_reference(
            launch_times=launch_times,
            speeds=speeds,
            phi_axis=[0.0],
            radius_axis=[20.0],
            output_first_center=start,
            output_bins=4,
            bin_width="2min",
            omega_degrees_s=OMEGA_DEGREES_S,
            launch_radius_rsun=20.0,
            solar_radius_km=SOLAR_RADIUS_KM,
        )
        np.testing.assert_allclose(cube[:, 0, 0], speeds, equal_nan=True)

    def test_cross_boundary_collision_keeps_faster_parcel_in_any_order(self):
        boundary = pd.Timestamp("2020-01-01 00:00:00")
        launch_times = pd.DatetimeIndex(
            [boundary - pd.Timedelta(minutes=1), boundary + pd.Timedelta(minutes=1)]
        )
        speeds = np.array([400.0, 800.0])
        collision_radius = 20.0 + 96_000.0 / SOLAR_RADIUS_KM

        def run(order):
            return propagate_analytic_reference(
                launch_times=launch_times[order],
                speeds=speeds[order],
                phi_axis=[0.0],
                radius_axis=[collision_radius],
                output_first_center=boundary,
                output_bins=3,
                bin_width="2min",
                omega_degrees_s=OMEGA_DEGREES_S,
                launch_radius_rsun=20.0,
                solar_radius_km=SOLAR_RADIUS_KM,
            )

        forward = run(np.array([0, 1]))
        reverse = run(np.array([1, 0]))
        np.testing.assert_array_equal(forward, reverse)
        self.assertEqual(float(forward[2, 0, 0]), 800.0)

    def test_outputs_stay_within_finite_input_speed_range(self):
        start = pd.Timestamp("2020-01-01 00:00:00")
        speeds = np.array([325.0, 510.0, 700.0])
        cube = propagate_analytic_reference(
            launch_times=pd.date_range(start, periods=3, freq="2min"),
            speeds=speeds,
            phi_axis=[0.0, 15.0],
            radius_axis=[20.0, 21.0],
            output_first_center=start,
            output_bins=2000,
            bin_width="2min",
            omega_degrees_s=OMEGA_DEGREES_S,
            launch_radius_rsun=20.0,
            solar_radius_km=SOLAR_RADIUS_KM,
        )
        finite = cube[np.isfinite(cube)]
        self.assertGreater(len(finite), 0)
        self.assertGreaterEqual(float(finite.min()), float(speeds.min()))
        self.assertLessEqual(float(finite.max()), float(speeds.max()))

    def test_frame_artifact_round_trip(self):
        frame = np.array([[300.0, np.nan], [425.0, 700.0]], dtype=np.float32)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frames.npz"
            np.savez_compressed(path, candidate=frame)
            with np.load(path) as saved:
                np.testing.assert_array_equal(saved["candidate"], frame)


if __name__ == "__main__":
    unittest.main()
