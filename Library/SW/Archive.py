"""Rounded-Carrington-rotation solar-wind archive I/O."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import h5py
import numpy as np
import pandas as pd
import xarray as xr
from sunpy.coordinates.sun import carrington_rotation_number, carrington_rotation_time

from Library.SW.Coords import build_centered_time_axis


def _utc_naive(value):
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_convert("UTC").tz_localize(None)
    return timestamp


def _astronomical_cr_bounds(cr):
    cr = int(cr)
    return (
        pd.Timestamp(carrington_rotation_time(cr).datetime),
        pd.Timestamp(carrington_rotation_time(cr + 1).datetime),
    )


def cr_bounds(cr, output_step_minutes=60):
    """Nearest-centre CR boundaries on the global UTC output lattice."""
    step = pd.Timedelta(minutes=int(output_step_minutes))
    assert step > pd.Timedelta(0)
    start, end = _astronomical_cr_bounds(cr)
    return start.round(step), end.round(step)


def iter_crs(start, end, output_step_minutes=60):
    """Yield rotations intersecting a half-open, rounded-core interval."""
    start = _utc_naive(start)
    end = _utc_naive(end)
    assert start < end
    cr = int(np.floor(carrington_rotation_number(start)))
    while start < cr_bounds(cr, output_step_minutes)[0]:
        cr -= 1
    while start >= cr_bounds(cr, output_step_minutes)[1]:
        cr += 1
    while True:
        cr_start, cr_end = cr_bounds(cr, output_step_minutes)
        if cr_start >= end:
            break
        if cr_end > start:
            yield cr
        cr += 1


def cr_for_time(time, output_step_minutes=60):
    time = _utc_naive(time)
    return next(iter_crs(time, time + pd.Timedelta(nanoseconds=1), output_step_minutes))


def _cr_dir(archive_root, cr):
    return Path(archive_root) / f"CR{int(cr):04d}"


def write_cr(
    archive_root,
    cr,
    time_axis,
    phi_axis,
    r_axis,
    speed,
    is_slow_wind,
    inputs,
    prepared_inputs,
    series,
    metadata,
):
    """Publish one complete, nonoverlapping CR; never overwrite an archive."""
    archive_root = Path(archive_root)
    archive_root.mkdir(parents=True, exist_ok=True)
    target = _cr_dir(archive_root, cr)
    assert not target.exists(), f"Archive already exists: {target}"
    cr_start, cr_end = cr_bounds(cr, metadata["output_step_minutes"])
    astronomical_start, astronomical_end = _astronomical_cr_bounds(cr)
    times = pd.DatetimeIndex(time_axis)
    assert len(times) > 0 and times.is_unique and times.is_monotonic_increasing
    expected = build_centered_time_axis(
        cr_start, cr_end, metadata["output_step_minutes"]
    )
    assert times.equals(expected), "A CR archive must contain every rounded-core time centre"
    phi = np.asarray(phi_axis, dtype=np.float32)
    radius = np.asarray(r_axis, dtype=np.float32)
    values = np.asarray(speed, dtype=np.float32)
    slow = np.asarray(is_slow_wind, dtype=bool)
    assert values.shape == slow.shape == (len(times), len(phi), len(radius))
    assert not (slow & ~np.isfinite(values)).any()
    assert series.index.is_unique
    assert (series.index >= cr_start).all() and (series.index < cr_end).all()

    manifest = {
        "schema_version": 1,
        "cr": int(cr),
        "start_utc": cr_start.isoformat(),
        "end_utc": cr_end.isoformat(),
        "astronomical_start_utc": astronomical_start.isoformat(),
        "astronomical_end_utc": astronomical_end.isoformat(),
        "time_convention": "UTC centre labels in nearest-output-step-rounded half-open CR interval",
        "axes": {"time": "UTC ns", "phi": "degrees", "r": "solar radii"},
        "shape": list(values.shape),
        "finite_fraction": float(np.isfinite(values).mean()),
        **metadata,
    }
    with TemporaryDirectory(prefix=f"CR{int(cr):04d}-", dir=archive_root) as temp_name:
        staging = Path(temp_name)
        with h5py.File(staging / "cube.h5", "w") as cube:
            cube.create_dataset("time_ns", data=times.asi8)
            cube.create_dataset("phi", data=phi)
            cube.create_dataset("r", data=radius)
            chunks = (min(24, len(times)), len(phi), len(radius))
            cube.create_dataset(
                "speed", data=values, chunks=chunks, compression="gzip",
                compression_opts=4, shuffle=True,
            )
            cube.create_dataset(
                "is_slow_wind", data=slow, chunks=chunks, compression="gzip",
                compression_opts=4, shuffle=True,
            )
        inputs.to_parquet(staging / "inputs.parquet", index=False)
        prepared_inputs.to_parquet(staging / "prepared_inputs.parquet", index=False)
        series.to_parquet(staging / "series.parquet")
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2))
        staging.rename(target)

    index_path = archive_root / "index.parquet"
    row = pd.DataFrame([{
        "cr": int(cr), "start_utc": cr_start, "end_utc": cr_end,
        "path": target.name, "frames": len(times),
        "finite_fraction": manifest["finite_fraction"],
    }])
    index = pd.read_parquet(index_path) if index_path.exists() else row.iloc[0:0]
    assert int(cr) not in set(index["cr"])
    index = pd.concat([index, row], ignore_index=True).sort_values("cr")
    index.to_parquet(index_path)
    return target


def load_cr(archive_root, cr):
    """Load one CR cube with named coordinates."""
    directory = _cr_dir(archive_root, cr)
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["cr"] == int(cr)
    with h5py.File(directory / "cube.h5", "r") as cube:
        time = pd.to_datetime(cube["time_ns"][:], utc=True).tz_localize(None)
        phi = cube["phi"][:]
        radius = cube["r"][:]
        speed = cube["speed"][:]
        slow = cube["is_slow_wind"][:]
    return xr.Dataset(
        {
            "speed": (("time", "phi", "r"), speed),
            "is_slow_wind": (("time", "phi", "r"), slow),
        },
        coords={"time": time, "phi": phi, "r": radius},
        attrs=manifest,
    )


def load_inputs(archive_root, cr, prepared=False):
    """Read the source rows or the empirical-speed rows used for one CR."""
    filename = "prepared_inputs.parquet" if prepared else "inputs.parquet"
    return pd.read_parquet(_cr_dir(archive_root, cr) / filename)


def load_cube(archive_root, start, end, context_frames=0):
    """Read only intersecting HDF5 time chunks and optional prior frame."""
    start = _utc_naive(start)
    end = _utc_naive(end)
    assert start < end and context_frames in (0, 1)
    rotations = list(iter_crs(start, end))
    if context_frames:
        with h5py.File(_cr_dir(archive_root, rotations[0]) / "cube.h5", "r") as first_cube:
            first_time = pd.Timestamp(first_cube["time_ns"][0])
        if start <= first_time:
            rotations.insert(0, rotations[0] - 1)
    cubes = []
    for index, cr in enumerate(rotations):
        directory = _cr_dir(archive_root, cr)
        with h5py.File(directory / "cube.h5", "r") as cube:
            times = pd.to_datetime(cube["time_ns"][:], utc=True).tz_localize(None)
            first = int(times.searchsorted(start))
            last = int(times.searchsorted(end))
            if index == 0:
                first = max(0, first - context_frames)
            if last <= first:
                continue
            part = xr.Dataset(
                {
                    "speed": (("time", "phi", "r"), cube["speed"][first:last]),
                    "is_slow_wind": (("time", "phi", "r"), cube["is_slow_wind"][first:last]),
                },
                coords={
                    "time": times[first:last],
                    "phi": cube["phi"][:],
                    "r": cube["r"][:],
                },
            )
            cubes.append(part)
    assert cubes, "No archived time centres in the requested range"
    for cube in cubes[1:]:
        assert np.array_equal(cubes[0].phi.values, cube.phi.values)
        assert np.array_equal(cubes[0].r.values, cube.r.values)
    combined = xr.concat(cubes, dim="time")
    assert pd.DatetimeIndex(combined.time.values).is_unique
    return combined


def load_series(archive_root, start, end):
    """Concatenate rounded-core series without duplicate boundary timestamps."""
    start = _utc_naive(start)
    end = _utc_naive(end)
    assert start < end
    rotations = list(iter_crs(start, end))
    frames = [pd.read_parquet(_cr_dir(archive_root, cr) / "series.parquet") for cr in rotations]
    combined = pd.concat(frames).sort_index()
    assert combined.index.is_unique
    first = int(combined.index.searchsorted(start))
    last = int(combined.index.searchsorted(end))
    return combined.iloc[first:last]
