#!/usr/bin/env python3
import argparse
import sys
from pathlib import Path

import multiprocessing as mp
from functools import partial
from types import SimpleNamespace
import pandas as pd
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT_DIR))

import numpy as np
import psycopg
import sunpy.map
import userpwd
from scipy import ndimage
from skimage.draw import polygon
from sunpy.map.maputils import all_coordinates_from_map, coordinate_is_on_solar_disk

from Library.CH import project
from Library.Config import apply_config, paths
from Library.IO import pmap_path, prepare_pmap
from Library.Metrics import generate_omask
from Library.Processing import get_postprocessing_params, pmap_to_mask
from Library.IO import prepare_mask
from Models.CH_SW_Correspondence.Shugay import load as load_ch_sw_model


def chunk_rows(rows, chunk_size):
    if chunk_size <= 0:
        return [rows]
    return [rows[i : i + chunk_size] for i in range(0, len(rows), chunk_size)]


def idl_polygon_mask(x, y, shape):
    rows, columns = polygon(y, x, shape=shape)
    mask = np.zeros(shape, dtype=bool)
    mask[rows, columns] = True
    return mask


def idl_region_mask(m, shape):
    """Reproduce the fixed ±20° longitude, ±40° latitude IDL SW region."""
    height, width = shape
    center_x = float(m.meta["CRPIX1"])
    center_y = float(m.meta["CRPIX2"])
    solar_radius = m.rsun_obs.to_value("arcsec") / abs(float(m.meta["CDELT1"]))
    b0 = m.observer_coordinate.heliographic_stonyhurst.lat.to_value("rad")

    latitude_limit = np.deg2rad(40.0)
    longitude_limit = np.deg2rad(20.0)
    region_radius_x = solar_radius * np.sin(longitude_limit)
    y_up = center_y + solar_radius * (
        np.cos(b0) * np.sin(latitude_limit)
        - np.sin(b0) * np.cos(latitude_limit)
    )
    y_down = center_y + solar_radius * (
        np.cos(b0) * np.sin(-latitude_limit)
        - np.sin(b0) * np.cos(-latitude_limit)
    )

    angle = np.deg2rad(np.arange(361, dtype=float))
    y = np.empty_like(angle)
    y[:180] = np.sin(angle[:180]) * (y_up - center_y) + center_y
    y[180:] = np.sin(angle[180:]) * (center_y - y_down) + center_y
    x = np.cos(angle) * region_radius_x + center_x
    return idl_polygon_mask(x, y, (height, width)), solar_radius, center_x, center_y, b0


def idl_xy_to_lonlat(x, y, center_x, center_y, solar_radius, b0):
    dx = float(x) - center_x
    dy = float(y) - center_y
    radius = min(np.hypot(dx, dy), solar_radius)
    if radius == 0:
        return np.rad2deg(b0), 0.0

    rho = np.arcsin(radius / solar_radius)
    latitude = np.arcsin(
        np.cos(rho) * np.sin(b0)
        + np.sin(rho) * np.cos(b0) * dy / radius
    )
    longitude = np.arcsin(np.sin(rho) * dx / (radius * np.cos(latitude)))
    return np.rad2deg(latitude), np.rad2deg(longitude)


def idl_contour_corrected_area(mask, solar_radius, center_x, center_y, b0, min_area):
    """Port IDL's outer-contour correction used for ch_relative_correct_sphere_area."""
    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=int))
    corrected_area = 0.0

    for label in range(1, count + 1):
        component = labels == label
        mask_area = int(component.sum())
        if mask_area < min_area:
            continue

        y_coords, x_coords = np.nonzero(component)
        max_x_index = np.argmax(x_coords)
        min_x_index = np.argmin(x_coords)
        max_y_index = np.argmax(y_coords)
        min_y_index = np.argmin(y_coords)
        max_x, max_xy = x_coords[max_x_index], y_coords[max_x_index]
        min_x, min_xy = x_coords[min_x_index], y_coords[min_x_index]
        max_yx, max_y = x_coords[max_y_index], y_coords[max_y_index]
        min_yx, min_y = x_coords[min_y_index], y_coords[min_y_index]

        max_lat_1, max_lon = idl_xy_to_lonlat(
            max_x, max_xy, center_x, center_y, solar_radius, b0
        )
        min_lat_1, min_lon = idl_xy_to_lonlat(
            min_x, min_xy, center_x, center_y, solar_radius, b0
        )
        max_lat, max_lon_1 = idl_xy_to_lonlat(
            max_yx, max_y, center_x, center_y, solar_radius, b0
        )
        min_lat, min_lon_1 = idl_xy_to_lonlat(
            min_yx, min_y, center_x, center_y, solar_radius, b0
        )

        def spherical_distance(lat_a, lon_a, lat_b, lon_b):
            cosine = (
                np.sin(np.deg2rad(lat_a)) * np.sin(np.deg2rad(lat_b))
                + np.cos(np.deg2rad(lat_a))
                * np.cos(np.deg2rad(lat_b))
                * np.cos(np.deg2rad(lon_a - lon_b))
            )
            return solar_radius * np.arccos(np.clip(cosine, -1.0, 1.0))

        along_lat = spherical_distance(max_lat_1, max_lon, min_lat_1, min_lon)
        along_lon = spherical_distance(max_lat, max_lon_1, min_lat, min_lon_1)
        screen_lat = np.hypot(max_x - min_x, max_xy - min_xy)
        screen_lon = np.hypot(max_y - min_y, max_yx - min_yx)
        if screen_lat == 0 or screen_lon == 0:
            continue
        corrected_area += mask_area * along_lat / screen_lat * along_lon / screen_lon

    return corrected_area


def idl_relative_corrected_sphere_area(m, mask):
    """Return the IDL-compatible corrected CH area in database percent units."""
    region_mask, solar_radius, center_x, center_y, b0 = idl_region_mask(m, mask.shape)
    fits_native_mask = np.flipud(mask.astype(bool))
    angle = np.deg2rad(np.arange(361, dtype=float))
    limb_mask = idl_polygon_mask(
        np.cos(angle) * solar_radius + center_x,
        np.sin(angle) * solar_radius + center_y,
        mask.shape,
    )
    min_area = round(limb_mask.sum() * 0.004)
    corrected_area = idl_contour_corrected_area(
        fits_native_mask & region_mask,
        solar_radius,
        center_x,
        center_y,
        b0,
        min_area,
    )
    return 100.0 * corrected_area / (2.0 * np.pi * solar_radius**2)


def validate_against_db(out_df, start, end):
    """Report exact-observation-time error in the solar-wind model's units."""
    assert pd.Timestamp(end) < pd.Timestamp("2019-01-01"), (
        "This validator targets the frozen pre-2019 sdo CH series."
    )
    predicted = out_df.reset_index()
    sql = """
        SELECT dt, source_image, ch_relative_correct_sphere_area
        FROM sdo.sdo_ch_sw_0193
        WHERE dt >= %(start)s AND dt < %(end)s
    """
    with psycopg.connect(
        host="213.131.1.41",
        user="selector",
        dbname="smdc",
        password=userpwd.userpwd_postgre,
    ) as conn:
        with conn.cursor() as cur:
            cur.execute(
                sql,
                {
                    "start": pd.Timestamp(start),
                    "end": pd.Timestamp(end) + pd.Timedelta(days=1),
                },
            )
            expected = pd.DataFrame(
                cur.fetchall(),
                columns=["dt", "db_source_image", "db_relative_area_pct"],
            )

    predicted["source_dt"] = pd.to_datetime(predicted["source_dt"])
    expected["dt"] = pd.to_datetime(expected["dt"])
    compared = predicted.merge(
        expected,
        left_on="source_dt",
        right_on="dt",
        how="inner",
    )
    assert not compared.empty, (
        "No exact observation timestamps overlap the calculated masks and database. "
        "Hourly-bin comparisons are not valid for the interpolated SW profile."
    )

    model = load_ch_sw_model()
    compared["calculated_sw_speed"] = model.v_from_area(
        compared["s_idl_exact_pct"], compared["source_dt"]
    )
    compared["db_sw_speed"] = model.v_from_area(
        compared["db_relative_area_pct"], compared["source_dt"]
    )
    speed_error = compared["calculated_sw_speed"] - compared["db_sw_speed"]
    within_tolerance = speed_error.abs() <= 2.5
    print(
        "IDL exact-timestamp compatibility validation: "
        f"{len(compared)}/{len(predicted)} calculated frames matched; "
        f"{within_tolerance.mean():.1%} within ±2.5 km/s; "
        f"MAE={speed_error.abs().mean():.3f} km/s; "
        f"P95={speed_error.abs().quantile(0.95):.3f} km/s; "
        f"max={speed_error.abs().max():.3f} km/s"
    )
    return bool(within_tolerance.all())


def worker_compute(rows, specs, smoothing_params, area_mode):
    results = []

    for row in rows:
        row_ns = SimpleNamespace(fits_path=row[1], mask_path=row[2])
        m = sunpy.map.Map(row[1])
        record = {
            "key": row[0],
            "fits_path": row[1],
            "mask_path": row[2],
            "source_dt": pd.Timestamp(m.date.datetime).floor("s"),
        }

        idl_mask = prepare_mask(row[2])

        if area_mode == "idl-exact":
            record["s_idl_exact_pct"] = idl_relative_corrected_sphere_area(m, idl_mask)
            results.append(record)
            continue

        omask = generate_omask(row_ns)
        inv_mu = project(m, np.ones_like(omask, dtype=float))

        hpc_coords = all_coordinates_from_map(m)
        disk_mask = coordinate_is_on_solar_disk(hpc_coords)
        sun = project(m, disk_mask).sum()

        def rel_area_from_mask(mask_2d):
            ch_mask = mask_2d * omask
            proj = np.zeros_like(ch_mask, dtype=float)
            good = (ch_mask != 0) & (inv_mu > 1e-3)
            proj[good] = ch_mask[good] * inv_mu[good]
            ch_area = np.nan_to_num(proj, 0).sum()
            return ch_area / sun

        record["s_idl"] = rel_area_from_mask(idl_mask)

        for arch, date_id in specs:
            spec = f"{arch}{date_id}"
            row_ns = SimpleNamespace(fits_path=row[1], mask_path=row[2])
            pmap_file = pmap_path(row_ns, arch, date_id)
            if not Path(pmap_file).exists():
                record[f"s_{spec.lower()}"] = float("nan")
                continue
            pmap = prepare_pmap(pmap_file)
            model_mask = pmap_to_mask(pmap, smoothing_params)
            record[f"s_{spec.lower()}"] = rel_area_from_mask(model_mask)

        results.append(record)

    return results


def main(argv):
    parser = argparse.ArgumentParser(
        description="Calculate CH areas from masks.",
    )
    parser.add_argument("start", help="inclusive YYYYMMDD date")
    parser.add_argument("end", help="inclusive YYYYMMDD date")
    parser.add_argument(
        "--area-mode",
        choices=("projected", "idl-exact"),
        default="projected",
        help="IDL-compatible mode writes s_idl_exact_pct in database percent units.",
    )
    parser.add_argument(
        "--validate-db",
        action="store_true",
        help="Report exact-mode error against sdo.sdo_ch_sw_0193 in km/s.",
    )
    parser.add_argument(
        "--paths-parquet",
        type=Path,
        default=Path(paths["artifact_root"]) / "Paths.parquet",
        help="Dataset index; filamentless inference writes a compatible Paths parquet.",
    )
    args = parser.parse_args(argv[1:])
    start = args.start
    end = args.end

    paths_parquet = args.paths_parquet
    if not paths_parquet.exists():
        print(f"Missing {paths_parquet}")
        return 1

    df = pd.read_parquet(paths_parquet)
    start_key = f"{start}_0000" if len(start) == 8 else start
    end_key = f"{end}_9999" if len(end) == 8 else end
    df = df.loc[start_key:end_key]
    if df.empty:
        print("No rows in requested date range.")
        return 1

    specs = [("A1", "D1"), ("A2", "D1"), ("A2", "D2")]
    smoothing_params = get_postprocessing_params("P0")

    rows = list(df[["fits_path", "mask_path"]].itertuples(index=True, name=None))
    max_workers = max(1, int(apply_config.get("plot_threads", 1)))
    chunk_size = int(apply_config.get("apply_batch_size", 16))
    chunks = chunk_rows(rows, chunk_size)

    results = []
    ctx = mp.get_context("spawn")
    worker_fn = partial(
        worker_compute,
        specs=specs,
        smoothing_params=smoothing_params,
        area_mode=args.area_mode,
    )
    with ctx.Pool(processes=min(max_workers, len(chunks))) as pool:
        iterator = pool.imap_unordered(worker_fn, chunks)
        for chunk_result in tqdm(iterator, total=len(chunks), desc="CH areas"):
            results.extend(chunk_result)

    out_df = pd.DataFrame(results).set_index("key")
    if args.area_mode == "idl-exact":
        out_df["dt"] = pd.to_datetime(out_df["source_dt"])
        out_df["ch_relative_area"] = out_df["s_idl_exact_pct"]
        ch_sw_model = load_ch_sw_model()
        out_df["forecast_sw_speed"] = ch_sw_model.v_from_area(
            out_df["ch_relative_area"],
            out_df["dt"],
        )

    out_dir = Path("./Outputs")
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.area_mode == "projected" else f" {args.area_mode}"
    out_path = out_dir / f"CH Areas {start}-{end}{suffix}.parquet"
    out_df.to_parquet(out_path)
    print(f"Saved {out_path}")
    if args.validate_db:
        assert args.area_mode == "idl-exact", "--validate-db requires --area-mode idl-exact"
        assert validate_against_db(out_df, start, end), (
            "Exact-timestamp SW-speed validation exceeded ±2.5 km/s."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
