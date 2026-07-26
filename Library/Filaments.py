from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

import astropy.units as u
import numpy as np
import pandas as pd
from astropy.coordinates import SkyCoord
from reproject import reproject_interp
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage.draw import line
from skimage.morphology import skeletonize
import sunpy.map
from sunpy.coordinates import HeliographicCarrington, frames
from sunpy.map.maputils import all_coordinates_from_map, coordinate_is_on_solar_disk
from tqdm import tqdm

from Library.IO import prepare_fits, prepare_mask


FEATURE_COLUMNS = [
    "log_area",
    "elongation",
    "centerline_length_px",
    "estimated_width_px",
    "aia193_mean",
    "aia193_local_contrast",
    "aia304_mean",
    "aia304_local_contrast",
    "aia304_dark_skew",
    "hmi_los_mean",
    "hmi_abs_mean",
    "hmi_strong_fraction",
    "hmi_abs_mean_strong",
    "hmi_abs_skew_strong",
    "hmi_flux_imbalance_strong",
]
HMI_SMOOTH_SIGMA_PX = 1.0
HMI_STRONG_FIELD_G = 12.0
CATALOG_DISTANCE_QUANTILE = 0.90
CATALOG_SUPPORT_RADIUS_PX = 10.0
CATALOG_ALIGNMENT_TOLERANCE_DEG = 30.0
CATALOG_TANGENT_WINDOW_PX = 6.0
CATALOG_MIN_ALIGNED_CENTERLINE_PX = 5
CATALOG_MIN_ALIGNED_CENTERLINE_FRACTION = 0.15
MAGFILO_SUPPORT_RADIUS_PX = 8.0
MAGFILO_POLYGON_TOLERANCE_PX = 2.0
MAGFILO_ALIGNMENT_TOLERANCE_DEG = 30.0
MAGFILO_MIN_ALIGNED_CENTERLINE_PX = 5
MAGFILO_MIN_ALIGNED_CENTERLINE_FRACTION = 0.15


def load_kislovodsk_catalog(path):
    catalog = pd.read_csv(Path(path), parse_dates=["datetime"])
    required = {"datetime", "lat1", "lon1", "lat2", "lon2"}
    missing = required.difference(catalog.columns)
    assert not missing, f"Missing Kislovodsk columns: {sorted(missing)}"
    return catalog.sort_values("datetime").reset_index(drop=True)


def select_catalog_segments(catalog, observation_dt, window_hours):
    delta = pd.Timedelta(hours=window_hours)
    candidates = catalog[
        (catalog["datetime"] >= observation_dt - delta)
        & (catalog["datetime"] <= observation_dt + delta)
    ]
    if candidates.empty:
        return candidates
    catalog_times = candidates["datetime"].drop_duplicates()
    nearest_time = catalog_times.iloc[
        np.argmin(np.abs(catalog_times - observation_dt).to_numpy())
    ]
    return candidates[candidates["datetime"] == nearest_time]


def project_catalog_segments(aia_map, filaments):
    """Project catalog segments into the prepared-array pixel convention.

    The feature and display arrays are vertically flipped from FITS storage so
    that FITS pixel index (0, 0) is at the bottom left.  Returned endpoints are
    therefore ``(column, row)`` in that display convention.
    """
    height, width = aia_map.data.shape
    frame = HeliographicCarrington(
        observer=aia_map.observer_coordinate,
        obstime=aia_map.date,
    )

    projected = []
    for filament in filaments.itertuples(index=False):
        start = SkyCoord(
            lon=filament.lon1 * u.deg,
            lat=filament.lat1 * u.deg,
            frame=frame,
        )
        end = SkyCoord(
            lon=filament.lon2 * u.deg,
            lat=filament.lat2 * u.deg,
            frame=frame,
        )
        start_pixel = aia_map.world_to_pixel(start)
        end_pixel = aia_map.world_to_pixel(end)
        pixel_values = np.array(
            [
                start_pixel.x.value,
                start_pixel.y.value,
                end_pixel.x.value,
                end_pixel.y.value,
            ]
        )
        if not np.isfinite(pixel_values).all():
            continue

        x1, y1, x2, y2 = np.rint(pixel_values).astype(int)
        y1 = height - 1 - y1
        y2 = height - 1 - y2
        if not (
            0 <= x1 < width
            and 0 <= y1 < height
            and 0 <= x2 < width
            and 0 <= y2 < height
        ):
            continue

        projected.append((x1, y1, x2, y2))

    return np.asarray(projected, dtype=np.float32).reshape(-1, 4)


def rasterize_catalog_segments(aia_map, filaments, return_projected_segments=False):
    """Rasterize catalog centrelines, optionally retaining their endpoints."""
    height, width = aia_map.data.shape
    filament_mask = np.zeros((height, width), dtype=bool)
    projected_segments = project_catalog_segments(aia_map, filaments)

    for x1, y1, x2, y2 in projected_segments.astype(int):
        rows, columns = line(y1, x1, y2, x2)
        filament_mask[rows, columns] = True

    if return_projected_segments:
        return filament_mask, len(projected_segments), projected_segments
    return filament_mask, len(projected_segments)


def compute_elongation(binary_mask):
    rows, columns = np.where(binary_mask)
    if len(rows) < 2:
        return 0.0

    rows = rows - rows.mean()
    columns = columns - columns.mean()
    covariance = np.array(
        [
            [np.mean(columns * columns), np.mean(columns * rows)],
            [np.mean(columns * rows), np.mean(rows * rows)],
        ]
    )
    eigenvalues = np.linalg.eigvalsh(covariance)
    major = np.sqrt(max(eigenvalues[-1], 0.0))
    minor = np.sqrt(max(eigenvalues[0], 0.0))
    if major == 0:
        return 0.0
    return float(1.0 - minor / major)


def compute_hmi_input(aia_map, hmi_path):
    hmi_map = sunpy.map.Map(hmi_path)
    hmi_reprojected, _ = reproject_interp(
        hmi_map,
        aia_map.wcs,
        shape_out=aia_map.data.shape,
    )
    hmi = np.flipud(hmi_reprojected.astype(np.float32))
    hmi = np.nan_to_num(hmi, nan=0.0)

    coordinates = all_coordinates_from_map(aia_map)
    helioprojective = coordinates.transform_to(
        frames.Helioprojective(observer=aia_map.observer_coordinate)
    )
    rho = np.sqrt(
        helioprojective.Tx.to_value(u.deg) ** 2
        + helioprojective.Ty.to_value(u.deg) ** 2
    )
    solar_radius = aia_map.rsun_obs.to_value(u.deg)
    theta = np.arcsin(np.clip(rho / solar_radius, 0.0, 1.0))
    mu = np.flipud(np.cos(theta).astype(np.float32))
    disk_mask = np.flipud(coordinate_is_on_solar_disk(coordinates))
    valid = disk_mask & (mu >= 0.5)

    valid_float = valid.astype(float)
    smoothed_numerator = ndimage.gaussian_filter(
        hmi * valid_float,
        sigma=HMI_SMOOTH_SIGMA_PX,
    )
    smoothed_denominator = ndimage.gaussian_filter(
        valid_float,
        sigma=HMI_SMOOTH_SIGMA_PX,
    )
    smoothed_los = np.zeros_like(hmi, dtype=np.float32)
    supported = smoothed_denominator > 0
    smoothed_los[supported] = (
        smoothed_numerator[supported] / smoothed_denominator[supported]
    )

    hmi_radial = np.zeros_like(hmi, dtype=np.float32)
    hmi_radial[valid] = smoothed_los[valid] / mu[valid]
    return smoothed_los, hmi_radial, valid


def compute_catalog_centerline_metrics(
    component,
    filament_distance,
    catalog_available,
    projected_segments=None,
    support_radius_px=CATALOG_SUPPORT_RADIUS_PX,
    alignment_tolerance_deg=CATALOG_ALIGNMENT_TOLERANCE_DEG,
    tangent_window_px=CATALOG_TANGENT_WINDOW_PX,
):
    """Measure how closely a component's centreline follows catalog segments.

    The Kislovodsk catalog supplies centreline segments rather than filament
    widths. Distances are therefore evaluated only on the skeleton of the
    candidate component. A skeleton point is aligned only when a nearby finite
    catalog segment has a similar local, undirected tangent. This allows many
    catalog segments to support one region and one catalog segment to support
    several fractured regions.
    """
    if not catalog_available:
        return {
            "catalog_centerline_distance_q90_px": np.nan,
            "catalog_supported_centerline_px": np.nan,
            "catalog_supported_centerline_fraction": np.nan,
            "catalog_aligned_centerline_px": np.nan,
            "catalog_aligned_centerline_fraction": np.nan,
            "catalog_nearby_orientation_q90_deg": np.nan,
            "catalog_centerline_px": np.nan,
        }

    assert support_radius_px > 0.0
    assert 0.0 < alignment_tolerance_deg <= 90.0
    assert tangent_window_px > 0.0
    centerline = skeletonize(component)
    distances = filament_distance[centerline]
    if not distances.size or not np.isfinite(distances).any():
        return {
            "catalog_centerline_distance_q90_px": np.inf,
            "catalog_supported_centerline_px": 0,
            "catalog_supported_centerline_fraction": 0.0,
            "catalog_aligned_centerline_px": 0,
            "catalog_aligned_centerline_fraction": 0.0,
            "catalog_nearby_orientation_q90_deg": np.inf,
            "catalog_centerline_px": int(centerline.sum()),
        }

    finite_distances = distances[np.isfinite(distances)]
    supported = finite_distances <= support_radius_px
    rows, columns = np.nonzero(centerline)
    skeleton_points = np.column_stack((columns, rows)).astype(np.float32)
    aligned = np.zeros(len(skeleton_points), dtype=bool)
    nearby_orientation_q90 = np.inf

    if projected_segments is not None and len(projected_segments):
        segment_starts = projected_segments[:, :2]
        segment_vectors = projected_segments[:, 2:] - segment_starts
        segment_length_squared = np.sum(segment_vectors**2, axis=1)
        valid_segments = segment_length_squared > 0.0
        segment_starts = segment_starts[valid_segments]
        segment_vectors = segment_vectors[valid_segments]
        segment_length_squared = segment_length_squared[valid_segments]

        if len(segment_starts):
            tangent_tree = cKDTree(skeleton_points)
            tangents = np.full_like(skeleton_points, np.nan)
            for point_index, neighbours in enumerate(
                tangent_tree.query_ball_point(skeleton_points, tangent_window_px)
            ):
                neighbourhood = skeleton_points[neighbours].astype(np.float64)
                neighbourhood = neighbourhood[np.isfinite(neighbourhood).all(axis=1)]
                if len(neighbourhood) < 3:
                    continue
                centered = neighbourhood - neighbourhood.mean(axis=0)
                covariance = centered.T @ centered / len(neighbourhood)
                eigenvalues, eigenvectors = np.linalg.eigh(covariance)
                if eigenvalues[-1] <= 0.0:
                    continue
                tangents[point_index] = eigenvectors[:, -1]

            tangent_valid = np.isfinite(tangents).all(axis=1)
            point_offsets = skeleton_points[:, None, :] - segment_starts[None, :, :]
            projected_fraction = np.sum(
                point_offsets * segment_vectors[None, :, :], axis=2
            ) / segment_length_squared[None, :]
            projected_fraction = np.clip(projected_fraction, 0.0, 1.0)
            closest_points = (
                segment_starts[None, :, :]
                + projected_fraction[..., None] * segment_vectors[None, :, :]
            )
            segment_distances = np.linalg.norm(
                skeleton_points[:, None, :] - closest_points,
                axis=2,
            )
            segment_directions = segment_vectors / np.sqrt(
                segment_length_squared)[:, None]
            dot_products = np.abs(tangents @ segment_directions.T)
            angle_degrees = np.degrees(
                np.arccos(np.clip(dot_products, 0.0, 1.0))
            )
            nearby = segment_distances <= support_radius_px
            nearby_angles = np.where(nearby & tangent_valid[:, None], angle_degrees, np.inf)
            minimum_nearby_angle = nearby_angles.min(axis=1)
            aligned = minimum_nearby_angle <= alignment_tolerance_deg
            finite_nearby_angles = minimum_nearby_angle[
                np.isfinite(minimum_nearby_angle)
            ]
            if finite_nearby_angles.size:
                nearby_orientation_q90 = float(
                    np.quantile(finite_nearby_angles, CATALOG_DISTANCE_QUANTILE)
                )

    return {
        "catalog_centerline_distance_q90_px": float(
            np.quantile(finite_distances, CATALOG_DISTANCE_QUANTILE)
        ),
        "catalog_supported_centerline_px": int(supported.sum()),
        "catalog_supported_centerline_fraction": float(supported.mean()),
        "catalog_aligned_centerline_px": int(aligned.sum()),
        "catalog_aligned_centerline_fraction": float(aligned.mean()),
        "catalog_nearby_orientation_q90_deg": nearby_orientation_q90,
        "catalog_centerline_px": int(centerline.sum()),
    }


def compute_magfilo_centerline_metrics(
    component,
    spine_distance,
    spine_mask,
    polygon_mask,
    projected_spine_segments,
    catalog_available,
    support_radius_px=MAGFILO_SUPPORT_RADIUS_PX,
    polygon_tolerance_px=MAGFILO_POLYGON_TOLERANCE_PX,
    alignment_tolerance_deg=MAGFILO_ALIGNMENT_TOLERANCE_DEG,
):
    """Score one dec1 region against the MAGFiLO spine and segmentation union.

    MAGFiLO gives a manually traced spine and a narrow segmentation polygon.
    The spine drives distance and tangent agreement; the polygon is retained as
    an independent morphology check.  The reverse spine coverage is descriptive
    only, so one physical filament may still support several fractured regions.
    """
    if not catalog_available:
        return {
            "magfilo_centerline_distance_q90_px": np.nan,
            "magfilo_supported_centerline_px": np.nan,
            "magfilo_supported_centerline_fraction": np.nan,
            "magfilo_aligned_centerline_px": np.nan,
            "magfilo_aligned_centerline_fraction": np.nan,
            "magfilo_nearby_orientation_q90_deg": np.nan,
            "magfilo_region_centerline_px": np.nan,
            "magfilo_polygon_supported_centerline_fraction": np.nan,
            "magfilo_polygon_overlap_fraction": np.nan,
            "magfilo_spine_near_component_fraction": np.nan,
        }

    spine_metrics = compute_catalog_centerline_metrics(
        component,
        spine_distance,
        True,
        projected_segments=projected_spine_segments,
        support_radius_px=support_radius_px,
        alignment_tolerance_deg=alignment_tolerance_deg,
    )
    centerline = skeletonize(component)
    polygon_distance = ndimage.distance_transform_edt(~polygon_mask)
    polygon_supported = polygon_distance[centerline] <= polygon_tolerance_px
    spine_to_component = ndimage.distance_transform_edt(~component)
    spine_to_component = spine_to_component[spine_mask]
    polygon_area = int(polygon_mask.sum())

    return {
        "magfilo_centerline_distance_q90_px": spine_metrics[
            "catalog_centerline_distance_q90_px"
        ],
        "magfilo_supported_centerline_px": spine_metrics[
            "catalog_supported_centerline_px"
        ],
        "magfilo_supported_centerline_fraction": spine_metrics[
            "catalog_supported_centerline_fraction"
        ],
        "magfilo_aligned_centerline_px": spine_metrics[
            "catalog_aligned_centerline_px"
        ],
        "magfilo_aligned_centerline_fraction": spine_metrics[
            "catalog_aligned_centerline_fraction"
        ],
        "magfilo_nearby_orientation_q90_deg": spine_metrics[
            "catalog_nearby_orientation_q90_deg"
        ],
        "magfilo_region_centerline_px": spine_metrics["catalog_centerline_px"],
        "magfilo_polygon_supported_centerline_fraction": float(
            polygon_supported.mean()
        ) if polygon_supported.size else 0.0,
        "magfilo_polygon_overlap_fraction": (
            float((component & polygon_mask).sum() / polygon_area)
            if polygon_area
            else 0.0
        ),
        "magfilo_spine_near_component_fraction": float(
            (spine_to_component <= support_radius_px).mean()
        ) if spine_to_component.size else 0.0,
    }


def magfilo_is_filament(
    metrics,
    min_aligned_centerline_px=MAGFILO_MIN_ALIGNED_CENTERLINE_PX,
    min_aligned_centerline_fraction=MAGFILO_MIN_ALIGNED_CENTERLINE_FRACTION,
):
    return int(
        metrics["magfilo_aligned_centerline_px"]
        >= min_aligned_centerline_px
        and metrics["magfilo_aligned_centerline_fraction"]
        >= min_aligned_centerline_fraction
    )


def summarize_component(
    component,
    aia193,
    aia304,
    hmi_los,
    hmi_radial,
    hmi_valid,
    filament_distance,
    catalog_available,
    projected_segments,
    catalog_support_radius_px,
    catalog_alignment_tolerance_deg,
    catalog_min_aligned_centerline_px,
    catalog_min_aligned_centerline_fraction,
):
    area = int(component.sum())
    centerline_length = int(skeletonize(component).sum())
    local_annulus = ndimage.binary_dilation(component, iterations=8) & ~component
    values_193 = aia193[component]
    values_304 = aia304[component]
    local_193 = aia193[local_annulus]
    local_304 = aia304[local_annulus]
    valid_hmi = hmi_radial[component & hmi_valid]
    valid_hmi_los = hmi_los[component & hmi_valid]
    strong_hmi = hmi_radial[
        component
        & hmi_valid
        & (np.abs(hmi_los) >= HMI_STRONG_FIELD_G)
    ]

    brightness_193_mean = float(values_193.mean())
    brightness_mean = float(values_304.mean())
    brightness_inverted = 1.0 - values_304
    brightness_std = float(brightness_inverted.std())
    if values_304.size >= 3 and brightness_std > 0:
        brightness_centered = brightness_inverted - brightness_inverted.mean()
        brightness_dark_skew = float(
            np.mean(brightness_centered**3) / brightness_std**3
        )
    else:
        brightness_dark_skew = np.nan

    strong_mean = float(strong_hmi.mean()) if strong_hmi.size else np.nan
    strong_std = float(strong_hmi.std()) if strong_hmi.size else np.nan
    if strong_hmi.size >= 3 and strong_std > 0:
        strong_centered = strong_hmi - strong_mean
        strong_skew = float(np.mean(strong_centered**3) / strong_std**3)
    else:
        strong_skew = np.nan
    strong_absolute_flux = float(np.abs(strong_hmi).sum())
    strong_flux_imbalance = (
        float(np.abs(strong_hmi.sum()) / strong_absolute_flux)
        if strong_absolute_flux > 0
        else np.nan
    )

    distances = filament_distance[component]
    minimum_distance = (
        float(distances.min()) if catalog_available and distances.size else np.nan
    )
    centerline_metrics = compute_catalog_centerline_metrics(
        component,
        filament_distance,
        catalog_available,
        projected_segments=projected_segments,
        support_radius_px=catalog_support_radius_px,
        alignment_tolerance_deg=catalog_alignment_tolerance_deg,
    )
    is_filament = (
        int(
            centerline_metrics["catalog_aligned_centerline_px"]
            >= catalog_min_aligned_centerline_px
            and centerline_metrics["catalog_aligned_centerline_fraction"]
            >= catalog_min_aligned_centerline_fraction
        )
        if catalog_available
        else np.nan
    )
    return {
        "area_px": area,
        "log_area": float(np.log1p(area)),
        "elongation": compute_elongation(component),
        "centerline_length_px": centerline_length,
        "estimated_width_px": (
            float(area / centerline_length) if centerline_length else np.nan
        ),
        "aia193_mean": brightness_193_mean,
        "aia193_local_contrast": (
            float(brightness_193_mean - local_193.mean())
            if local_193.size
            else np.nan
        ),
        "aia304_mean": brightness_mean,
        "aia304_local_contrast": (
            float(brightness_mean - local_304.mean())
            if local_304.size
            else np.nan
        ),
        "aia304_dark_skew": brightness_dark_skew,
        "hmi_los_mean": (
            float(valid_hmi_los.mean()) if valid_hmi_los.size else np.nan
        ),
        "hmi_abs_mean": (
            float(np.abs(valid_hmi).mean()) if valid_hmi.size else np.nan
        ),
        "hmi_strong_fraction": (
            float(strong_hmi.size / valid_hmi.size) if valid_hmi.size else np.nan
        ),
        "hmi_abs_mean_strong": (
            float(np.abs(strong_hmi).mean()) if strong_hmi.size else np.nan
        ),
        "hmi_abs_skew_strong": abs(strong_skew),
        "hmi_flux_imbalance_strong": strong_flux_imbalance,
        "catalog_min_distance_px": minimum_distance,
        **centerline_metrics,
        "is_filament": is_filament,
    }


def build_filament_feature_frame(task):
    (
        frame_key,
        observation,
        filaments,
        catalog_support_radius_px,
        catalog_alignment_tolerance_deg,
        catalog_min_aligned_centerline_px,
        catalog_min_aligned_centerline_fraction,
    ) = task
    observation_dt = pd.to_datetime(frame_key, format="%Y%m%d_%H%M")
    candidate_mask = prepare_mask(observation.mask_path).astype(bool)
    labels, component_count = ndimage.label(
        candidate_mask,
        structure=np.ones((3, 3), dtype=int),
    )
    assert component_count > 0, f"Empty mask passed for {frame_key}"
    catalog_available = not filaments.empty

    assert pd.notna(observation.hmi_path), f"Missing HMI path for {frame_key}"
    assert pd.notna(observation.aia304_path), (
        f"Missing AIA 304 path for {frame_key}"
    )
    aia_map, aia193 = prepare_fits(observation.fits_path)
    _, aia304 = prepare_fits(observation.aia304_path)
    assert candidate_mask.shape == aia193.shape == aia304.shape == aia_map.data.shape

    hmi_los, hmi_radial, hmi_valid = compute_hmi_input(
        aia_map,
        observation.hmi_path,
    )
    if catalog_available:
        filament_mask, rasterized, projected_segments = rasterize_catalog_segments(
            aia_map,
            filaments,
            return_projected_segments=True,
        )
        if filament_mask.any():
            filament_distance = ndimage.distance_transform_edt(~filament_mask)
        else:
            filament_distance = np.full(candidate_mask.shape, np.inf)
    else:
        filament_distance = np.full(candidate_mask.shape, np.inf)
        rasterized = 0
        projected_segments = np.empty((0, 4), dtype=np.float32)

    rows = []
    for component_id in range(1, component_count + 1):
        component = labels == component_id
        summary = summarize_component(
            component,
            aia193,
            aia304,
            hmi_los,
            hmi_radial,
            hmi_valid,
            filament_distance,
            catalog_available,
            projected_segments,
            catalog_support_radius_px,
            catalog_alignment_tolerance_deg,
            catalog_min_aligned_centerline_px,
            catalog_min_aligned_centerline_fraction,
        )
        rows.append(
            {
                "frame_key": frame_key,
                "observation_dt": observation_dt,
                "component_id": component_id,
                "mask_path": observation.mask_path,
                "catalog_available": catalog_available,
                "catalog_datetime": (
                    filaments["datetime"].iloc[0] if catalog_available else pd.NaT
                ),
                "catalog_segments": len(filaments),
                "catalog_segments_rasterized": rasterized,
                **summary,
            }
        )
    return rows


def build_filament_feature_table(
    paths_df,
    catalog,
    catalog_window_hours=0.5,
    catalog_support_radius_px=CATALOG_SUPPORT_RADIUS_PX,
    catalog_alignment_tolerance_deg=CATALOG_ALIGNMENT_TOLERANCE_DEG,
    catalog_min_aligned_centerline_px=CATALOG_MIN_ALIGNED_CENTERLINE_PX,
    catalog_min_aligned_centerline_fraction=CATALOG_MIN_ALIGNED_CENTERLINE_FRACTION,
    training_only=False,
    label_frame_keys=None,
    workers=1,
):
    rows = []
    assert workers >= 1, "workers must be positive"
    required_paths = ["fits_path", "mask_path", "hmi_path", "aia304_path"]
    missing_columns = set(required_paths).difference(paths_df.columns)
    assert not missing_columns, (
        "Filament feature paths are missing columns: "
        f"{sorted(missing_columns)}"
    )
    skipped_empty = 0
    skipped_missing_inputs = 0
    skipped_unlabeled = 0
    eligible = []

    filter_progress = tqdm(
        paths_df.iterrows(),
        total=len(paths_df),
        desc="Filament frame filter",
    )
    for frame_key, observation in filter_progress:
        if any(pd.isna(getattr(observation, column)) for column in required_paths):
            skipped_missing_inputs += 1
            filter_progress.set_postfix(
                queued=len(eligible),
                empty=skipped_empty,
                missing_inputs=skipped_missing_inputs,
                unlabeled=skipped_unlabeled,
            )
            continue

        observation_dt = pd.to_datetime(frame_key, format="%Y%m%d_%H%M")
        candidate_mask = prepare_mask(observation.mask_path).astype(bool)
        labels, component_count = ndimage.label(
            candidate_mask,
            structure=np.ones((3, 3), dtype=int),
        )
        if component_count == 0:
            skipped_empty += 1
            filter_progress.set_postfix(
                queued=len(eligible),
                empty=skipped_empty,
                missing_inputs=skipped_missing_inputs,
                unlabeled=skipped_unlabeled,
            )
            continue

        filaments = select_catalog_segments(
            catalog,
            observation_dt,
            catalog_window_hours,
        )
        catalog_available = not filaments.empty
        label_available = (
            frame_key in label_frame_keys
            if label_frame_keys is not None
            else catalog_available
        )
        if training_only and not label_available:
            skipped_unlabeled += 1
            filter_progress.set_postfix(
                queued=len(eligible),
                empty=skipped_empty,
                missing_inputs=skipped_missing_inputs,
                unlabeled=skipped_unlabeled,
            )
            continue

        eligible.append(
            (
                frame_key,
                observation,
                filaments,
                catalog_support_radius_px,
                catalog_alignment_tolerance_deg,
                catalog_min_aligned_centerline_px,
                catalog_min_aligned_centerline_fraction,
            )
        )
        filter_progress.set_postfix(
            queued=len(eligible),
            empty=skipped_empty,
            missing_inputs=skipped_missing_inputs,
            unlabeled=skipped_unlabeled,
        )

    if not eligible:
        print(
            "Feature collection: "
            f"processed 0 frames; skipped {skipped_empty} empty masks and "
            f"{skipped_missing_inputs} frames with missing inputs and "
            f"{skipped_unlabeled} unlabeled frames."
        )
        return pd.DataFrame(rows)

    if workers == 1:
        feature_rows = map(build_filament_feature_frame, eligible)
        for frame_rows in tqdm(
            feature_rows,
            total=len(eligible),
            desc="Filament features",
        ):
            rows.extend(frame_rows)
    else:
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=context,
        ) as executor:
            feature_rows = executor.map(build_filament_feature_frame, eligible)
            for frame_rows in tqdm(
                feature_rows,
                total=len(eligible),
                desc=f"Filament features ({workers} workers)",
            ):
                rows.extend(frame_rows)

    print(
        "Feature collection: "
        f"processed {len(eligible)} frames; skipped {skipped_empty} empty masks and "
        f"{skipped_missing_inputs} frames with missing inputs and "
        f"{skipped_unlabeled} unlabeled frames."
    )

    return pd.DataFrame(rows)
