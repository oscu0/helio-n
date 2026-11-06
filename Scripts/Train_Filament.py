#!/usr/bin/env python3
import argparse
from concurrent.futures import ProcessPoolExecutor
import html
import json
import multiprocessing as mp
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import astropy.units as u
import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.optimize import minimize
from skimage.morphology import skeletonize
import sunpy.map
from matplotlib import pyplot as plt
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from Library.Config import filament_feature_workers, paths
from Library.Filament_Stats import (
    KISLOVODSK_RADII_ARCSEC,
    REFERENCE_ANGLE_DEG,
    REFERENCE_OVERLAP,
    SCORE_COLUMNS,
    collect_stats_frame,
    initialize_stats_worker,
    yearly_catalog_stats,
)
from Library.Filaments import (
    CATALOG_ALIGNMENT_TOLERANCE_DEG,
    CATALOG_DISTANCE_QUANTILE,
    CATALOG_MIN_SUPPORTED_CENTERLINE_PX,
    FEATURE_COLUMNS,
    MAGFILO_ALIGNMENT_TOLERANCE_DEG,
    MAGFILO_MIN_ALIGNED_CENTERLINE_FRACTION,
    MAGFILO_MIN_ALIGNED_CENTERLINE_PX,
    MAGFILO_POLYGON_TOLERANCE_PX,
    MAGFILO_SUPPORT_RADIUS_PX,
    boundary_contact_lengths,
    build_filament_feature_table,
    compute_catalog_centerline_metrics,
    compute_magfilo_centerline_metrics,
    kislovodsk_is_filament,
    load_kislovodsk_catalog,
    magfilo_is_filament,
    rasterize_catalog_segments,
    sample_filament_spines,
    select_catalog_segments,
)
from Library.GONG import (
    load_magfilo,
    magfilo_observations,
    project_magfilo_observation,
    projected_spine_segments,
    rasterize_projected_annotations,
    rasterize_projected_spines,
)
from Library.IO import prepare_fits, prepare_mask


OVERLAP_MAGFILO_CATALOG = None
OVERLAP_MAGFILO_FITS = None


def sigmoid(values):
    positive = values >= 0
    probabilities = np.empty_like(values, dtype=np.float64)
    probabilities[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    probabilities[~positive] = exponential / (1.0 + exponential)
    return probabilities


def logistic_loss_gradient(parameters, features, labels, weights, l2):
    coefficients = parameters[:-1]
    intercept = parameters[-1]
    linear = features @ coefficients + intercept
    losses = np.logaddexp(0.0, linear) - labels * linear
    normalization = weights.sum()
    loss = float(
        np.sum(weights * losses) / normalization
        + 0.5 * l2 * np.sum(coefficients**2)
    )
    residual = weights * (sigmoid(linear) - labels) / normalization
    gradient = np.concatenate(
        [
            features.T @ residual + l2 * coefficients,
            [residual.sum()],
        ]
    )
    return loss, gradient


def fit_logistic(features, labels, l2, max_iterations):
    negatives = int(np.sum(labels == 0))
    positives = int(np.sum(labels == 1))
    assert negatives > 0 and positives > 0, "Training requires both classes."
    class_weights = {
        0: len(labels) / (2.0 * negatives),
        1: len(labels) / (2.0 * positives),
    }
    weights = np.where(labels == 1, class_weights[1], class_weights[0])
    initial = np.zeros(features.shape[1] + 1, dtype=np.float64)
    result = minimize(
        logistic_loss_gradient,
        initial,
        args=(features, labels, weights, l2),
        method="L-BFGS-B",
        jac=True,
        options={"maxiter": max_iterations},
    )
    assert result.success, f"Logistic optimization failed: {result.message}"
    return result.x[:-1], float(result.x[-1]), class_weights, result


def split_by_day(features, validation_fraction):
    frame_days = (
        pd.to_datetime(features["observation_dt"]).dt.normalize().drop_duplicates()
    )
    assert len(frame_days) >= 2, "At least two catalog-covered days are required."
    validation_days = max(1, int(np.ceil(len(frame_days) * validation_fraction)))
    split_day = frame_days.iloc[-validation_days]
    validation_mask = (
        pd.to_datetime(features["observation_dt"]).dt.normalize() >= split_day
    )
    train = features.loc[~validation_mask].copy()
    validation = features.loc[validation_mask].copy()
    assert not train.empty and not validation.empty
    assert train["is_filament"].nunique() == 2, (
        "Training split must contain both filament classes."
    )
    assert validation["is_filament"].nunique() == 2, (
        "Validation split must contain both filament classes."
    )
    return train, validation


def prepare_feature_values(train, validation):
    medians = train[FEATURE_COLUMNS].median()
    assert medians.notna().all(), (
        "Features without a finite training median: "
        f"{medians.index[medians.isna()].tolist()}"
    )
    train_imputed = train[FEATURE_COLUMNS].fillna(medians)
    validation_imputed = validation[FEATURE_COLUMNS].fillna(medians)
    means = train_imputed.mean()
    scales = train_imputed.std(ddof=0)
    scales[scales == 0.0] = 1.0
    train_values = ((train_imputed - means) / scales).to_numpy(dtype=np.float64)
    validation_values = (
        (validation_imputed - means) / scales
    ).to_numpy(dtype=np.float64)
    return train_values, validation_values, medians, means, scales


def classification_metrics(labels, probabilities, threshold=0.5):
    predictions = probabilities >= threshold
    labels = labels.astype(bool)
    true_positive = int(np.sum(predictions & labels))
    true_negative = int(np.sum(~predictions & ~labels))
    false_positive = int(np.sum(predictions & ~labels))
    false_negative = int(np.sum(~predictions & labels))
    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive
        else 0.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative
        else 0.0
    )
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return {
        "threshold": threshold,
        "accuracy": float(np.mean(predictions == labels)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }


def select_precision_threshold(labels, probabilities, target_precision):
    candidates = np.unique(np.append(probabilities, 0.5))
    scored = [
        classification_metrics(labels, probabilities, threshold=float(threshold))
        for threshold in candidates
    ]
    eligible = [
        item
        for item in scored
        if item["precision"] >= target_precision and item["true_positive"] > 0
    ]
    if eligible:
        return max(
            eligible,
            key=lambda item: (
                item["recall"],
                item["precision"],
                item["f1"],
                item["threshold"],
            ),
        )
    return max(
        scored,
        key=lambda item: (
            item["precision"],
            item["recall"],
            item["f1"],
            item["threshold"],
        ),
    )


def merge_external_labels(features, labels_path):
    labels = pd.read_parquet(labels_path)
    required = {
        "frame_key",
        "component_id",
        "kislovodsk_available",
        "kislovodsk_is_filament",
        "magfilo_available",
        "magfilo_is_filament",
    }
    missing = required.difference(labels.columns)
    assert not missing, f"External label table is missing: {sorted(missing)}"
    label_columns = sorted(required)
    features = features.drop(
        columns=[column for column in label_columns[2:] if column in features],
    )
    return features.merge(
        labels[label_columns],
        on=["frame_key", "component_id"],
        how="left",
        validate="one_to_one",
    )


def external_label_frame_keys(labels_path):
    labels = pd.read_parquet(labels_path)
    required = {
        "frame_key",
        "kislovodsk_available",
        "magfilo_available",
    }
    missing = required.difference(labels.columns)
    assert not missing, f"External label table is missing: {sorted(missing)}"
    covered = (
        labels["kislovodsk_available"].fillna(False).astype(bool)
        | labels["magfilo_available"].fillna(False).astype(bool)
    )
    return set(labels.loc[covered, "frame_key"])


def matched_magfilo_frames(paths_df, observations, window_hours):
    assert window_hours > 0.0
    frame_times = pd.to_datetime(paths_df.index, format="%Y%m%d_%H%M")
    matches = {}
    unmatched = 0
    for observation in observations.itertuples(index=False):
        offsets = np.abs(frame_times - observation.observation_dt)
        frame_position = offsets.argmin()
        offset_hours = offsets[frame_position].total_seconds() / 3600.0
        if offset_hours > window_hours:
            unmatched += 1
            continue
        frame_key = paths_df.index[frame_position]
        matches.setdefault(frame_key, []).append(
            {
                "observation": observation,
                "time_offset_hours": offset_hours,
            }
        )
    return matches, unmatched


def magfilo_fits_by_name(fits_root):
    fits_paths = list(Path(fits_root).rglob("*.fits.fz"))
    paths_by_name = {path.name: path for path in fits_paths}
    assert len(paths_by_name) == len(fits_paths), (
        f"MAGFiLO FITS cache has duplicate filenames under {fits_root}"
    )
    return paths_by_name


def run_catalog_stats(args, output_dir):
    start, end = pd.to_datetime([args.start, args.end], format="%Y%m%d")
    assert start <= end, "The start date must precede or equal the end date."
    catalogs = ("kislovodsk", "magfilo") if args.stats_catalog == "both" else (args.stats_catalog,)
    destination = args.stats_output_dir or (
        output_dir / f"Catalog Stats {args.stats_catalog} {args.start}-{args.end}"
    )
    assert not destination.exists(), f"Choose a new output directory: {destination}"
    paths_df = pd.read_parquet(args.paths_parquet).sort_index()
    paths_df = paths_df.loc[f"{args.start}_0000":f"{args.end}_9999", ["fits_path", "mask_path"]]
    if args.max_frames is not None:
        paths_df = paths_df.iloc[:args.max_frames]
    assert len(paths_df) and paths_df.index.is_unique, "Expected unique frame keys in the interval."
    masks = paths_df["mask_path"].dropna().astype(str)
    assert masks.str.endswith("_CH_MASK_FINAL.png").all(), (
        "Catalog statistics require final IDL masks. Rebuild the dataset index with Scripts/Make.py Dataset."
    )
    frame_times = pd.Series(pd.to_datetime(paths_df.index, format="%Y%m%d_%H%M"), index=paths_df.index)
    mask_times = pd.to_datetime(masks.map(lambda path: Path(path).name[3:16]), format="%Y%m%d_%H%M")
    paths_df["mask_offset_minutes"] = (mask_times - frame_times).dt.total_seconds() / 60
    paths_df["input_status"] = "ready"
    for column, status in (("fits_path", "missing_aia"), ("mask_path", "missing_mask")):
        missing = ~paths_df[column].map(lambda path: pd.notna(path) and Path(path).is_file())
        paths_df.loc[missing & paths_df["input_status"].eq("ready"), "input_status"] = status
    paths_df.loc[
        paths_df["input_status"].eq("ready") & paths_df["mask_offset_minutes"].ne(0),
        "input_status",
    ] = "mask_frame_mismatch"

    k_catalog = pd.DataFrame(columns=["datetime", "lat1", "lon1", "lat2", "lon2"])
    if "kislovodsk" in catalogs:
        assert np.isfinite(args.catalog_window_hours) and args.catalog_window_hours > 0
        pad = pd.Timedelta(hours=args.catalog_window_hours)
        k_catalog = load_kislovodsk_catalog(args.catalog)
        k_catalog = k_catalog.loc[k_catalog["datetime"].between(start - pad, end + pd.Timedelta(days=1) + pad)]
    magfilo_matches = {}
    magfilo_fits = {}
    magfilo_path = None
    if "magfilo" in catalogs:
        assert np.isfinite(args.magfilo_window_hours) and args.magfilo_window_hours > 0
        magfilo_path = args.magfilo_catalog
        observations = magfilo_observations(load_magfilo(magfilo_path))
        eligible = paths_df.loc[paths_df["input_status"].eq("ready")]
        if not eligible.empty:
            pad = pd.Timedelta(hours=args.magfilo_window_hours)
            observations = observations.loc[
                observations["observation_dt"].between(start - pad, end + pd.Timedelta(days=1) + pad)
            ]
            magfilo_matches, _ = matched_magfilo_frames(eligible, observations, args.magfilo_window_hours)
            magfilo_fits = magfilo_fits_by_name(args.magfilo_fits_root)

    tasks = []
    for frame_key, row in paths_df.iterrows():
        frame_time = frame_times[frame_key]
        k_filaments = k_catalog.iloc[:0]
        if row.input_status == "ready" and not k_catalog.empty:
            k_filaments = select_catalog_segments(k_catalog, frame_time, args.catalog_window_hours)
        k_time = k_filaments["datetime"].iloc[0] if len(k_filaments) else pd.NaT
        frame = {
            "input_status": row.input_status,
            "mask_offset_minutes": row.mask_offset_minutes,
            "kislovodsk_status": "no_match" if "kislovodsk" in catalogs else "disabled",
            "kislovodsk_time_utc": k_time,
            "kislovodsk_offset_hours": (k_time - frame_time).total_seconds() / 3600,
            "kislovodsk_segments": len(k_filaments),
            "magfilo_status": "no_match" if "magfilo" in catalogs else "disabled",
            "magfilo_time_utc": pd.NaT,
            "magfilo_offset_hours": np.nan,
            "magfilo_url": None,
            "magfilo_fits_path": None,
            "magfilo_projected_annotations": None,
        }
        if len(k_filaments):
            frame["kislovodsk_status"] = "matched"
        magfilo = None
        if frame_key in magfilo_matches:
            match = min(magfilo_matches[frame_key], key=lambda m: m["time_offset_hours"])
            obs = match["observation"]
            fits_path = magfilo_fits.get(Path(obs.url).stem + ".fits.fz")
            frame.update(
                magfilo_status="matched" if fits_path is not None else "missing_gong_fits",
                magfilo_time_utc=obs.observation_dt,
                magfilo_offset_hours=(obs.observation_dt - frame_time).total_seconds() / 3600,
                magfilo_url=obs.url,
                magfilo_fits_path=str(fits_path) if fits_path is not None else None,
            )
            magfilo = {"image_ids": obs.image_ids, "fits_path": fits_path}
        tasks.append((frame_key, row.fits_path, row.mask_path, k_filaments, magfilo, frame))

    if args.workers == 1:
        initialize_stats_worker(magfilo_path)
        results = list(tqdm(map(collect_stats_frame, tasks), total=len(tasks), desc="Catalog statistics"))
    else:
        with ProcessPoolExecutor(
            max_workers=args.workers, mp_context=mp.get_context("spawn"),
            initializer=initialize_stats_worker, initargs=(magfilo_path,),
        ) as executor:
            results = list(tqdm(executor.map(collect_stats_frame, tasks), total=len(tasks), desc="Catalog statistics"))
    frames = pd.DataFrame([frame for frame, _ in results])
    scores = pd.DataFrame(
        [score for _, rows in results for score in rows], columns=SCORE_COLUMNS
    )
    annual = yearly_catalog_stats(frames, scores, range(start.year, end.year + 1), catalogs)

    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination.parent) as temporary:
        staging = Path(temporary) / "results"
        staging.mkdir()
        frames.to_parquet(staging / "frames.parquet", index=False)
        scores.to_parquet(staging / "component-scores.parquet", index=False)
        annual.to_csv(staging / "yearly-thresholds.csv", index=False)
        (staging / "parameters.json").write_text(json.dumps({
            "start": args.start,
            "end": args.end,
            "paths_parquet": str(args.paths_parquet.resolve()),
            "max_frames": args.max_frames,
            "catalogs": catalogs,
            "kislovodsk_catalog": str(args.catalog.resolve()) if "kislovodsk" in catalogs else None,
            "magfilo_catalog": str(magfilo_path.resolve()) if magfilo_path is not None else None,
            "magfilo_fits_root": str(args.magfilo_fits_root.resolve()) if magfilo_path is not None else None,
            "kislovodsk_window_hours": args.catalog_window_hours,
            "magfilo_window_hours": args.magfilo_window_hours,
            "kislovodsk_radii_arcsec": KISLOVODSK_RADII_ARCSEC,
            "reference_overlap_fraction": REFERENCE_OVERLAP,
            "reference_angle_q80_deg": REFERENCE_ANGLE_DEG,
            "matching_time": "UTC frame key at minute resolution",
            "mask_matching": "require the mask filename to match the AIA frame key to the minute",
            "magfilo_matching": "nearest AIA frame per GONG exposure, then nearest exposure per AIA frame",
            "count_unit": "one connected component per image per catalog and threshold",
            "kislovodsk_overlap": "fraction of CH skeleton length near finite segments from the matched exposure",
            "magfilo_overlap": "fraction of CH component area inside a single annotation polygon",
            "orientation": "arc-length-weighted 80th percentile undirected local angle on supported skeleton",
        }, indent=2) + "\n")
        staging.rename(destination)
    print(f"Saved {len(frames)} frames, {len(scores)} score rows, and annual sweeps to {destination}")
    print("Frame inputs:", frames["input_status"].value_counts().to_dict())
    for catalog in catalogs:
        status = frames.loc[frames["input_status"].eq("ready"), f"{catalog}_status"]
        print(f"{catalog} coverage:", status.value_counts().to_dict())
    return 0


def initialize_overlap_worker(magfilo_catalog_path, magfilo_fits_root):
    global OVERLAP_MAGFILO_CATALOG, OVERLAP_MAGFILO_FITS
    OVERLAP_MAGFILO_CATALOG = load_magfilo(magfilo_catalog_path)
    OVERLAP_MAGFILO_FITS = magfilo_fits_by_name(magfilo_fits_root)


def build_catalog_overlap_frame(task):
    """Measure raw catalog/component contacts or aligned boundary runs.

    Kislovodsk has no widths, so a contact is any candidate skeleton pixel
    within the supplied support radius.  MAGFiLO supplies polygons, so direct
    and tolerance-dilated pixel intersections are both retained.  This is a
    candidate finder, not the orientation-aware training labeler.
    """
    (
        frame_key,
        observation,
        kislovodsk_filaments,
        magfilo_matches_for_frame,
        catalog_support_radius_px,
        magfilo_polygon_tolerance_px,
        boundary_contact,
    ) = task
    candidate_mask = prepare_mask(observation["mask_path"]).astype(bool)
    labels, component_count = ndimage.label(
        candidate_mask,
        structure=np.ones((3, 3), dtype=int),
    )
    kislovodsk_available = not kislovodsk_filaments.empty
    magfilo_available = bool(magfilo_matches_for_frame)
    summary = {
        "frame_key": frame_key,
        "components": component_count,
        "candidate_pixels": int(candidate_mask.sum()),
        "kislovodsk_available": kislovodsk_available,
        "magfilo_available": magfilo_available,
        "kislovodsk_segments": len(kislovodsk_filaments),
        "magfilo_observations": len(magfilo_matches_for_frame),
        "magfilo_annotations": int(
            sum(match["filament_annotations"] for match in magfilo_matches_for_frame)
        ),
    }
    if boundary_contact is not None:
        summary["kislovodsk_boundary_contact_components"] = 0
        summary["magfilo_boundary_contact_components"] = 0
    else:
        summary["kislovodsk_contact_components"] = 0
        summary["magfilo_direct_overlap_components"] = 0
        summary["magfilo_tolerance_overlap_components"] = 0
    summary["any_contact_components"] = 0
    if component_count == 0 or not (kislovodsk_available or magfilo_available):
        return [], summary

    # SunPy map construction supplies the AIA WCS without the normalization
    # and disk-statistic work performed by prepare_fits.
    aia_map = sunpy.map.Map(observation["fits_path"])
    assert candidate_mask.shape == aia_map.data.shape
    if kislovodsk_available:
        kislovodsk_mask, rasterized_segments, kislovodsk_segments = (
            rasterize_catalog_segments(
                aia_map, kislovodsk_filaments, return_projected_segments=True
            )
        )
        kislovodsk_distance = (
            ndimage.distance_transform_edt(~kislovodsk_mask)
            if kislovodsk_mask.any()
            else np.full(candidate_mask.shape, np.inf)
        )
    else:
        rasterized_segments = 0
        kislovodsk_segments = np.empty((0, 4))
        kislovodsk_distance = np.full(candidate_mask.shape, np.inf)
    summary["kislovodsk_segments_rasterized"] = rasterized_segments

    projected_magfilo = []
    if magfilo_available:
        for magfilo_match in magfilo_matches_for_frame:
            fits_name = Path(magfilo_match["url"]).stem + ".fits.fz"
            assert fits_name in OVERLAP_MAGFILO_FITS, (
                f"MAGFiLO FITS is not cached for {frame_key}: {fits_name}"
            )
            gong_map = sunpy.map.Map(OVERLAP_MAGFILO_FITS[fits_name])
            projected_magfilo.extend(
                project_magfilo_observation(
                    OVERLAP_MAGFILO_CATALOG,
                    {"image_ids": magfilo_match["image_ids"]},
                    gong_map,
                    aia_map,
                )
            )

    if boundary_contact is not None:
        distance_arcsec, angle_deg, min_run_arcsec = boundary_contact
        scale_arcsec = np.abs(u.Quantity(aia_map.scale).to_value(u.arcsec / u.pix))
        curves = {
            "kislovodsk": sample_filament_spines(
                kislovodsk_segments.reshape(-1, 2, 2), scale_arcsec
            ),
            "magfilo": sample_filament_spines(
                [annotation["spine"] for annotation in projected_magfilo], scale_arcsec
            ),
        }
        records = []
        for component_id in range(1, component_count + 1):
            longest = boundary_contact_lengths(
                labels == component_id, scale_arcsec, curves, distance_arcsec, angle_deg
            )
            record = {"frame_key": frame_key, "component_id": component_id}
            any_contact = False
            for catalog, length in longest.items():
                contact = length >= min_run_arcsec
                record[f"{catalog}_aligned_boundary_run_arcsec"] = length
                record[f"{catalog}_boundary_contact"] = contact if np.isfinite(length) else pd.NA
                summary[f"{catalog}_boundary_contact_components"] += int(contact)
                any_contact |= contact
            summary["any_contact_components"] += int(any_contact)
            records.append(record)
        return records, summary

    if magfilo_available:
        magfilo_polygon_mask = rasterize_projected_annotations(
            projected_magfilo,
            candidate_mask.shape,
        )
        magfilo_polygon_tolerance_mask = ndimage.binary_dilation(
            magfilo_polygon_mask,
            iterations=int(np.ceil(magfilo_polygon_tolerance_px)),
        )
    else:
        magfilo_polygon_mask = np.zeros(candidate_mask.shape, dtype=bool)
        magfilo_polygon_tolerance_mask = magfilo_polygon_mask
    summary["magfilo_polygon_pixels"] = int(magfilo_polygon_mask.sum())

    records = []
    for component_id in range(1, component_count + 1):
        component = labels == component_id
        centerline = skeletonize(component)
        centerline_distances = kislovodsk_distance[centerline]
        supported_centerline_px = int(
            np.sum(centerline_distances <= catalog_support_radius_px)
        )
        direct_polygon_overlap_px = int((component & magfilo_polygon_mask).sum())
        tolerance_polygon_overlap_px = int(
            (component & magfilo_polygon_tolerance_mask).sum()
        )
        kislovodsk_contact = supported_centerline_px > 0
        magfilo_direct_overlap = direct_polygon_overlap_px > 0
        magfilo_tolerance_overlap = tolerance_polygon_overlap_px > 0
        any_contact = kislovodsk_contact or magfilo_tolerance_overlap
        summary["kislovodsk_contact_components"] += int(kislovodsk_contact)
        summary["magfilo_direct_overlap_components"] += int(
            magfilo_direct_overlap
        )
        summary["magfilo_tolerance_overlap_components"] += int(
            magfilo_tolerance_overlap
        )
        summary["any_contact_components"] += int(any_contact)
        if not any_contact:
            continue
        records.append(
            {
                "frame_key": frame_key,
                "component_id": component_id,
                "component_pixels": int(component.sum()),
                "component_centerline_px": int(centerline.sum()),
                "kislovodsk_available": kislovodsk_available,
                "kislovodsk_supported_centerline_px": supported_centerline_px,
                "kislovodsk_centerline_distance_min_px": (
                    float(centerline_distances.min())
                    if centerline_distances.size
                    else np.inf
                ),
                "magfilo_available": magfilo_available,
                "magfilo_polygon_direct_overlap_px": direct_polygon_overlap_px,
                "magfilo_polygon_tolerance_overlap_px": (
                    tolerance_polygon_overlap_px
                ),
                "magfilo_polygon_overlap_fraction": (
                    float(direct_polygon_overlap_px / component.sum())
                ),
            }
        )
    return records, summary


def build_catalog_overlap_table(
    paths_df,
    kislovodsk_catalog,
    magfilo_matches,
    catalog_window_hours,
    catalog_support_radius_px,
    magfilo_polygon_tolerance_px,
    workers,
    magfilo_catalog_path,
    magfilo_fits_root,
    boundary_contact=None,
):
    assert workers >= 1
    tasks = []
    for frame_key, observation in paths_df.iterrows():
        kislovodsk_filaments = select_catalog_segments(
            kislovodsk_catalog,
            pd.to_datetime(frame_key, format="%Y%m%d_%H%M"),
            catalog_window_hours,
        )
        magfilo_matches_for_frame = [
            {
                "url": match["observation"].url,
                "image_ids": list(match["observation"].image_ids),
                "filament_annotations": int(
                    match["observation"].filament_annotations
                ),
            }
            for match in magfilo_matches.get(frame_key, [])
        ]
        if kislovodsk_filaments.empty and not magfilo_matches_for_frame:
            continue
        tasks.append(
            (
                frame_key,
                observation.to_dict(),
                kislovodsk_filaments,
                magfilo_matches_for_frame,
                catalog_support_radius_px,
                magfilo_polygon_tolerance_px,
                boundary_contact,
            )
        )

    records = []
    summaries = []
    initializer_args = (magfilo_catalog_path, magfilo_fits_root)
    if workers == 1:
        initialize_overlap_worker(*initializer_args)
        results = map(build_catalog_overlap_frame, tasks)
        for frame_records, summary in tqdm(
            results,
            total=len(tasks),
            desc="Catalog overlap scan",
        ):
            records.extend(frame_records)
            summaries.append(summary)
    else:
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=context,
            initializer=initialize_overlap_worker,
            initargs=initializer_args,
        ) as executor:
            results = executor.map(build_catalog_overlap_frame, tasks)
            for frame_records, summary in tqdm(
                results,
                total=len(tasks),
                desc=f"Catalog overlap scan ({workers} workers)",
            ):
                records.extend(frame_records)
                summaries.append(summary)
    overlaps = pd.DataFrame(records)
    if boundary_contact is not None:
        column_types = {
            "frame_key": "string",
            "component_id": "int64",
            "kislovodsk_aligned_boundary_run_arcsec": "float64",
            "kislovodsk_boundary_contact": "boolean",
            "magfilo_aligned_boundary_run_arcsec": "float64",
            "magfilo_boundary_contact": "boolean",
        }
        overlaps = overlaps.reindex(columns=list(column_types)).astype(column_types)
    return overlaps, pd.DataFrame(summaries), len(tasks)


def build_catalog_label_table(
    paths_df,
    kislovodsk_catalog,
    magfilo_catalog,
    magfilo_matches,
    magfilo_fits,
    catalog_window_hours,
    catalog_support_radius_px,
    catalog_alignment_tolerance_deg,
    catalog_min_supported_centerline_px,
    magfilo_support_radius_px,
    magfilo_polygon_tolerance_px,
    magfilo_alignment_tolerance_deg,
    magfilo_min_aligned_centerline_px,
    magfilo_min_aligned_centerline_fraction,
):
    label_frames = []
    for frame_key in paths_df.index:
        observation_dt = pd.to_datetime(frame_key, format="%Y%m%d_%H%M")
        kislovodsk_filaments = select_catalog_segments(
            kislovodsk_catalog,
            observation_dt,
            catalog_window_hours,
        )
        if not kislovodsk_filaments.empty or frame_key in magfilo_matches:
            label_frames.append((frame_key, kislovodsk_filaments))

    records = []
    for frame_key, kislovodsk_filaments in tqdm(
        label_frames,
        desc="Catalog labels",
    ):
        observation = paths_df.loc[frame_key]
        assert pd.notna(observation.fits_path), f"Missing AIA 193 path for {frame_key}"
        assert pd.notna(observation.mask_path), f"Missing mask path for {frame_key}"
        aia_map, _ = prepare_fits(observation.fits_path)
        candidate_mask = prepare_mask(observation.mask_path).astype(bool)
        assert candidate_mask.shape == aia_map.data.shape
        labels, component_count = ndimage.label(
            candidate_mask,
            structure=np.ones((3, 3), dtype=int),
        )
        if component_count == 0:
            continue

        kislovodsk_available = not kislovodsk_filaments.empty
        if kislovodsk_available:
            kislovodsk_mask, kislovodsk_rasterized, kislovodsk_segments = (
                rasterize_catalog_segments(
                    aia_map,
                    kislovodsk_filaments,
                    return_projected_segments=True,
                )
            )
            kislovodsk_distance = (
                ndimage.distance_transform_edt(~kislovodsk_mask)
                if kislovodsk_mask.any()
                else np.full(candidate_mask.shape, np.inf)
            )
        else:
            kislovodsk_rasterized = 0
            kislovodsk_segments = np.empty((0, 4), dtype=np.float32)
            kislovodsk_distance = np.full(candidate_mask.shape, np.inf)

        magfilo_matches_for_frame = magfilo_matches.get(frame_key, [])
        magfilo_available = bool(magfilo_matches_for_frame)
        if magfilo_available:
            projected_magfilo = []
            for magfilo_match in magfilo_matches_for_frame:
                magfilo_observation = magfilo_match["observation"]
                fits_name = Path(magfilo_observation.url).stem + ".fits.fz"
                assert fits_name in magfilo_fits, (
                    f"MAGFiLO FITS is not cached for {frame_key}: {fits_name}"
                )
                gong_map = sunpy.map.Map(magfilo_fits[fits_name])
                projected_magfilo.extend(
                    project_magfilo_observation(
                        magfilo_catalog,
                        {"image_ids": magfilo_observation.image_ids},
                        gong_map,
                        aia_map,
                    )
                )
            magfilo_polygon_mask = rasterize_projected_annotations(
                projected_magfilo,
                candidate_mask.shape,
            )
            magfilo_spine_mask = rasterize_projected_spines(
                projected_magfilo,
                candidate_mask.shape,
            )
            magfilo_segments = projected_spine_segments(projected_magfilo)
            magfilo_distance = (
                ndimage.distance_transform_edt(~magfilo_spine_mask)
                if magfilo_spine_mask.any()
                else np.full(candidate_mask.shape, np.inf)
            )
        else:
            magfilo_polygon_mask = np.zeros(candidate_mask.shape, dtype=bool)
            magfilo_spine_mask = np.zeros(candidate_mask.shape, dtype=bool)
            magfilo_segments = np.empty((0, 4), dtype=np.float32)
            magfilo_distance = np.full(candidate_mask.shape, np.inf)

        for component_id in range(1, component_count + 1):
            component = labels == component_id
            kislovodsk_metrics = compute_catalog_centerline_metrics(
                component,
                kislovodsk_distance,
                kislovodsk_available,
                projected_segments=kislovodsk_segments,
                support_radius_px=catalog_support_radius_px,
                alignment_tolerance_deg=catalog_alignment_tolerance_deg,
            )
            magfilo_metrics = compute_magfilo_centerline_metrics(
                component,
                magfilo_distance,
                magfilo_spine_mask,
                magfilo_polygon_mask,
                magfilo_segments,
                magfilo_available,
                support_radius_px=magfilo_support_radius_px,
                polygon_tolerance_px=magfilo_polygon_tolerance_px,
                alignment_tolerance_deg=magfilo_alignment_tolerance_deg,
            )
            records.append(
                {
                    "frame_key": frame_key,
                    "component_id": component_id,
                    "kislovodsk_available": kislovodsk_available,
                    "kislovodsk_is_filament": (
                        kislovodsk_is_filament(
                            kislovodsk_metrics,
                            catalog_min_supported_centerline_px,
                        )
                        if kislovodsk_available
                        else np.nan
                    ),
                    "magfilo_available": magfilo_available,
                    "magfilo_is_filament": (
                        magfilo_is_filament(
                            magfilo_metrics,
                            magfilo_min_aligned_centerline_px,
                            magfilo_min_aligned_centerline_fraction,
                        )
                        if magfilo_available
                        else np.nan
                    ),
                    "magfilo_observation_dt": (
                        min(
                            match["observation"].observation_dt
                            for match in magfilo_matches_for_frame
                        )
                        if magfilo_available
                        else pd.NaT
                    ),
                    "magfilo_time_offset_hours": (
                        min(
                            match["time_offset_hours"]
                            for match in magfilo_matches_for_frame
                        )
                        if magfilo_available
                        else np.nan
                    ),
                    "kislovodsk_segments": len(kislovodsk_filaments),
                    "kislovodsk_segments_rasterized": kislovodsk_rasterized,
                    **{
                        key.replace("catalog_", "kislovodsk_"): value
                        for key, value in kislovodsk_metrics.items()
                    },
                    **magfilo_metrics,
                }
            )
    return pd.DataFrame(records), len(label_frames)


def render_magfilo_review(
    paths_df,
    magfilo_catalog,
    magfilo_matches,
    magfilo_fits,
    output_dir,
):
    cases_dir = output_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)
    cards = []
    for frame_key in tqdm(
        sorted(magfilo_matches),
        desc="MAGFiLO review",
    ):
        observation = paths_df.loc[frame_key]
        aia_map, aia193 = prepare_fits(observation.fits_path)
        candidate_mask = prepare_mask(observation.mask_path).astype(bool)
        assert candidate_mask.shape == aia_map.data.shape

        projected_magfilo = []
        observation_times = []
        annotation_count = 0
        for magfilo_match in magfilo_matches[frame_key]:
            magfilo_observation = magfilo_match["observation"]
            fits_name = Path(magfilo_observation.url).stem + ".fits.fz"
            assert fits_name in magfilo_fits, (
                f"MAGFiLO FITS is not cached for {frame_key}: {fits_name}"
            )
            gong_map = sunpy.map.Map(magfilo_fits[fits_name])
            projected_magfilo.extend(
                project_magfilo_observation(
                    magfilo_catalog,
                    {"image_ids": magfilo_observation.image_ids},
                    gong_map,
                    aia_map,
                )
            )
            observation_times.append(magfilo_observation.observation_dt)
            annotation_count += magfilo_observation.filament_annotations

        figure, axes = plt.subplots(1, 2, figsize=(16, 8), layout="constrained")
        candidate_axis, overlay_axis = axes
        for axis in axes:
            axis.imshow(aia193, cmap="sdoaia193", origin="upper")
            axis.set_axis_off()
        candidate_axis.contour(
            candidate_mask,
            levels=[0.5],
            colors=["cyan"],
            linewidths=0.8,
        )
        candidate_axis.set_title("Dec1 candidate regions", fontsize=12)
        overlay_axis.contour(
            candidate_mask,
            levels=[0.5],
            colors=["cyan"],
            linewidths=0.8,
        )
        for annotation in projected_magfilo:
            for polygon in annotation["polygons"]:
                overlay_axis.fill(
                    polygon[:, 0],
                    polygon[:, 1],
                    facecolor="gold",
                    edgecolor="gold",
                    alpha=0.2,
                    linewidth=0.7,
                )
            overlay_axis.plot(
                annotation["spine"][:, 0],
                annotation["spine"][:, 1],
                color="white",
                linewidth=1.1,
            )
        overlay_axis.set_title(
            f"MAGFiLO: {annotation_count} annotations; cyan = candidates",
            fontsize=12,
        )
        figure.suptitle(
            f"{frame_key} | GONG "
            f"{', '.join(time.strftime('%Y-%m-%d %H:%M') for time in observation_times)}",
            fontsize=13,
        )
        image_name = f"{frame_key}.png"
        figure.savefig(cases_dir / image_name, dpi=150)
        plt.close(figure)
        cards.append(
            "<article>"
            f"<a href='cases/{image_name}'><img src='cases/{image_name}' loading='lazy'></a>"
            f"<h2>{html.escape(frame_key)}</h2>"
            f"<p>{annotation_count} MAGFiLO annotations; "
            f"{int(candidate_mask.sum())} candidate pixels.</p>"
            "</article>"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "index.html").write_text(
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>2017 MAGFiLO projection review</title>"
        "<style>body{font-family:system-ui;margin:24px;background:#181818;color:#eee}"
        "main{display:grid;grid-template-columns:repeat(auto-fit,minmax(500px,1fr));gap:18px}"
        "article{background:#272727;padding:10px;border-radius:8px}"
        "img{width:100%;height:auto}h1{margin-bottom:4px}h2{font-size:1rem;margin:8px 0 2px}"
        "p{font-size:.86rem;margin:0;color:#ccc}</style></head><body>"
        "<h1>2017 MAGFiLO projection review</h1>"
        "<p>Left: exact Dec1 candidate-region contours. Right: the same contours "
        "with projected MAGFiLO polygons (gold) and spines (white).</p>"
        f"<main>{''.join(cards)}</main></body></html>"
    )
    return len(cards)


def assign_training_labels(features):
    if "kislovodsk_available" not in features:
        features["kislovodsk_available"] = features["catalog_available"]
        features["kislovodsk_is_filament"] = features["is_filament"]
    if "magfilo_available" not in features:
        features["magfilo_available"] = False
        features["magfilo_is_filament"] = np.nan

    kislovodsk_available = features["kislovodsk_available"].fillna(False).astype(bool)
    magfilo_available = features["magfilo_available"].fillna(False).astype(bool)
    kislovodsk_positive = features["kislovodsk_is_filament"].fillna(0).astype(bool)
    magfilo_positive = features["magfilo_is_filament"].fillna(0).astype(bool)
    labeled = kislovodsk_available | magfilo_available
    positive = kislovodsk_positive | magfilo_positive
    features["is_filament"] = np.nan
    features.loc[labeled, "is_filament"] = positive[labeled].astype(int)
    features["label_source"] = "unlabeled"
    features.loc[labeled & ~positive, "label_source"] = "covered-negative"
    features.loc[kislovodsk_positive & ~magfilo_positive, "label_source"] = "K"
    features.loc[magfilo_positive & ~kislovodsk_positive, "label_source"] = "M"
    features.loc[kislovodsk_positive & magfilo_positive, "label_source"] = "K+M"
    return labeled


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Build features and train the region-level logistic filament model.",
    )
    parser.add_argument("start", help="inclusive YYYYMMDD")
    parser.add_argument("end", help="inclusive YYYYMMDD")
    parser.add_argument(
        "--mode",
        choices=("train", "labels", "review", "overlaps", "boundary-contacts", "catalog-stats"),
        default="train",
        help=(
            "Run model training, build labels, render the MAGFiLO review gallery, "
            "scan raw or aligned CH boundary contacts, or collect yearly catalog statistics."
        ),
    )
    parser.add_argument(
        "--paths-parquet",
        type=Path,
        default=Path(paths["artifact_root"]) / "Paths.parquet",
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=ROOT_DIR / "Data" / "Kislovodsk Filaments.csv",
    )
    parser.add_argument("--labels-parquet", type=Path)
    parser.add_argument("--output-labels-parquet", type=Path)
    parser.add_argument("--output-overlaps-parquet", type=Path)
    parser.add_argument("--review-output-dir", type=Path)
    parser.add_argument(
        "--magfilo-catalog",
        type=Path,
        default=ROOT_DIR / "Data" / "MAGFiLO" / "magfilo_2024_v1.0.json",
    )
    parser.add_argument(
        "--magfilo-fits-root",
        type=Path,
        default=ROOT_DIR / "Data" / "MAGFiLO",
    )
    parser.add_argument("--magfilo-window-hours", type=float, default=0.5)
    parser.add_argument(
        "--stats-catalog", choices=("both", "kislovodsk", "magfilo"), default="both",
        help="Catalog coverage to measure in catalog-stats mode.",
    )
    parser.add_argument("--stats-output-dir", type=Path, help="New directory for catalog-stats outputs.")
    parser.add_argument("--features-parquet", type=Path)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--catalog-window-hours", type=float, default=0.5)
    parser.add_argument("--catalog-support-radius-px", type=float, default=10.0)
    parser.add_argument(
        "--boundary-distance-arcsec", type=float, default=12.0,
        help="Maximum spine-to-boundary separation in boundary-contacts mode.",
    )
    parser.add_argument(
        "--boundary-angle-deg", type=float, default=30.0,
        help="Maximum undirected tangent difference in boundary-contacts mode.",
    )
    parser.add_argument(
        "--boundary-min-run-arcsec", type=float, default=36.0,
        help="Minimum continuous qualifying spine length in boundary-contacts mode.",
    )
    parser.add_argument(
        "--catalog-alignment-tolerance-deg",
        type=float,
        default=CATALOG_ALIGNMENT_TOLERANCE_DEG,
    )
    parser.add_argument(
        "--catalog-min-supported-centerline-px",
        type=int,
        default=CATALOG_MIN_SUPPORTED_CENTERLINE_PX,
    )
    parser.add_argument(
        "--magfilo-support-radius-px",
        type=float,
        default=MAGFILO_SUPPORT_RADIUS_PX,
    )
    parser.add_argument(
        "--magfilo-polygon-tolerance-px",
        type=float,
        default=MAGFILO_POLYGON_TOLERANCE_PX,
    )
    parser.add_argument(
        "--magfilo-alignment-tolerance-deg",
        type=float,
        default=MAGFILO_ALIGNMENT_TOLERANCE_DEG,
    )
    parser.add_argument(
        "--magfilo-min-aligned-centerline-px",
        type=int,
        default=MAGFILO_MIN_ALIGNED_CENTERLINE_PX,
    )
    parser.add_argument(
        "--magfilo-min-aligned-centerline-fraction",
        type=float,
        default=MAGFILO_MIN_ALIGNED_CENTERLINE_FRACTION,
    )
    parser.add_argument("--l2", type=float, default=1.0)
    parser.add_argument("--max-iterations", type=int, default=1000)
    parser.add_argument("--target-precision", type=float, default=0.9)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument(
        "--workers",
        type=int,
        default=filament_feature_workers,
        help="Feature-collection worker processes (from Machine.json by default).",
    )
    parser.add_argument(
        "--training-only",
        action="store_true",
        help=(
            "Skip empty masks and frames without a Kislovodsk or MAGFiLO label "
            "before loading AIA 304 or HMI. Do not use for inference features."
        ),
    )
    parser.add_argument("--reuse-features", action="store_true")
    parser.add_argument("--features-only", action="store_true")
    args = parser.parse_args(argv)
    assert 0.0 < args.validation_fraction < 1.0
    assert args.l2 >= 0.0
    assert 0.0 < args.target_precision <= 1.0
    assert args.workers >= 1, "workers must be positive"

    output_dir = ROOT_DIR / "Outputs" / "Filaments"
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.start}-{args.end}"
    features_path = args.features_parquet or output_dir / f"Features {stem}.parquet"
    model_path = args.model_path or output_dir / f"Classifier {stem}.json"

    if args.mode == "catalog-stats":
        return run_catalog_stats(args, output_dir)

    if args.mode in ("overlaps", "boundary-contacts"):
        assert args.magfilo_window_hours > 0.0
        boundary_contact = None
        if args.mode == "boundary-contacts":
            boundary_contact = (
                args.boundary_distance_arcsec,
                args.boundary_angle_deg,
                args.boundary_min_run_arcsec,
            )
            assert np.isfinite(boundary_contact).all() and min(boundary_contact) > 0
            assert args.boundary_angle_deg < 90
        paths_df = pd.read_parquet(args.paths_parquet)
        start_key = f"{args.start}_0000"
        end_key = f"{args.end}_9999"
        paths_df = paths_df.loc[start_key:end_key].copy()
        if args.max_frames is not None:
            paths_df = paths_df.iloc[: args.max_frames]
        assert not paths_df.empty, "No observations in the requested interval."
        assert paths_df[["fits_path", "mask_path"]].notna().all().all(), (
            "Catalog overlap scanning requires AIA 193 and mask paths for every frame."
        )
        magfilo_catalog = load_magfilo(args.magfilo_catalog)
        observations = magfilo_observations(magfilo_catalog)
        observations = observations.loc[
            observations["observation_dt"].dt.strftime("%Y%m%d").between(
                args.start,
                args.end,
            )
        ].reset_index(drop=True)
        magfilo_matches, unmatched_magfilo = matched_magfilo_frames(
            paths_df,
            observations,
            args.magfilo_window_hours,
        )
        overlaps, overlap_frames, scanned_frames = build_catalog_overlap_table(
            paths_df,
            load_kislovodsk_catalog(args.catalog),
            magfilo_matches,
            args.catalog_window_hours,
            args.catalog_support_radius_px,
            args.magfilo_polygon_tolerance_px,
            args.workers,
            args.magfilo_catalog,
            args.magfilo_fits_root,
            boundary_contact,
        )
        output_name = "Overlap Scan"
        if boundary_contact is not None:
            output_name = "Boundary Contacts"
            parameters = {
                "max_distance_arcsec": args.boundary_distance_arcsec,
                "max_tangent_difference_deg": args.boundary_angle_deg,
                "min_run_arcsec": args.boundary_min_run_arcsec,
                "kislovodsk_window_hours": args.catalog_window_hours,
                "magfilo_window_hours": args.magfilo_window_hours,
            }
            overlaps.attrs["boundary_contact"] = parameters
            overlap_frames.attrs["boundary_contact"] = parameters
        overlaps_path = (
            args.output_overlaps_parquet
            or output_dir / f"{output_name} {stem}.parquet"
        )
        overlaps_path.parent.mkdir(parents=True, exist_ok=True)
        overlaps.to_parquet(overlaps_path, index=False)
        overlap_frames.to_parquet(
            overlaps_path.with_suffix(".frames.parquet"),
            index=False,
        )
        print(
            f"Scanned {scanned_frames} catalog-covered frames; "
            f"MAGFiLO matched: {sum(len(matches) for matches in magfilo_matches.values())} "
            f"observations on {len(magfilo_matches)} frames / {len(observations)} "
            f"({unmatched_magfilo} outside {args.magfilo_window_hours:g} h)."
        )
        if boundary_contact is not None:
            for name, catalog in (("Kislovodsk", "kislovodsk"), ("MAGFiLO", "magfilo")):
                count = overlap_frames.get(f"{catalog}_boundary_contact_components")
                total = 0 if count is None else int(count.sum())
                print(f"{name} aligned boundary contacts: {total}.")
        else:
            print(
                f"Kislovodsk contacts: "
                f"{int(overlap_frames['kislovodsk_contact_components'].sum())}; "
                f"MAGFiLO direct polygon contacts: "
                f"{int(overlap_frames['magfilo_direct_overlap_components'].sum())}; "
                f"MAGFiLO tolerance contacts: "
                f"{int(overlap_frames['magfilo_tolerance_overlap_components'].sum())}."
            )
        print(f"Saved {overlaps_path}")
        print(f"Saved {overlaps_path.with_suffix('.frames.parquet')}")
        return 0

    if args.mode in ("labels", "review"):
        assert args.magfilo_window_hours > 0.0
        paths_df = pd.read_parquet(args.paths_parquet)
        start_key = f"{args.start}_0000"
        end_key = f"{args.end}_9999"
        paths_df = paths_df.loc[start_key:end_key].copy()
        if args.max_frames is not None:
            paths_df = paths_df.iloc[: args.max_frames]
        assert not paths_df.empty, "No observations in the requested interval."
        assert paths_df[["fits_path", "mask_path"]].notna().all().all(), (
            "Catalog labels require AIA 193 and mask paths for every frame."
        )

        magfilo_catalog = load_magfilo(args.magfilo_catalog)
        observations = magfilo_observations(magfilo_catalog)
        observations = observations.loc[
            observations["observation_dt"].dt.strftime("%Y%m%d").between(
                args.start,
                args.end,
            )
        ].reset_index(drop=True)
        magfilo_matches, unmatched_magfilo = matched_magfilo_frames(
            paths_df,
            observations,
            args.magfilo_window_hours,
        )
        magfilo_fits = magfilo_fits_by_name(args.magfilo_fits_root)
        if args.mode == "review":
            review_dir = (
                args.review_output_dir
                or output_dir / f"MAGFiLO Review {stem}"
            )
            rendered = render_magfilo_review(
                paths_df,
                magfilo_catalog,
                magfilo_matches,
                magfilo_fits,
                review_dir,
            )
            print(
                f"Rendered {rendered} MAGFiLO-covered AIA frames; "
                f"{unmatched_magfilo} observations were outside "
                f"{args.magfilo_window_hours:g} h."
            )
            print(f"Saved {review_dir / 'index.html'}")
            return 0
        labels, label_frames = build_catalog_label_table(
            paths_df,
            load_kislovodsk_catalog(args.catalog),
            magfilo_catalog,
            magfilo_matches,
            magfilo_fits,
            args.catalog_window_hours,
            args.catalog_support_radius_px,
            args.catalog_alignment_tolerance_deg,
            args.catalog_min_supported_centerline_px,
            args.magfilo_support_radius_px,
            args.magfilo_polygon_tolerance_px,
            args.magfilo_alignment_tolerance_deg,
            args.magfilo_min_aligned_centerline_px,
            args.magfilo_min_aligned_centerline_fraction,
        )
        assert not labels.empty, "No components were found in catalog-covered frames."
        labels_path = (
            args.output_labels_parquet
            or output_dir / f"Labels {stem}.parquet"
        )
        labels_path.parent.mkdir(parents=True, exist_ok=True)
        labels.to_parquet(labels_path, index=False)
        matched_magfilo_observations = sum(
            len(matches) for matches in magfilo_matches.values()
        )
        print(
            f"Catalog labels: {len(labels)} components across {label_frames} frames; "
            f"MAGFiLO matched: {matched_magfilo_observations} observations on "
            f"{len(magfilo_matches)} AIA frames / {len(observations)} "
            f"({unmatched_magfilo} outside {args.magfilo_window_hours:g} h)."
        )
        print(
            "Kislovodsk positives: "
            f"{int(labels['kislovodsk_is_filament'].fillna(0).sum())}; "
            "MAGFiLO positives: "
            f"{int(labels['magfilo_is_filament'].fillna(0).sum())}."
        )
        print(f"Saved {labels_path}")
        return 0

    if args.reuse_features:
        features = pd.read_parquet(features_path)
    else:
        paths_df = pd.read_parquet(args.paths_parquet)
        start_key = f"{args.start}_0000"
        end_key = f"{args.end}_9999"
        paths_df = paths_df.loc[start_key:end_key].copy()
        if args.max_frames is not None:
            paths_df = paths_df.iloc[: args.max_frames]
        assert not paths_df.empty, "No observations in the requested interval."

        catalog = load_kislovodsk_catalog(args.catalog)
        label_frame_keys = (
            external_label_frame_keys(args.labels_parquet)
            if args.training_only and args.labels_parquet is not None
            else None
        )
        features = build_filament_feature_table(
            paths_df,
            catalog,
            catalog_window_hours=args.catalog_window_hours,
            catalog_support_radius_px=args.catalog_support_radius_px,
            catalog_alignment_tolerance_deg=args.catalog_alignment_tolerance_deg,
            catalog_min_supported_centerline_px=(
                args.catalog_min_supported_centerline_px
            ),
            training_only=args.training_only,
            label_frame_keys=label_frame_keys,
            workers=args.workers,
        )
        assert not features.empty, "No mask components were produced."
        features_path.parent.mkdir(parents=True, exist_ok=True)
        features.to_parquet(features_path, index=False)
        print(f"Saved {features_path}")

    if "catalog_available" not in features:
        features["catalog_available"] = True
    catalog_covered = features["catalog_available"].fillna(False).astype(bool)
    features.loc[catalog_covered, "is_filament"] = (
        features.loc[catalog_covered, "catalog_supported_centerline_px"]
        >= args.catalog_min_supported_centerline_px
    ).astype(int)
    features.loc[~catalog_covered, "is_filament"] = np.nan

    if args.labels_parquet is not None:
        features = merge_external_labels(features, args.labels_parquet)
    labeled = assign_training_labels(features)
    print(
        f"Components: {len(features)}; labeled: {int(labeled.sum())}; "
        f"filaments: {int(features['is_filament'].fillna(0).sum())}; "
        f"frames: {features['frame_key'].nunique()}; "
        f"labeled frames: {features.loc[labeled, 'frame_key'].nunique()}"
    )
    print(features.loc[labeled, "label_source"].value_counts().to_string())
    if args.features_only:
        return 0

    features = features.loc[labeled].copy()
    assert not features.empty, "No catalog-covered components are available."
    assert pd.to_datetime(features["observation_dt"]).max() < pd.Timestamp("2018-01-01"), (
        "2018 is locked out for final evaluation and cannot be used for training."
    )
    train, validation = split_by_day(features, args.validation_fraction)
    (
        train_values,
        validation_values,
        medians,
        means,
        scales,
    ) = prepare_feature_values(train, validation)
    train_labels = train["is_filament"].to_numpy(dtype=np.float64)
    validation_labels = validation["is_filament"].to_numpy(dtype=np.float64)

    coefficients, intercept, class_weights, optimizer = fit_logistic(
        train_values,
        train_labels,
        args.l2,
        args.max_iterations,
    )
    validation_probabilities = sigmoid(validation_values @ coefficients + intercept)
    metrics_at_0_5 = classification_metrics(
        validation_labels,
        validation_probabilities,
        threshold=0.5,
    )
    metrics = select_precision_threshold(
        validation_labels,
        validation_probabilities,
        args.target_precision,
    )
    metrics.update(
        {
            "probability_threshold": metrics["threshold"],
            "metrics_at_0_5": metrics_at_0_5,
            "target_precision": args.target_precision,
            "train_components": len(train),
            "validation_components": len(validation),
            "train_frames": train["frame_key"].nunique(),
            "validation_frames": validation["frame_key"].nunique(),
            "train_filaments": int(train_labels.sum()),
            "validation_filaments": int(validation_labels.sum()),
            "catalog_window_hours": args.catalog_window_hours,
            "catalog_support_radius_px": args.catalog_support_radius_px,
            "catalog_alignment_tolerance_deg": args.catalog_alignment_tolerance_deg,
            "catalog_distance_quantile": CATALOG_DISTANCE_QUANTILE,
            "catalog_min_supported_centerline_px": (
                args.catalog_min_supported_centerline_px
            ),
            "feature_columns": FEATURE_COLUMNS,
            "optimizer_iterations": int(optimizer.nit),
            "optimizer_loss": float(optimizer.fun),
        }
    )
    model = {
        "model_type": "l2_logistic_regression",
        "feature_columns": FEATURE_COLUMNS,
        "feature_medians": medians.to_dict(),
        "feature_means": means.to_dict(),
        "feature_scales": scales.to_dict(),
        "coefficients": dict(zip(FEATURE_COLUMNS, coefficients)),
        "intercept": intercept,
        "l2": args.l2,
        "class_weights": class_weights,
        "probability_threshold": metrics["probability_threshold"],
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text(json.dumps(model, indent=2))
    metrics_path = model_path.with_suffix(".metrics.json")
    metrics_path.write_text(json.dumps(metrics, indent=2))
    validation_output = validation[
        [
            "frame_key",
            "observation_dt",
            "component_id",
            "is_filament",
            "label_source",
        ]
    ].copy()
    validation_output["filament_probability"] = validation_probabilities
    validation_output.to_parquet(
        model_path.with_suffix(".validation.parquet"),
        index=False,
    )
    print(f"Saved {model_path}")
    print(f"Saved {metrics_path}")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
