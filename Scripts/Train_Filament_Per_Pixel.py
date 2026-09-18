#!/usr/bin/env python3
"""Train a region classifier from MAGFiLO polygon regions.

The name is retained from the initial pixel-classifier proposal.  This trainer
uses each projected MAGFiLO polygon as a positive *region*, because the HMI
polarity features are meaningful over a region rather than at one pixel.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.optimize import minimize
import sunpy.map
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from Library.Config import filament_feature_workers, paths
from Library.Filaments import (
    FEATURE_COLUMNS,
    CATALOG_ALIGNMENT_TOLERANCE_DEG,
    CATALOG_MIN_ALIGNED_CENTERLINE_FRACTION,
    CATALOG_MIN_ALIGNED_CENTERLINE_PX,
    compute_hmi_input,
    summarize_component,
)
from Library.GONG import (
    load_magfilo,
    magfilo_observations,
    project_magfilo_observation,
    rasterize_projected_annotations,
)
from Library.IO import prepare_fits, prepare_mask


PHYSICAL_FEATURE_COLUMNS = [
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

WORKER_CATALOG = None
WORKER_FITS = None


def sigmoid(values):
    positive = values >= 0.0
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


def classification_metrics(labels, probabilities, threshold):
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
        "threshold": float(threshold),
        "accuracy": float(np.mean(predictions == labels)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "true_positive": true_positive,
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
    }


def select_precision_threshold(labels, probabilities, target_precision):
    candidates = np.unique(np.append(probabilities, 0.5))
    scored = [
        classification_metrics(labels, probabilities, threshold)
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


def split_by_day(features, validation_fraction):
    days = pd.to_datetime(features["observation_dt"]).dt.normalize().drop_duplicates()
    assert len(days) >= 2, "At least two MAGFiLO-covered days are required."
    validation_days = max(1, int(np.ceil(len(days) * validation_fraction)))
    split_day = days.iloc[-validation_days]
    validation = features.loc[
        pd.to_datetime(features["observation_dt"]).dt.normalize() >= split_day
    ].copy()
    train = features.loc[features.index.difference(validation.index)].copy()
    assert train["is_filament"].nunique() == 2, (
        "Training split must contain polygon and Dec1-component regions."
    )
    assert validation["is_filament"].nunique() == 2, (
        "Validation split must contain polygon and Dec1-component regions."
    )
    return train, validation


def prepare_feature_values(train, validation, feature_columns):
    medians = train[feature_columns].median()
    assert medians.notna().all(), (
        "Features without a finite training median: "
        f"{medians.index[medians.isna()].tolist()}"
    )
    train_imputed = train[feature_columns].fillna(medians)
    validation_imputed = validation[feature_columns].fillna(medians)
    means = train_imputed.mean()
    scales = train_imputed.std(ddof=0)
    scales[scales == 0.0] = 1.0
    return (
        ((train_imputed - means) / scales).to_numpy(dtype=np.float64),
        ((validation_imputed - means) / scales).to_numpy(dtype=np.float64),
        medians,
        means,
        scales,
    )


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
                "url": observation.url,
                "image_ids": observation.image_ids,
                "observation_dt": observation.observation_dt,
                "time_offset_hours": offset_hours,
            }
        )
    return matches, unmatched


def initialize_worker(catalog_path, fits_root):
    global WORKER_CATALOG, WORKER_FITS
    WORKER_CATALOG = load_magfilo(catalog_path)
    fits_paths = list(Path(fits_root).rglob("*.fits.fz"))
    WORKER_FITS = {path.name: path for path in fits_paths}
    assert len(WORKER_FITS) == len(fits_paths), (
        f"MAGFiLO FITS cache has duplicate filenames under {fits_root}"
    )


def summarize_region(component, aia193, aia304, hmi_los, hmi_radial, hmi_valid):
    return summarize_component(
        component,
        aia193,
        aia304,
        hmi_los,
        hmi_radial,
        hmi_valid,
        np.full(component.shape, np.inf),
        False,
        np.empty((0, 4), dtype=np.float32),
        1.0,
        CATALOG_ALIGNMENT_TOLERANCE_DEG,
        CATALOG_MIN_ALIGNED_CENTERLINE_PX,
        CATALOG_MIN_ALIGNED_CENTERLINE_FRACTION,
    )


def build_region_rows(task):
    frame_key, observation, magfilo_matches, negative_buffer_px = task
    aia_map, aia193 = prepare_fits(observation["fits_path"])
    _, aia304 = prepare_fits(observation["aia304_path"])
    candidate_mask = prepare_mask(observation["mask_path"]).astype(bool)
    assert candidate_mask.shape == aia193.shape == aia304.shape == aia_map.data.shape
    hmi_los, hmi_radial, hmi_valid = compute_hmi_input(
        aia_map,
        observation["hmi_path"],
    )

    projected = []
    magfilo_times = []
    for match in magfilo_matches:
        fits_name = Path(match["url"]).stem + ".fits.fz"
        assert fits_name in WORKER_FITS, (
            f"MAGFiLO FITS is not cached for {frame_key}: {fits_name}"
        )
        gong_map = sunpy.map.Map(WORKER_FITS[fits_name])
        projected.extend(
            project_magfilo_observation(
                WORKER_CATALOG,
                {"image_ids": match["image_ids"]},
                gong_map,
                aia_map,
            )
        )
        magfilo_times.append(match["observation_dt"])

    polygon_union = rasterize_projected_annotations(projected, candidate_mask.shape)
    polygon_buffer = ndimage.binary_dilation(
        polygon_union,
        iterations=negative_buffer_px,
    )
    observation_dt = pd.to_datetime(frame_key, format="%Y%m%d_%H%M")
    base = {
        "frame_key": frame_key,
        "observation_dt": observation_dt,
        "magfilo_observation_dt": min(magfilo_times),
        "magfilo_time_offset_hours": min(
            abs((observation_dt - timestamp).total_seconds()) / 3600.0
            for timestamp in magfilo_times
        ),
    }
    rows = []
    for annotation in projected:
        polygon_mask = rasterize_projected_annotations([annotation], candidate_mask.shape)
        if not polygon_mask.any():
            continue
        rows.append(
            {
                **base,
                **summarize_region(
                    polygon_mask,
                    aia193,
                    aia304,
                    hmi_los,
                    hmi_radial,
                    hmi_valid,
                ),
                "region_source": "magfilo_polygon",
                "is_filament": 1,
                "magfilo_annotation_id": annotation["annotation_id"],
                "component_id": np.nan,
            }
        )

    labels, component_count = ndimage.label(
        candidate_mask,
        structure=np.ones((3, 3), dtype=int),
    )
    for component_id in range(1, component_count + 1):
        component = labels == component_id
        if (component & polygon_buffer).any():
            continue
        rows.append(
            {
                **base,
                **summarize_region(
                    component,
                    aia193,
                    aia304,
                    hmi_los,
                    hmi_radial,
                    hmi_valid,
                ),
                "region_source": "dec1_component",
                "is_filament": 0,
                "magfilo_annotation_id": None,
                "component_id": component_id,
            }
        )

    return rows, {
        "frame_key": frame_key,
        "magfilo_polygons": int(sum(row["is_filament"] for row in rows)),
        "dec1_negatives": int(sum(row["is_filament"] == 0 for row in rows)),
        "polygon_pixels": int(polygon_union.sum()),
        "candidate_pixels": int(candidate_mask.sum()),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Train a region-level filament classifier from MAGFiLO polygons.",
    )
    parser.add_argument("start", help="inclusive YYYYMMDD")
    parser.add_argument("end", help="inclusive YYYYMMDD")
    parser.add_argument(
        "--paths-parquet",
        type=Path,
        default=Path(paths["artifact_root"]) / "Paths.parquet",
    )
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
    parser.add_argument("--negative-buffer-px", type=int, default=4)
    parser.add_argument(
        "--feature-set",
        choices=("physical", "all"),
        default="physical",
        help="Use physical AIA/HMI features only, or additionally include region shape.",
    )
    parser.add_argument("--features-parquet", type=Path)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--l2", type=float, default=1.0)
    parser.add_argument("--max-iterations", type=int, default=1000)
    parser.add_argument("--target-precision", type=float, default=0.9)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--workers", type=int, default=filament_feature_workers)
    parser.add_argument("--reuse-features", action="store_true")
    parser.add_argument("--features-only", action="store_true")
    args = parser.parse_args(argv)

    assert args.end < "20180101", "2018 is reserved for final evaluation."
    assert args.magfilo_window_hours > 0.0
    assert args.negative_buffer_px >= 0
    assert 0.0 < args.validation_fraction < 1.0
    assert args.l2 >= 0.0
    assert 0.0 < args.target_precision <= 1.0
    assert args.workers >= 1

    output_dir = ROOT_DIR / "Outputs" / "Filaments"
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.start}-{args.end}"
    features_path = args.features_parquet or output_dir / f"Polygon Features {stem}.parquet"
    model_path = args.model_path or output_dir / f"Polygon Classifier {stem}.json"
    feature_columns = (
        PHYSICAL_FEATURE_COLUMNS if args.feature_set == "physical" else FEATURE_COLUMNS
    )

    if args.reuse_features:
        features = pd.read_parquet(features_path)
    else:
        paths_df = pd.read_parquet(args.paths_parquet).loc[
            f"{args.start}_0000":f"{args.end}_9999"
        ].copy()
        required_paths = ["fits_path", "mask_path", "hmi_path", "aia304_path"]
        available = paths_df[required_paths].notna().all(axis=1)
        existing_paths = pd.Series(False, index=paths_df.index)
        existing_paths.loc[available] = paths_df.loc[available, required_paths].map(
            lambda path: Path(path).is_file()
        ).all(axis=1)
        available &= existing_paths
        skipped_missing = int((~available).sum())
        paths_df = paths_df.loc[available]
        if args.max_frames is not None:
            paths_df = paths_df.iloc[: args.max_frames]
        assert not paths_df.empty, "No complete AIA/HMI/mask observations in range."

        catalog = load_magfilo(args.magfilo_catalog)
        observations = magfilo_observations(catalog)
        observations = observations.loc[
            observations["observation_dt"].dt.strftime("%Y%m%d").between(
                args.start,
                args.end,
            )
        ].reset_index(drop=True)
        matches, unmatched = matched_magfilo_frames(
            paths_df,
            observations,
            args.magfilo_window_hours,
        )
        assert matches, "No MAGFiLO observations matched complete feature frames."
        tasks = [
            (
                frame_key,
                paths_df.loc[frame_key].to_dict(),
                matches[frame_key],
                args.negative_buffer_px,
            )
            for frame_key in sorted(matches)
        ]
        initializer_args = (args.magfilo_catalog, args.magfilo_fits_root)
        rows = []
        summaries = []
        if args.workers == 1:
            initialize_worker(*initializer_args)
            results = map(build_region_rows, tasks)
            for frame_rows, summary in tqdm(
                results,
                total=len(tasks),
                desc="MAGFiLO polygon features",
            ):
                rows.extend(frame_rows)
                summaries.append(summary)
        else:
            context = mp.get_context("spawn")
            with ProcessPoolExecutor(
                max_workers=args.workers,
                mp_context=context,
                initializer=initialize_worker,
                initargs=initializer_args,
            ) as executor:
                results = executor.map(build_region_rows, tasks)
                for frame_rows, summary in tqdm(
                    results,
                    total=len(tasks),
                    desc=f"MAGFiLO polygon features ({args.workers} workers)",
                ):
                    rows.extend(frame_rows)
                    summaries.append(summary)
        features = pd.DataFrame(rows)
        assert not features.empty, "No projected MAGFiLO polygon or Dec1 regions found."
        features_path.parent.mkdir(parents=True, exist_ok=True)
        features.to_parquet(features_path, index=False)
        pd.DataFrame(summaries).to_parquet(
            features_path.with_suffix(".frames.parquet"),
            index=False,
        )
        print(
            f"Matched {len(matches)} AIA frames; {unmatched} MAGFiLO observations "
            f"were outside {args.magfilo_window_hours:g} h; skipped {skipped_missing} "
            "frames with missing inputs."
        )
        print(f"Saved {features_path}")

    assert set(feature_columns).issubset(features.columns)
    print(
        f"Regions: {len(features)}; MAGFiLO polygons: "
        f"{int(features['is_filament'].sum())}; Dec1 negatives: "
        f"{int((features['is_filament'] == 0).sum())}; frames: "
        f"{features['frame_key'].nunique()}; feature set: {args.feature_set}."
    )
    if args.features_only:
        return 0

    train, validation = split_by_day(features, args.validation_fraction)
    train_values, validation_values, medians, means, scales = prepare_feature_values(
        train,
        validation,
        feature_columns,
    )
    train_labels = train["is_filament"].to_numpy(dtype=np.float64)
    validation_labels = validation["is_filament"].to_numpy(dtype=np.float64)
    coefficients, intercept, class_weights, optimizer = fit_logistic(
        train_values,
        train_labels,
        args.l2,
        args.max_iterations,
    )
    validation_probabilities = sigmoid(validation_values @ coefficients + intercept)
    metrics = select_precision_threshold(
        validation_labels,
        validation_probabilities,
        args.target_precision,
    )
    metrics.update(
        {
            "probability_threshold": metrics["threshold"],
            "metrics_at_0_5": classification_metrics(
                validation_labels,
                validation_probabilities,
                0.5,
            ),
            "feature_set": args.feature_set,
            "feature_columns": feature_columns,
            "negative_buffer_px": args.negative_buffer_px,
            "magfilo_window_hours": args.magfilo_window_hours,
            "train_regions": len(train),
            "validation_regions": len(validation),
            "train_frames": train["frame_key"].nunique(),
            "validation_frames": validation["frame_key"].nunique(),
            "train_filaments": int(train_labels.sum()),
            "validation_filaments": int(validation_labels.sum()),
            "optimizer_iterations": int(optimizer.nit),
            "optimizer_loss": float(optimizer.fun),
        }
    )
    model = {
        "model_type": "l2_logistic_polygon_region_regression",
        "feature_columns": feature_columns,
        "feature_medians": medians.to_dict(),
        "feature_means": means.to_dict(),
        "feature_scales": scales.to_dict(),
        "coefficients": dict(zip(feature_columns, coefficients)),
        "intercept": intercept,
        "l2": args.l2,
        "class_weights": class_weights,
        "probability_threshold": metrics["probability_threshold"],
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_text(json.dumps(model, indent=2))
    model_path.with_suffix(".metrics.json").write_text(json.dumps(metrics, indent=2))
    validation_output = validation[
        [
            "frame_key",
            "observation_dt",
            "region_source",
            "is_filament",
            "magfilo_annotation_id",
            "component_id",
        ]
    ].copy()
    validation_output["filament_probability"] = validation_probabilities
    validation_output.to_parquet(model_path.with_suffix(".validation.parquet"), index=False)
    print(f"Saved {model_path}")
    print(f"Saved {model_path.with_suffix('.metrics.json')}")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
