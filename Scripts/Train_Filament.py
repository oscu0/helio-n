#!/usr/bin/env python3
import argparse
import html
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.optimize import minimize
import sunpy.map
from matplotlib import pyplot as plt
from tqdm import tqdm
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from Library.Config import filament_feature_workers, paths
from Library.Filaments import (
    CATALOG_ALIGNMENT_TOLERANCE_DEG,
    CATALOG_DISTANCE_QUANTILE,
    CATALOG_MIN_ALIGNED_CENTERLINE_FRACTION,
    CATALOG_MIN_ALIGNED_CENTERLINE_PX,
    FEATURE_COLUMNS,
    MAGFILO_ALIGNMENT_TOLERANCE_DEG,
    MAGFILO_MIN_ALIGNED_CENTERLINE_FRACTION,
    MAGFILO_MIN_ALIGNED_CENTERLINE_PX,
    MAGFILO_POLYGON_TOLERANCE_PX,
    MAGFILO_SUPPORT_RADIUS_PX,
    build_filament_feature_table,
    compute_catalog_centerline_metrics,
    compute_magfilo_centerline_metrics,
    load_kislovodsk_catalog,
    magfilo_is_filament,
    rasterize_catalog_segments,
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


def build_catalog_label_table(
    paths_df,
    kislovodsk_catalog,
    magfilo_catalog,
    magfilo_matches,
    magfilo_fits,
    catalog_window_hours,
    catalog_support_radius_px,
    catalog_alignment_tolerance_deg,
    catalog_min_aligned_centerline_px,
    catalog_min_aligned_centerline_fraction,
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
                        int(
                            kislovodsk_metrics["catalog_aligned_centerline_px"]
                            >= catalog_min_aligned_centerline_px
                            and kislovodsk_metrics[
                                "catalog_aligned_centerline_fraction"
                            ]
                            >= catalog_min_aligned_centerline_fraction
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
        choices=("train", "labels", "review"),
        default="train",
        help="Run model training, build labels, or render the MAGFiLO review gallery.",
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
    parser.add_argument("--features-parquet", type=Path)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--catalog-window-hours", type=float, default=0.5)
    parser.add_argument("--catalog-support-radius-px", type=float, default=10.0)
    parser.add_argument(
        "--catalog-alignment-tolerance-deg",
        type=float,
        default=CATALOG_ALIGNMENT_TOLERANCE_DEG,
    )
    parser.add_argument(
        "--catalog-min-aligned-centerline-px",
        type=int,
        default=CATALOG_MIN_ALIGNED_CENTERLINE_PX,
    )
    parser.add_argument(
        "--catalog-min-aligned-centerline-fraction",
        type=float,
        default=CATALOG_MIN_ALIGNED_CENTERLINE_FRACTION,
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
            args.catalog_min_aligned_centerline_px,
            args.catalog_min_aligned_centerline_fraction,
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
            catalog_min_aligned_centerline_px=(
                args.catalog_min_aligned_centerline_px
            ),
            catalog_min_aligned_centerline_fraction=(
                args.catalog_min_aligned_centerline_fraction
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
        (
            features.loc[catalog_covered, "catalog_aligned_centerline_px"]
            >= args.catalog_min_aligned_centerline_px
        )
        & (
            features.loc[catalog_covered, "catalog_aligned_centerline_fraction"]
            >= args.catalog_min_aligned_centerline_fraction
        )
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
            "catalog_min_aligned_centerline_px": (
                args.catalog_min_aligned_centerline_px
            ),
            "catalog_min_aligned_centerline_fraction": (
                args.catalog_min_aligned_centerline_fraction
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
