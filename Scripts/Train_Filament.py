#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from Library.Config import paths
from Library.Filaments import (
    CATALOG_ALIGNMENT_TOLERANCE_DEG,
    CATALOG_DISTANCE_QUANTILE,
    CATALOG_MIN_ALIGNED_CENTERLINE_FRACTION,
    CATALOG_MIN_ALIGNED_CENTERLINE_PX,
    FEATURE_COLUMNS,
    build_filament_feature_table,
    load_kislovodsk_catalog,
)


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
    parser.add_argument("--features-parquet", type=Path)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--catalog-window-hours", type=float, default=12.0)
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
    parser.add_argument("--l2", type=float, default=1.0)
    parser.add_argument("--max-iterations", type=int, default=1000)
    parser.add_argument("--target-precision", type=float, default=0.9)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--reuse-features", action="store_true")
    parser.add_argument("--features-only", action="store_true")
    args = parser.parse_args(argv)
    assert 0.0 < args.validation_fraction < 1.0
    assert args.l2 >= 0.0
    assert 0.0 < args.target_precision <= 1.0

    output_dir = ROOT_DIR / "Outputs" / "Filaments"
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.start}-{args.end}"
    features_path = args.features_parquet or output_dir / f"Features {stem}.parquet"
    model_path = args.model_path or output_dir / f"Classifier {stem}.json"

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
