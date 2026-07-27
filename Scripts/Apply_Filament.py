#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import PIL.Image
from scipy import ndimage
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from Library.Config import paths
from Library.Filaments import FEATURE_COLUMNS
from Library.IO import prepare_mask


def sigmoid(values):
    positive = values >= 0
    probabilities = np.empty_like(values, dtype=np.float64)
    probabilities[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    probabilities[~positive] = exponential / (1.0 + exponential)
    return probabilities


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Remove classifier-positive or catalog-contact components from IDL masks.",
    )
    parser.add_argument("start", help="inclusive YYYYMMDD")
    parser.add_argument("end", help="inclusive YYYYMMDD")
    selection_source = parser.add_mutually_exclusive_group(required=True)
    selection_source.add_argument("--model-path", type=Path)
    selection_source.add_argument("--contacts-parquet", type=Path)
    parser.add_argument("--features-parquet", type=Path)
    parser.add_argument(
        "--min-kislovodsk-supported-centerline-px",
        type=int,
        default=1,
        help="Minimum Kislovodsk-supported skeleton pixels for contact removal.",
    )
    parser.add_argument(
        "--paths-parquet",
        type=Path,
        default=Path(paths["artifact_root"]) / "Paths.parquet",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=ROOT_DIR / "Outputs" / "Filaments" / "Masks",
    )
    parser.add_argument("--output-paths-parquet", type=Path)
    parser.add_argument("--probability-threshold", type=float)
    args = parser.parse_args(argv)

    start_key = f"{args.start}_0000"
    end_key = f"{args.end}_9999"
    paths_df = pd.read_parquet(args.paths_parquet).loc[start_key:end_key].copy()
    assert not paths_df.empty, "No observations in the requested interval."

    if args.contacts_parquet is not None:
        assert args.features_parquet is None, (
            "--features-parquet is only valid with --model-path"
        )
        contacts = pd.read_parquet(args.contacts_parquet)
        required = {
            "frame_key",
            "component_id",
            "kislovodsk_supported_centerline_px",
        }
        missing = required.difference(contacts.columns)
        assert not missing, f"Contact table is missing columns: {sorted(missing)}"
        contacts = contacts[
            (contacts["frame_key"] >= start_key)
            & (contacts["frame_key"] <= end_key)
            & (
                contacts["kislovodsk_supported_centerline_px"]
                >= args.min_kislovodsk_supported_centerline_px
            )
        ].copy()
        contacts_by_frame = {
            frame_key: frame_contacts
            for frame_key, frame_contacts in contacts.groupby("frame_key")
        }
        assert set(contacts_by_frame).issubset(paths_df.index), (
            "Contact table contains frames absent from the supplied Paths parquet."
        )
        removal_source = "kislovodsk-contact"
    else:
        assert args.features_parquet is not None, (
            "--features-parquet is required with --model-path"
        )
        features = pd.read_parquet(args.features_parquet)
        features = features[
            (features["frame_key"] >= start_key) & (features["frame_key"] <= end_key)
        ].copy()
        assert not features.empty, "No component features in the requested interval."

        model = json.loads(args.model_path.read_text())
        assert model["model_type"] == "l2_logistic_regression"
        assert model["feature_columns"] == FEATURE_COLUMNS
        medians = pd.Series(model["feature_medians"])[FEATURE_COLUMNS]
        means = pd.Series(model["feature_means"])[FEATURE_COLUMNS]
        scales = pd.Series(model["feature_scales"])[FEATURE_COLUMNS]
        coefficients = pd.Series(model["coefficients"])[FEATURE_COLUMNS]
        assert medians.notna().all()
        assert means.notna().all()
        assert scales.notna().all() and (scales > 0.0).all()
        assert coefficients.notna().all()
        probability_threshold = (
            args.probability_threshold
            if args.probability_threshold is not None
            else float(model["probability_threshold"])
        )

        feature_values = features[FEATURE_COLUMNS].fillna(medians)
        standardized = (
            (feature_values - means) / scales
        ).to_numpy(dtype=np.float64)
        features["filament_probability"] = sigmoid(
            standardized @ coefficients.to_numpy(dtype=np.float64)
            + float(model["intercept"])
        )
        features["remove_component"] = (
            features["filament_probability"] >= probability_threshold
        )
        removal_source = "classifier"

    output_rows = []
    filamentless_paths = paths_df.copy()
    filamentless_paths["original_mask_path"] = filamentless_paths["mask_path"]

    for frame_key, observation in tqdm(
        paths_df.iterrows(),
        total=len(paths_df),
        desc="Filamentless masks",
    ):
        mask = prepare_mask(observation.mask_path).astype(bool)
        labels, component_count = ndimage.label(
            mask,
            structure=np.ones((3, 3), dtype=int),
        )
        expected_components = set(range(1, component_count + 1))
        if args.contacts_parquet is not None:
            frame_contacts = contacts_by_frame.get(frame_key)
            remove_ids = (
                frame_contacts["component_id"].drop_duplicates().to_numpy(dtype=int)
                if frame_contacts is not None
                else np.empty(0, dtype=int)
            )
            assert set(remove_ids).issubset(expected_components), (
                f"{frame_key}: contact components {sorted(remove_ids)} do not match "
                f"mask components {sorted(expected_components)}"
            )
            max_probability = np.nan
            max_supported_centerline_px = (
                int(frame_contacts["kislovodsk_supported_centerline_px"].max())
                if frame_contacts is not None
                else 0
            )
        else:
            frame_features = features[features["frame_key"] == frame_key]
            actual_components = set(frame_features["component_id"].astype(int))
            assert actual_components == expected_components, (
                f"{frame_key}: feature components {sorted(actual_components)} do not match "
                f"mask components {sorted(expected_components)}"
            )
            remove_ids = frame_features.loc[
                frame_features["remove_component"],
                "component_id",
            ].to_numpy(dtype=int)
            max_probability = float(frame_features["filament_probability"].max())
            max_supported_centerline_px = np.nan
        cleaned = mask & ~np.isin(labels, remove_ids)
        year = frame_key[:4]
        month = frame_key[4:6]
        output_dir = args.output_root / year / month
        output_dir.mkdir(parents=True, exist_ok=True)
        input_name = Path(observation.mask_path).name
        if input_name.endswith("_CH_MASK_FINAL.png"):
            output_name = input_name.replace(
                "_CH_MASK_FINAL.png",
                "_CH_MASK_FILAMENTLESS.png",
            )
        else:
            output_name = f"{Path(input_name).stem}_FILAMENTLESS.png"
        output_path = output_dir / output_name
        PIL.Image.fromarray((cleaned.astype(np.uint8) * 255)).save(output_path)
        filamentless_paths.loc[frame_key, "mask_path"] = str(output_path)

        output_rows.append(
            {
                "frame_key": frame_key,
                "original_mask_path": observation.mask_path,
                "mask_path": str(output_path),
                "components": component_count,
                "components_removed": len(remove_ids),
                "pixels_removed": int(mask.sum() - cleaned.sum()),
                "removal_source": removal_source,
                "max_filament_probability": max_probability,
                "max_kislovodsk_supported_centerline_px": max_supported_centerline_px,
            }
        )

    output_paths_parquet = args.output_paths_parquet or (
        ROOT_DIR
        / "Outputs"
        / "Filaments"
        / f"Paths Filamentless {args.start}-{args.end}.parquet"
    )
    output_paths_parquet.parent.mkdir(parents=True, exist_ok=True)
    filamentless_paths.to_parquet(output_paths_parquet)
    summary_path = output_paths_parquet.with_suffix(".summary.parquet")
    summary = pd.DataFrame(output_rows).set_index("frame_key")
    summary.to_parquet(summary_path)
    print(
        f"Removed {int(summary['components_removed'].sum())} components and "
        f"{int(summary['pixels_removed'].sum())} pixels."
    )
    print(f"Saved {output_paths_parquet}")
    print(f"Saved {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
