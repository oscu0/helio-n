#!/usr/bin/env python3
"""Render region-level Kislovodsk/ISWAT filament-label review galleries."""

import argparse
import html
import json
import re
import sys
from pathlib import Path

import astropy.units as u
import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
import sunpy.map
from scipy import ndimage
from skimage.morphology import skeletonize
from tqdm import tqdm

from matplotlib import pyplot as plt

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from Library.Filaments import (
    CATALOG_ALIGNMENT_TOLERANCE_DEG,
    CATALOG_MIN_ALIGNED_CENTERLINE_FRACTION,
    CATALOG_MIN_ALIGNED_CENTERLINE_PX,
    CATALOG_SUPPORT_RADIUS_PX,
    MAGFILO_ALIGNMENT_TOLERANCE_DEG,
    MAGFILO_MIN_ALIGNED_CENTERLINE_FRACTION,
    MAGFILO_MIN_ALIGNED_CENTERLINE_PX,
    MAGFILO_POLYGON_TOLERANCE_PX,
    MAGFILO_SUPPORT_RADIUS_PX,
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
    select_magfilo_observation,
)
from Library.IO import mask_fits, prepare_mask


ISWAT_TIME_PATTERN = re.compile(
    r"(\d{4})-(\d{2})-(\d{2})T(\d{2})_(\d{2})"
)
DEC1_MASK_TIME_PATTERN = re.compile(r"AIA(\d{8}_\d{6})_0193_CH_MASK_FINAL\.png$")
DEC1_HOST = "dec1.sinp.msu.ru"
DEC1_MASK_ROOT = "/mnt/sun/sdo/helio-n/masks/chsun/0193"


def date_key_from_path(path):
    match = ISWAT_TIME_PATTERN.search(path.name)
    assert match, f"Cannot obtain ISWAT timestamp from {path.name}"
    return "".join(match.groups()[:3]) + "_" + "".join(match.groups()[3:])


def normalized_display_data(aia_map):
    data = mask_fits(aia_map)
    if np.ma.isMaskedArray(data):
        data = data.filled(np.nan)
    data = np.flipud(np.asarray(data, dtype=np.float32))
    low, high = np.nanpercentile(data, [1, 99])
    data = np.clip(data, low, high)
    return np.nan_to_num((data - low) / (high - low + 1e-6), nan=0.0)


def index_iswat_cases(iswat_root):
    annotation_root = iswat_root / "Coronal Hole Labels" / "Labels"
    annotations = {}
    for annotation_path in annotation_root.glob("*.png"):
        key = date_key_from_path(annotation_path)
        annotations.setdefault(key, []).append(annotation_path)

    rows = []
    for fits_path in sorted((iswat_root / "193").glob("*.fits")):
        key = date_key_from_path(fits_path)
        candidates = sorted(
            annotations[key],
            key=lambda path: ("-annot" not in path.name, path.name),
        )
        rows.append(
            {
                "frame_key": key,
                "observation_dt": pd.to_datetime(key, format="%Y%m%d_%H%M"),
                "fits_path": fits_path,
                "annotation_path": candidates[0],
            }
        )
    cases = pd.DataFrame(rows).sort_values("observation_dt").reset_index(drop=True)
    assert not cases.empty, f"No ISWAT AIA 193 FITS under {iswat_root}"
    return cases


def index_dec1_masks(mask_root):
    rows = []
    for mask_path in mask_root.rglob("*_CH_MASK_FINAL.png"):
        match = DEC1_MASK_TIME_PATTERN.search(mask_path.name)
        assert match, f"Unexpected dec1 mask name: {mask_path.name}"
        rows.append(
            {
                "mask_path": mask_path,
                "mask_datetime": pd.to_datetime(
                    match.group(1),
                    format="%Y%m%d_%H%M%S",
                ),
                "dec1_mask_source": (
                    f"{DEC1_HOST}:{DEC1_MASK_ROOT}/"
                    f"{match.group(1)[:4]}/{match.group(1)[4:6]}/{mask_path.name}"
                ),
            }
        )
    masks = pd.DataFrame(rows).sort_values("mask_datetime").reset_index(drop=True)
    assert not masks.empty, f"No staged dec1 masks under {mask_root}"
    return masks


def match_dec1_masks(cases, masks):
    matched = []
    for case in cases.itertuples(index=False):
        same_day = masks[
            masks["mask_datetime"].dt.normalize()
            == case.observation_dt.normalize()
        ]
        assert not same_day.empty, f"No dec1 mask staged for {case.frame_key}"
        nearest_index = (same_day["mask_datetime"] - case.observation_dt).abs().idxmin()
        match = masks.loc[nearest_index]
        matched.append(
            {
                **case._asdict(),
                "mask_path": match.mask_path,
                "mask_datetime": match.mask_datetime,
                "dec1_mask_source": match.dec1_mask_source,
                "mask_time_offset_min": (
                    match.mask_datetime - case.observation_dt
                ).total_seconds()
                / 60.0,
            }
        )
    return pd.DataFrame(matched)


def match_magfilo_observations(cases, observations, window_hours, fits_root):
    matched = []
    for case in cases.itertuples(index=False):
        observation = select_magfilo_observation(
            observations,
            case.observation_dt,
            window_hours,
        )
        if observation is None:
            matched.append(
                {
                    **case._asdict(),
                    "magfilo_available": False,
                    "magfilo_observation_dt": pd.NaT,
                    "magfilo_time_offset_hours": np.nan,
                    "magfilo_url": None,
                    "magfilo_image_ids": [],
                    "magfilo_annotation_instances": 0,
                    "magfilo_filament_annotations": 0,
                    "magfilo_fits_path": None,
                }
            )
            continue

        fits_name = Path(observation.url).stem + ".fits.fz"
        fits_path = fits_root / fits_name
        assert fits_path.exists(), (
            f"MAGFiLO observation matched {case.frame_key} but its exact GONG FITS "
            f"is not cached: {fits_path}"
        )
        matched.append(
            {
                **case._asdict(),
                "magfilo_available": True,
                "magfilo_observation_dt": observation.observation_dt,
                "magfilo_time_offset_hours": observation.time_offset_hours,
                "magfilo_url": observation.url,
                "magfilo_image_ids": observation.image_ids,
                "magfilo_annotation_instances": observation.annotation_instances,
                "magfilo_filament_annotations": observation.filament_annotations,
                "magfilo_fits_path": str(fits_path),
            }
        )
    return pd.DataFrame(matched)


def render_case(
    case,
    kislovodsk_catalog,
    magfilo_catalog,
    output_path,
    kislovodsk_window_hours,
    kislovodsk_support_radius_px,
    kislovodsk_alignment_tolerance_deg,
    kislovodsk_min_aligned_centerline_px,
    kislovodsk_min_aligned_centerline_fraction,
    magfilo_support_radius_px,
    magfilo_polygon_tolerance_px,
    magfilo_alignment_tolerance_deg,
    magfilo_min_aligned_centerline_px,
    magfilo_min_aligned_centerline_fraction,
):
    source_map = sunpy.map.Map(case.fits_path)
    aia_map = source_map.resample(u.Quantity([1024, 1024], u.pixel))
    aia193 = normalized_display_data(aia_map)
    candidate_mask = prepare_mask(case.mask_path).astype(bool)
    assert candidate_mask.shape == aia193.shape == aia_map.data.shape

    kislovodsk_filaments = select_catalog_segments(
        kislovodsk_catalog,
        case.observation_dt,
        kislovodsk_window_hours,
    )
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
        kislovodsk_distance = np.full(candidate_mask.shape, np.inf)
        kislovodsk_rasterized = 0
        kislovodsk_segments = np.empty((0, 4), dtype=np.float32)

    magfilo_projected = []
    if case.magfilo_available:
        gong_map = sunpy.map.Map(case.magfilo_fits_path)
        magfilo_projected = project_magfilo_observation(
            magfilo_catalog,
            {"image_ids": case.magfilo_image_ids},
            gong_map,
            aia_map,
        )
        magfilo_polygon_mask = rasterize_projected_annotations(
            magfilo_projected,
            candidate_mask.shape,
        )
        magfilo_spine_mask = rasterize_projected_spines(
            magfilo_projected,
            candidate_mask.shape,
        )
        magfilo_spine_segments = projected_spine_segments(magfilo_projected)
        magfilo_distance = (
            ndimage.distance_transform_edt(~magfilo_spine_mask)
            if magfilo_spine_mask.any()
            else np.full(candidate_mask.shape, np.inf)
        )
    else:
        magfilo_polygon_mask = np.zeros(candidate_mask.shape, dtype=bool)
        magfilo_spine_mask = np.zeros(candidate_mask.shape, dtype=bool)
        magfilo_spine_segments = np.empty((0, 4), dtype=np.float32)
        magfilo_distance = np.full(candidate_mask.shape, np.inf)

    labels, component_count = ndimage.label(
        candidate_mask,
        structure=np.ones((3, 3), dtype=int),
    )
    records = []
    for component_id in range(1, component_count + 1):
        component = labels == component_id
        kislovodsk_metrics = compute_catalog_centerline_metrics(
            component,
            kislovodsk_distance,
            kislovodsk_available,
            projected_segments=kislovodsk_segments,
            support_radius_px=kislovodsk_support_radius_px,
            alignment_tolerance_deg=kislovodsk_alignment_tolerance_deg,
        )
        kislovodsk_is_filament = (
            int(
                kislovodsk_metrics["catalog_aligned_centerline_px"]
                >= kislovodsk_min_aligned_centerline_px
                and kislovodsk_metrics["catalog_aligned_centerline_fraction"]
                >= kislovodsk_min_aligned_centerline_fraction
            )
            if kislovodsk_available
            else np.nan
        )
        magfilo_metrics = compute_magfilo_centerline_metrics(
            component,
            magfilo_distance,
            magfilo_spine_mask,
            magfilo_polygon_mask,
            magfilo_spine_segments,
            case.magfilo_available,
            support_radius_px=magfilo_support_radius_px,
            polygon_tolerance_px=magfilo_polygon_tolerance_px,
            alignment_tolerance_deg=magfilo_alignment_tolerance_deg,
        )
        magfilo_label = (
            magfilo_is_filament(
                magfilo_metrics,
                magfilo_min_aligned_centerline_px,
                magfilo_min_aligned_centerline_fraction,
            )
            if case.magfilo_available
            else np.nan
        )
        centroid_row, centroid_column = ndimage.center_of_mass(component)
        records.append(
            {
                "frame_key": case.frame_key,
                "observation_dt": case.observation_dt,
                "mask_datetime": case.mask_datetime,
                "mask_time_offset_min": case.mask_time_offset_min,
                "mask_path": str(case.mask_path),
                "dec1_mask_source": case.dec1_mask_source,
                "iswat_fits_path": str(case.fits_path),
                "annotation_path": str(case.annotation_path),
                "component_id": component_id,
                "area_px": int(component.sum()),
                "centroid_row": float(centroid_row),
                "centroid_column": float(centroid_column),
                "kislovodsk_available": kislovodsk_available,
                "kislovodsk_datetime": (
                    kislovodsk_filaments["datetime"].iloc[0]
                    if kislovodsk_available
                    else pd.NaT
                ),
                "kislovodsk_segments": len(kislovodsk_filaments),
                "kislovodsk_segments_rasterized": kislovodsk_rasterized,
                "kislovodsk_is_filament": kislovodsk_is_filament,
                "magfilo_available": case.magfilo_available,
                "magfilo_observation_dt": case.magfilo_observation_dt,
                "magfilo_time_offset_hours": case.magfilo_time_offset_hours,
                "magfilo_annotation_instances": case.magfilo_annotation_instances,
                "magfilo_filament_annotations": case.magfilo_filament_annotations,
                "magfilo_is_filament": magfilo_label,
                **{
                    key.replace("catalog_", "kislovodsk_"): value
                    for key, value in kislovodsk_metrics.items()
                },
                **magfilo_metrics,
            }
        )

    annotation = plt.imread(case.annotation_path)
    figure, axes = plt.subplots(1, 3, figsize=(23, 8), layout="constrained")
    kislovodsk_axis, magfilo_axis, iswat_axis = axes
    kislovodsk_axis.imshow(aia193, cmap="sdoaia193", origin="upper")
    if kislovodsk_available:
        support_band = kislovodsk_distance <= kislovodsk_support_radius_px
        kislovodsk_axis.contour(
            support_band,
            levels=[0.5],
            colors=["lime"],
            linewidths=0.35,
            alpha=0.55,
        )
    for x1, y1, x2, y2 in kislovodsk_segments:
        kislovodsk_axis.plot([x1, x2], [y1, y2], color="lime", linewidth=1.5)

    magfilo_axis.imshow(aia193, cmap="sdoaia193", origin="upper")
    for magfilo_annotation in magfilo_projected:
        for polygon in magfilo_annotation["polygons"]:
            magfilo_axis.fill(
                polygon[:, 0],
                polygon[:, 1],
                facecolor="gold",
                edgecolor="gold",
                alpha=0.18,
                linewidth=0.7,
            )
        magfilo_axis.plot(
            magfilo_annotation["spine"][:, 0],
            magfilo_annotation["spine"][:, 1],
            color="white",
            linewidth=1.0,
        )

    for axis, label_key, prefix in [
        (kislovodsk_axis, "kislovodsk_is_filament", "K"),
        (magfilo_axis, "magfilo_is_filament", "M"),
    ]:
        for record in records:
            component = labels == record["component_id"]
            label_value = record[label_key]
            color = (
                "magenta"
                if label_value == 1
                else "cyan"
                if label_value == 0
                else "gold"
            )
            axis.contour(component, levels=[0.5], colors=[color], linewidths=1.35)
            label = (
                f"{record['component_id']} {prefix}F"
                if label_value == 1
                else f"{record['component_id']} {prefix}CH"
                if label_value == 0
                else f"{record['component_id']} ?"
            )
            axis.text(
                record["centroid_column"],
                record["centroid_row"],
                label,
                color="white",
                fontsize=7,
                ha="center",
                va="center",
                bbox={"facecolor": color, "alpha": 0.65, "pad": 1.1},
            )

    kislovodsk_axis.set_title(
        "Kislovodsk segments + independent K labels\n"
        "magenta = K filament; cyan = K CH; gold = no K record"
    )
    magfilo_text = (
        f"{case.magfilo_filament_annotations} annotations from "
        f"{case.magfilo_annotation_instances} independent instances; "
        f"GONG − AIA = {case.magfilo_time_offset_hours:+.1f} h"
        if case.magfilo_available
        else "no MAGFiLO observation within the configured time window"
    )
    magfilo_axis.set_title(
        "MAGFiLO polygons/spines + independent M labels\n" + magfilo_text
    )
    iswat_axis.imshow(annotation)
    iswat_axis.set_title("ISWAT expert annotation")
    for axis in axes:
        axis.set_axis_off()

    kislovodsk_filament_count = int(
        sum(record["kislovodsk_is_filament"] == 1 for record in records)
    )
    magfilo_filament_count = int(
        sum(record["magfilo_is_filament"] == 1 for record in records)
    )
    kislovodsk_text = (
        f"Kislovodsk {kislovodsk_filaments['datetime'].iloc[0]:%Y-%m-%d %H:%M}; "
        f"{len(kislovodsk_filaments)} segments ({kislovodsk_rasterized} on disk)"
        if kislovodsk_available
        else "no Kislovodsk catalog record in time window"
    )
    figure.suptitle(
        f"{case.frame_key}: dec1 mask {case.mask_datetime:%Y-%m-%d %H:%M:%S} "
        f"({case.mask_time_offset_min:+.1f} min); {component_count} regions; "
        f"K={kislovodsk_filament_count} and M={magfilo_filament_count} provisional "
        f"filaments\n{kislovodsk_text}",
        fontsize=11,
    )
    figure.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(figure)
    return records


def write_gallery_html(cases, components, gallery_path):
    summaries = (
        components.groupby("frame_key", dropna=False)
        .agg(
            regions=("component_id", "size"),
            kislovodsk_filaments=("kislovodsk_is_filament", "sum"),
            kislovodsk_segments=("kislovodsk_segments", "first"),
            magfilo_filaments=("magfilo_is_filament", "sum"),
            magfilo_available=("magfilo_available", "first"),
            magfilo_offset_hours=("magfilo_time_offset_hours", "first"),
            magfilo_annotations=("magfilo_filament_annotations", "first"),
            magfilo_instances=("magfilo_annotation_instances", "first"),
            mask_time_offset_min=("mask_time_offset_min", "first"),
        )
        .reset_index()
    )
    for column in ["kislovodsk_filaments", "magfilo_filaments"]:
        summaries[column] = summaries[column].fillna(0).astype(int)
    summaries = cases[["frame_key"]].merge(summaries, on="frame_key", how="left")
    cards = []
    for row in summaries.itertuples(index=False):
        image_name = f"cases/{row.frame_key}.png"
        magfilo_text = (
            f"MAGFiLO: {row.magfilo_filaments} M-filaments; "
            f"{row.magfilo_annotations} annotations / {row.magfilo_instances} "
            f"instances; {row.magfilo_offset_hours:+.1f} h."
            if row.magfilo_available
            else "MAGFiLO: no observation in the configured window."
        )
        cards.append(
            "<article>"
            f"<a href='{image_name}'><img src='{image_name}' loading='lazy'></a>"
            f"<h2>{html.escape(row.frame_key)}</h2>"
            f"<p>{row.regions} regions; Kislovodsk: {row.kislovodsk_filaments} "
            f"K-filaments / {row.kislovodsk_segments} segments; "
            f"{magfilo_text} dec1 offset {row.mask_time_offset_min:+.1f} min.</p>"
            "</article>"
        )
    gallery_path.write_text(
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>ISWAT filament catalog comparison</title>"
        "<style>body{font-family:system-ui;margin:24px;background:#181818;color:#eee}"
        "main{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:18px}"
        "article{background:#272727;padding:10px;border-radius:8px}"
        "img{width:100%;height:auto}h1{margin-bottom:4px}h2{font-size:1rem;margin:8px 0 2px}"
        "p{font-size:.86rem;margin:0;color:#ccc}</style></head><body>"
        "<h1>ISWAT / Kislovodsk / MAGFiLO region-label review</h1>"
        "<p>Panels use the same latest-dec1 regions. K and M labels are separate "
        "criteria; missing catalog coverage is never interpreted as CH. MAGFiLO "
        "combines all independent annotation instances of its matched GONG observation.</p>"
        f"<main>{''.join(cards)}</main></body></html>"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate an ISWAT Kislovodsk/MAGFiLO region-label review gallery."
    )
    parser.add_argument(
        "--iswat-root",
        type=Path,
        default=Path.home() / "Developer" / " Misc" / "COSPAR ISWAT CH dataset",
    )
    parser.add_argument(
        "--mask-root",
        type=Path,
        default=(
            ROOT_DIR
            / "Outputs"
            / "Filaments"
            / "ISWAT Kislovodsk Gallery"
            / "dec1 masks"
        ),
    )
    parser.add_argument(
        "--kislovodsk-catalog",
        type=Path,
        default=ROOT_DIR / "Data" / "Kislovodsk Filaments.csv",
    )
    parser.add_argument(
        "--magfilo-catalog",
        type=Path,
        default=ROOT_DIR / "Data" / "MAGFiLO" / "magfilo_2024_v1.0.json",
    )
    parser.add_argument(
        "--magfilo-fits-root",
        type=Path,
        default=ROOT_DIR / "Data" / "MAGFiLO" / "fits",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            ROOT_DIR
            / "Outputs"
            / "Filaments"
            / "ISWAT Catalog Comparison Gallery"
        ),
    )
    parser.add_argument("--kislovodsk-window-hours", type=float, default=12.0)
    parser.add_argument(
        "--kislovodsk-support-radius-px",
        type=float,
        default=CATALOG_SUPPORT_RADIUS_PX,
    )
    parser.add_argument(
        "--kislovodsk-alignment-tolerance-deg",
        type=float,
        default=CATALOG_ALIGNMENT_TOLERANCE_DEG,
    )
    parser.add_argument(
        "--kislovodsk-min-aligned-centerline-px",
        type=int,
        default=CATALOG_MIN_ALIGNED_CENTERLINE_PX,
    )
    parser.add_argument(
        "--kislovodsk-min-aligned-centerline-fraction",
        type=float,
        default=CATALOG_MIN_ALIGNED_CENTERLINE_FRACTION,
    )
    parser.add_argument("--magfilo-window-hours", type=float, default=12.0)
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
    args = parser.parse_args(argv)

    cases = match_dec1_masks(
        index_iswat_cases(args.iswat_root),
        index_dec1_masks(args.mask_root),
    )
    magfilo_catalog = load_magfilo(args.magfilo_catalog)
    cases = match_magfilo_observations(
        cases,
        magfilo_observations(magfilo_catalog),
        args.magfilo_window_hours,
        args.magfilo_fits_root,
    )
    kislovodsk_catalog = load_kislovodsk_catalog(args.kislovodsk_catalog)
    cases_dir = args.output_dir / "cases"
    cases_dir.mkdir(parents=True, exist_ok=True)
    all_records = []
    for case in tqdm(cases.itertuples(index=False), total=len(cases), desc="ISWAT gallery"):
        all_records.extend(
            render_case(
                case,
                kislovodsk_catalog,
                magfilo_catalog,
                cases_dir / f"{case.frame_key}.png",
                args.kislovodsk_window_hours,
                args.kislovodsk_support_radius_px,
                args.kislovodsk_alignment_tolerance_deg,
                args.kislovodsk_min_aligned_centerline_px,
                args.kislovodsk_min_aligned_centerline_fraction,
                args.magfilo_support_radius_px,
                args.magfilo_polygon_tolerance_px,
                args.magfilo_alignment_tolerance_deg,
                args.magfilo_min_aligned_centerline_px,
                args.magfilo_min_aligned_centerline_fraction,
            )
        )

    components = pd.DataFrame(all_records)
    components.to_parquet(args.output_dir / "ISWAT Components.parquet", index=False)
    components.to_csv(args.output_dir / "ISWAT Components.csv", index=False)
    cases.to_csv(args.output_dir / "ISWAT Cases.csv", index=False)
    (args.output_dir / "Metric Parameters.json").write_text(
        json.dumps(
            {
                "kislovodsk": {
                    "window_hours": args.kislovodsk_window_hours,
                    "support_radius_px": args.kislovodsk_support_radius_px,
                    "alignment_tolerance_deg": args.kislovodsk_alignment_tolerance_deg,
                    "min_aligned_centerline_px": args.kislovodsk_min_aligned_centerline_px,
                    "min_aligned_centerline_fraction": (
                        args.kislovodsk_min_aligned_centerline_fraction
                    ),
                },
                "magfilo": {
                    "window_hours": args.magfilo_window_hours,
                    "support_radius_px": args.magfilo_support_radius_px,
                    "polygon_tolerance_px": args.magfilo_polygon_tolerance_px,
                    "alignment_tolerance_deg": args.magfilo_alignment_tolerance_deg,
                    "min_aligned_centerline_px": args.magfilo_min_aligned_centerline_px,
                    "min_aligned_centerline_fraction": (
                        args.magfilo_min_aligned_centerline_fraction
                    ),
                    "multi_annotator_policy": "union of all independent instances",
                },
                "segmentation_source": "latest dec1 CH_MASK_FINAL masks",
            },
            indent=2,
        )
    )
    write_gallery_html(cases, components, args.output_dir / "index.html")

    iswat_2017 = components[
        pd.to_datetime(components["observation_dt"]).dt.year == 2017
    ]
    print(
        f"Saved {len(cases)} case images and {len(components)} region labels to "
        f"{args.output_dir}"
    )
    print(
        "2017 ISWAT regions: "
        f"{len(iswat_2017)}; K filaments: "
        f"{int(iswat_2017['kislovodsk_is_filament'].fillna(0).sum())}; "
        f"MAGFiLO filaments: "
        f"{int(iswat_2017['magfilo_is_filament'].fillna(0).sum())}; "
        f"MAGFiLO coverage: {int(iswat_2017['magfilo_available'].sum())} regions"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
