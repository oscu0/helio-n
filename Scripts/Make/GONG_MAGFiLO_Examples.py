#!/usr/bin/env python3
"""Project example MAGFiLO annotations from GONG H-alpha onto AIA 193."""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import sunpy.map
from matplotlib import pyplot as plt
from PIL import Image

ROOT_DIR = Path(__file__).resolve().parents[2]
sys.path.append(str(ROOT_DIR))

from Library.GONG import (
    magfilo_annotations,
    magfilo_category_names,
    magfilo_image_record,
    project_magfilo_annotations,
    rasterize_projected_annotations,
    load_magfilo,
)
from Library.IO import prepare_fits, prepare_mask


CATEGORY_COLORS = {
    "Left": "#ff4fb3",
    "Right": "#24d9ff",
    "Unidentifiable": "#ffd84a",
}

EXAMPLE_CASES = [
    {
        "image_id": "010402-20170313145450Bh",
        "gong_stem": "20170313145450Bh",
        "aia_stem": "AIA20170313_1454_0193",
        "mask_stem": "AIA20170313_145404_0193_CH_MASK_FINAL",
    },
    {
        "image_id": "020101-20170701235630Lh",
        "gong_stem": "20170701235630Lh",
        "aia_stem": "AIA20170701_2354_0193",
        "mask_stem": "AIA20170701_235404_0193_CH_MASK_FINAL",
    },
]


def normalized_aia_display(aia_map):
    data = np.flipud(np.asarray(aia_map.data, dtype=np.float32))
    low, high = np.nanpercentile(data, [1, 99.7])
    data = np.clip(data, low, high)
    return np.nan_to_num((data - low) / (high - low), nan=0.0)


def plot_original_annotations(axis, image, annotations, category_names):
    axis.imshow(image, origin="upper", cmap="gray")
    for annotation in annotations:
        color = CATEGORY_COLORS[category_names[annotation["category_id"]]]
        for flat_polygon in annotation["segmentation"]:
            polygon = np.asarray(flat_polygon).reshape(-1, 2)
            axis.fill(
                polygon[:, 0],
                polygon[:, 1],
                facecolor=color,
                edgecolor=color,
                alpha=0.22,
                linewidth=1.1,
            )
        spine = np.asarray(annotation["spine"]).reshape(-1, 2)
        axis.plot(
            spine[:, 0],
            spine[:, 1],
            color="white",
            linewidth=1.25,
        )


def plot_projected_annotations(axis, projected):
    for annotation in projected:
        color = CATEGORY_COLORS[annotation["category_name"]]
        for polygon in annotation["polygons"]:
            axis.fill(
                polygon[:, 0],
                polygon[:, 1],
                facecolor=color,
                edgecolor=color,
                alpha=0.24,
                linewidth=1.15,
            )
        axis.plot(
            annotation["spine"][:, 0],
            annotation["spine"][:, 1],
            color="white",
            linewidth=1.25,
        )


def render_case(catalog, data_root, output_dir, case):
    image_record = magfilo_image_record(catalog, case["image_id"])
    annotations = magfilo_annotations(catalog, case["image_id"])
    categories = magfilo_category_names(catalog)

    gong_jpg_path = data_root / f"{case['gong_stem']}.jpg"
    gong_fits_path = data_root / f"{case['gong_stem']}.fits.fz"
    aia_fits_path = data_root / f"{case['aia_stem']}.fits"
    mask_path = data_root / f"{case['mask_stem']}.png"

    gong_image = np.asarray(Image.open(gong_jpg_path).convert("L"))
    gong_map, gong_display = prepare_fits(gong_fits_path)
    aia_map = sunpy.map.Map(aia_fits_path)
    aia_display = normalized_aia_display(aia_map)
    ch_mask = prepare_mask(mask_path).astype(bool)
    assert gong_image.shape == (
        image_record["height"],
        image_record["width"],
    )
    assert aia_display.shape == ch_mask.shape

    projected = project_magfilo_annotations(
        catalog,
        case["image_id"],
        gong_map,
        aia_map,
    )
    source_projected = project_magfilo_annotations(
        catalog,
        case["image_id"],
        gong_map,
        gong_map,
        solar_rotate_to_target=False,
    )
    projected_mask = rasterize_projected_annotations(
        projected,
        aia_display.shape,
    )
    time_offset_seconds = (gong_map.date - aia_map.date).to_value("s")

    figure, axes = plt.subplots(
        1,
        4,
        figsize=(24, 6.4),
        layout="constrained",
    )
    original_axis, source_axis, projected_axis, regions_axis = axes

    plot_original_annotations(
        original_axis,
        gong_image,
        annotations,
        categories,
    )
    original_axis.set_title(
        f"{case['image_id']} — {len(annotations)} filaments\n"
        "MAGFiLO on exact GONG H-alpha\n"
        f"{gong_map.date.to_datetime():%Y-%m-%d %H:%M:%S} UTC",
        fontsize=11,
    )

    source_axis.imshow(gong_display, origin="upper", cmap="gray")
    plot_projected_annotations(source_axis, source_projected)
    source_axis.set_title(
        "Same annotations after COCO → FITS mapping\n"
        "on the exact source GONG FITS",
        fontsize=11,
    )

    projected_axis.imshow(
        aia_display,
        origin="upper",
        cmap="sdoaia193",
    )
    plot_projected_annotations(projected_axis, projected)
    projected_axis.set_title(
        "Same polygons + spines projected to AIA 193\n"
        f"{aia_map.date.to_datetime():%Y-%m-%d %H:%M:%S} UTC",
        fontsize=11,
    )

    regions_axis.imshow(
        aia_display,
        origin="upper",
        cmap="sdoaia193",
    )
    regions_axis.contour(
        ch_mask,
        levels=[0.5],
        colors=["lime"],
        linewidths=1.0,
    )
    plot_projected_annotations(regions_axis, projected)
    regions_axis.set_title(
        "Projected filaments + latest dec1 regions\n"
        f"GONG − AIA = {time_offset_seconds:+.0f} s",
        fontsize=11,
    )

    for axis in axes:
        axis.set_xlim(0, axis.images[0].get_array().shape[1] - 1)
        axis.set_ylim(axis.images[0].get_array().shape[0] - 1, 0)
        axis.set_aspect("equal")
        axis.set_xlabel("display x (solar east ←)")
        axis.set_ylabel("display y (solar north ↑)")

    handles = [
        mpatches.Patch(color=color, label=f"{name} chirality")
        for name, color in CATEGORY_COLORS.items()
    ]
    handles.append(
        mpatches.Patch(
            facecolor="none",
            edgecolor="lime",
            label="dec1 region",
        )
    )
    figure.legend(handles=handles, loc="outside lower center", ncol=4)
    output_path = output_dir / f"{case['gong_stem']}_projection.png"
    figure.savefig(output_path, dpi=160)
    plt.close(figure)

    finite_vertices = sum(
        np.isfinite(polygon).all(axis=1).sum()
        for annotation in projected
        for polygon in annotation["polygons"]
    )
    total_vertices = sum(
        len(polygon)
        for annotation in projected
        for polygon in annotation["polygons"]
    )
    return {
        "image_id": case["image_id"],
        "gong_time": gong_map.date.isot,
        "aia_time": aia_map.date.isot,
        "time_offset_seconds": float(time_offset_seconds),
        "annotation_count": len(annotations),
        "projected_mask_pixels": int(projected_mask.sum()),
        "finite_projected_vertices": int(finite_vertices),
        "total_projected_vertices": int(total_vertices),
        "output_path": str(output_path),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--catalog",
        type=Path,
        default=ROOT_DIR / "Data" / "MAGFiLO" / "magfilo_2024_v1.0.json",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=ROOT_DIR / "Data" / "MAGFiLO",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=(
            ROOT_DIR
            / "Outputs"
            / "Filaments"
            / "GONG_MAGFiLO_Examples"
        ),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    catalog = load_magfilo(args.catalog)
    records = [
        render_case(
            catalog,
            args.data_root,
            args.output_dir,
            case,
        )
        for case in EXAMPLE_CASES
    ]
    summary_path = args.output_dir / "examples.json"
    summary_path.write_text(json.dumps(records, indent=2) + "\n")
    print(pd.DataFrame(records).to_string(index=False))


if __name__ == "__main__":
    main()
