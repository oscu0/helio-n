"""Collect per-pair footprint scores and yearly joint coverage statistics."""

from itertools import product

import numpy as np
import pandas as pd
from scipy import ndimage
import sunpy.map

from Library.Filament_Footprints import (
    KISLOVODSK_FULL_WIDTHS_MM,
    footprint_scores,
    kislovodsk_footprint_labels,
)
from Library.GONG import (
    load_magfilo,
    magfilo_ephemeris_map,
    project_magfilo_observation,
    rasterize_projected_annotations,
)
from Library.IO import prepare_mask


CH_COVERAGE_THRESHOLDS = (0.01, 0.02, 0.05, 0.10, 0.20, 0.50)
FOOTPRINT_CONTAINMENT_THRESHOLDS = (0.10, 0.20, 0.30, 0.50, 0.70, 0.90)
# Full width (Mm), minimum CH coverage, minimum footprint containment.
KISLOVODSK_REVIEW_SCREENS = {
    (8., 0.02, 0.30): "partial_association",
    (12., 0.20, 0.50): "substantial_overlap",
    (16., 0.20, 0.50): "substantial_overlap",
}

STATS_MAGFILO_CATALOG = None


def initialize_stats_worker(magfilo_catalog_path):
    global STATS_MAGFILO_CATALOG
    STATS_MAGFILO_CATALOG = (
        load_magfilo(magfilo_catalog_path) if magfilo_catalog_path is not None else None
    )


def collect_stats_frame(task):
    frame_key, fits_path, mask_path, kislovodsk, magfilo, frame = task
    frame = {"frame_key": frame_key, "fits_path": fits_path, "mask_path": mask_path, **frame}
    frame["components"] = 0
    if frame["input_status"] != "ready":
        return frame, []

    mask = prepare_mask(mask_path).astype(bool)
    labels, frame["components"] = ndimage.label(mask, structure=np.ones((3, 3)))
    if frame["kislovodsk_status"] != "matched" and frame["magfilo_status"] != "matched":
        return frame, []
    aia = sunpy.map.Map(fits_path)
    assert aia.data.shape == labels.shape, f"{frame_key}: AIA and mask shapes differ"
    frame["aia_date_obs_utc"] = aia.date.isot
    component_areas = np.bincount(labels.ravel())
    records = []

    if frame["kislovodsk_status"] == "matched":
        regions_by_width = kislovodsk_footprint_labels(aia, kislovodsk, KISLOVODSK_FULL_WIDTHS_MM)
        for width, (regions, count) in regions_by_width.items():
            frame[f"kislovodsk_regions_{width:g}_mm"] = count
            records.extend(footprint_scores(
                labels, component_areas, regions, frame_key, "kislovodsk", width_mm=width
            ))
        frame["kislovodsk_status"] = (
            "covered" if any(count for _, count in regions_by_width.values()) else "no_projected_footprints"
        )

    if frame["magfilo_status"] == "matched":
        if magfilo["fits_path"] is None:
            gong = magfilo_ephemeris_map(STATS_MAGFILO_CATALOG, magfilo)
        else:
            gong = sunpy.map.Map(magfilo["fits_path"])
        projected = project_magfilo_observation(STATS_MAGFILO_CATALOG, magfilo, gong, aia)
        projected_count = 0
        for annotation in projected:
            # A partly invalid outline would connect across an unobserved gap.
            if not all(np.isfinite(polygon).all() for polygon in annotation["polygons"]):
                continue
            footprint = rasterize_projected_annotations([annotation], labels.shape)
            if footprint.any():
                projected_count += 1
                records.extend(footprint_scores(
                    labels, component_areas, footprint, frame_key, "magfilo",
                    annotation_id=annotation["annotation_id"],
                ))
        frame["magfilo_projected_annotations"] = projected_count
        frame["magfilo_status"] = "covered" if projected_count or not projected else "no_projected_annotations"

    return frame, records


def yearly_catalog_stats(frames, scores, years, catalogs):
    records = []
    frame_year = frames["frame_key"].str[:4].astype(int)
    score_year = scores["frame_key"].astype(str).str[:4].astype(int)
    component_keys = ["frame_key", "component_id"]
    for year in years:
        indexed = frames.loc[frame_year.eq(year)]
        ready = indexed.loc[indexed["input_status"].eq("ready")]
        for catalog in catalogs:
            covered = ready.loc[ready[f"{catalog}_status"].eq("covered")]
            denominator = int(covered["components"].sum())
            frames_with_ephemeris = 0
            components_with_ephemeris = 0
            if catalog == "magfilo":
                ephemeris = covered["magfilo_geometry"].eq("ephemeris")
                frames_with_ephemeris = int(ephemeris.sum())
                components_with_ephemeris = int(covered.loc[ephemeris, "components"].sum())

            candidates = scores.loc[score_year.eq(year) & scores["catalog"].eq(catalog)]
            widths = KISLOVODSK_FULL_WIDTHS_MM if catalog == "kislovodsk" else (None,)
            for width in widths:
                subset = candidates.loc[
                    candidates["width_mm"].isna() if width is None else candidates["width_mm"].eq(width)
                ]
                for ch_min, footprint_min in product(CH_COVERAGE_THRESHOLDS, FOOTPRINT_CONTAINMENT_THRESHOLDS):
                    # Both fractions must pass for one pair before deduplicating the CH.
                    passing = subset["ch_covered_fraction"].ge(ch_min) & subset["footprint_inside_ch_fraction"].ge(footprint_min)
                    matches = len(subset.loc[passing, component_keys].drop_duplicates())
                    records.append({
                        "year": year, "catalog": catalog, "width_mm": width,
                        "min_ch_covered_fraction": ch_min,
                        "min_footprint_inside_ch_fraction": footprint_min,
                        "review_screen": KISLOVODSK_REVIEW_SCREENS.get((width, ch_min, footprint_min)),
                        "frames_indexed": len(indexed), "frames_with_inputs": len(ready),
                        "components_with_inputs": int(ready["components"].sum()),
                        "frames_covered": len(covered), "components_covered": denominator,
                        "frames_covered_with_ephemeris": frames_with_ephemeris,
                        "components_covered_with_ephemeris": components_with_ephemeris,
                        "matches": matches if denominator else None,
                        "match_fraction": matches / denominator if denominator else None,
                    })
    return pd.DataFrame(records).astype({"matches": "Int64"})
