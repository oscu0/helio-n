"""Component measurements and annual threshold sweeps for the filament catalogs."""

from itertools import product

import astropy.units as u
import numpy as np
import pandas as pd
from scipy import ndimage
from skimage.morphology import skeletonize
import sunpy.map

from Library.Filaments import project_catalog_segments, sample_contact_curve
from Library.GONG import (
    load_magfilo,
    project_magfilo_observation,
    projected_spine_segments,
    rasterize_projected_annotations,
)
from Library.IO import prepare_mask


KISLOVODSK_RADII_ARCSEC = (12.0, 24.0, 36.0)
OVERLAP_THRESHOLDS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
ANGLE_THRESHOLDS_DEG = (10.0, 20.0, 30.0)
REFERENCE_OVERLAP = 0.8
REFERENCE_ANGLE_DEG = 20.0
SCORE_COLUMNS = (
    "frame_key", "component_id", "catalog", "annotation_id", "radius_arcsec",
    "component_pixels", "skeleton_length_arcsec", "overlap_fraction",
    "supported_spine_arcsec", "compared_spine_arcsec", "angle_q80_deg",
    "longest_aligned_run_arcsec_at_20deg",
)

STATS_MAGFILO_CATALOG = None


def initialize_stats_worker(magfilo_catalog_path):
    global STATS_MAGFILO_CATALOG
    STATS_MAGFILO_CATALOG = (
        load_magfilo(magfilo_catalog_path) if magfilo_catalog_path is not None else None
    )


def sample_component_spine(component, slices, scale):
    """Trace each skeleton branch once and sample it in image-plane arcsec."""
    pixels = set(zip(*np.nonzero(skeletonize(component))))
    neighbours = {}
    for y, x in pixels:
        neighbours[y, x] = []
        for dy, dx in product((-1, 0, 1), repeat=2):
            neighbour = y + dy, x + dx
            if neighbour == (y, x) or neighbour not in pixels:
                continue
            # Suppress diagonal shortcuts across an existing right-angle turn.
            if dx and dy and ((y + dy, x) in pixels or (y, x + dx) in pixels):
                continue
            neighbours[y, x].append(neighbour)

    pending = {frozenset((p, q)) for p, links in neighbours.items() for q in links}
    offset = np.array([slices[1].start, slices[0].start])
    branches = []
    # Begin at endpoints and junctions. Degree-two pixels then pick up loops.
    for start in sorted(pixels, key=lambda p: (len(neighbours[p]) == 2, p)):
        for other in neighbours[start]:
            edge = frozenset((start, other))
            if edge not in pending:
                continue
            pending.remove(edge)
            branch = [start, other]
            while branch[-1] != start and len(neighbours[branch[-1]]) == 2:
                previous, current = branch[-2:]
                following, = (p for p in neighbours[current] if p != previous)
                pending.remove(frozenset((current, following)))
                branch.append(following)
            points = (np.asarray(branch)[:, ::-1] + offset) * scale
            closed = branch[-1] == start
            branches.append((*sample_contact_curve(points, closed=closed), closed))
    return branches


def segment_distances_and_angles(points, tangents, segments, scale):
    ends = segments.reshape(-1, 2, 2) * scale
    starts = ends[:, 0]
    vectors = ends[:, 1] - starts
    lengths = np.linalg.norm(vectors, axis=1)
    valid = np.isfinite(lengths) & (lengths > 0)
    if not valid.any():
        return np.empty((len(points), 0)), np.empty((len(points), 0))
    directions = vectors[valid] / lengths[valid, None]
    offsets = points[:, None] - starts[None, valid]
    along = np.clip(np.sum(offsets * directions, axis=2), 0, lengths[valid])
    distances = np.linalg.norm(offsets - along[..., None] * directions, axis=2)
    angles = np.degrees(np.arccos(np.clip(np.abs(tangents @ directions.T), 0, 1)))
    angles[~np.any(tangents, axis=1)] = np.inf
    return distances, angles


def score_spine(branches, segments, scale, *, radii=(), footprint=None):
    """Score proximity and orientation on the same segments or MAGFiLO instance."""
    scores = {radius: ([], [], [], 0.0) for radius in radii or (None,)}
    total_length = sum(len(points) * step for points, _, step, _ in branches)
    for points, tangents, step, closed in branches:
        distances, angles = segment_distances_and_angles(points, tangents, segments, scale)
        if footprint is not None:
            x, y = np.rint(points / scale).astype(int).T
            in_polygon = footprint[y, x]
        for radius, (covered_lengths, angle_values, weights, longest) in scores.items():
            if radius is not None:
                nearby = distances <= radius
                covered = nearby.any(axis=1)
                difference = np.where(nearby, angles, np.inf).min(axis=1, initial=np.inf)
            else:
                covered = in_polygon
                difference = (
                    angles[np.arange(len(points)), distances.argmin(axis=1)]
                    if distances.shape[1]
                    else np.full(len(points), np.inf)
                )
            compared = covered & np.isfinite(difference)
            covered_lengths.append(int(covered.sum()) * step)
            angle_values.extend(difference[compared])
            weights.extend(np.full(compared.sum(), step))
            aligned = compared & (difference <= REFERENCE_ANGLE_DEG)
            if closed and not aligned.all():
                aligned = np.roll(aligned, -int(np.argmin(aligned)))
            edges = np.flatnonzero(np.diff(np.r_[False, aligned, False]))
            if edges.size:
                longest = max(longest, int(np.max(edges[1::2] - edges[::2])) * step)
            scores[radius] = covered_lengths, angle_values, weights, longest

    rows = []
    for radius, (covered_lengths, angle_values, weights, longest) in scores.items():
        q80 = np.nan
        if angle_values:
            order = np.argsort(angle_values)
            cumulative = np.cumsum(np.asarray(weights)[order])
            q80 = angle_values[order[np.searchsorted(cumulative, 0.8 * cumulative[-1])]]
        rows.append({
            "radius_arcsec": radius,
            "skeleton_length_arcsec": total_length,
            "overlap_fraction": min(1.0, sum(covered_lengths) / total_length) if total_length else np.nan,
            "supported_spine_arcsec": sum(covered_lengths),
            "compared_spine_arcsec": sum(weights),
            "angle_q80_deg": q80,
            "longest_aligned_run_arcsec_at_20deg": longest,
        })
    return rows


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
    scale = np.abs(u.Quantity(aia.scale).to_value(u.arcsec / u.pix))
    frame["aia_date_obs_utc"] = aia.date.isot
    k_segments = project_catalog_segments(aia, kislovodsk)
    if frame["kislovodsk_status"] == "matched":
        if np.any(k_segments[:, :2] != k_segments[:, 2:]):
            frame["kislovodsk_status"] = "covered"
        else:
            frame["kislovodsk_status"] = "no_projected_segments"

    annotations = []
    if frame["magfilo_status"] == "matched":
        gong = sunpy.map.Map(magfilo["fits_path"])
        projected = project_magfilo_observation(STATS_MAGFILO_CATALOG, magfilo, gong, aia)
        for annotation in projected:
            # A partly invalid outline would connect across an unobserved gap.
            if not all(np.isfinite(polygon).all() for polygon in annotation["polygons"]):
                continue
            footprint = rasterize_projected_annotations([annotation], labels.shape)
            if footprint.any():
                annotations.append((annotation["annotation_id"], footprint, projected_spine_segments([annotation])))
        frame["magfilo_projected_annotations"] = len(annotations)
        frame["magfilo_status"] = "covered" if annotations or not projected else "no_projected_annotations"

    records = []
    for component_id, slices in enumerate(ndimage.find_objects(labels), 1):
        component = labels[slices] == component_id
        area = int(component.sum())
        branches = sample_component_spine(component, slices, scale)
        keys = {"frame_key": frame_key, "component_id": component_id, "component_pixels": area}
        if frame["kislovodsk_status"] == "covered":
            for score in score_spine(branches, k_segments, scale, radii=KISLOVODSK_RADII_ARCSEC):
                records.append({**keys, **score, "catalog": "kislovodsk", "annotation_id": None})
        if frame["magfilo_status"] == "covered":
            has_overlap = False
            for annotation_id, footprint, segments in annotations:
                overlap = int((component & footprint[slices]).sum())
                if overlap:
                    has_overlap = True
                    score, = score_spine(branches, segments, scale, footprint=footprint)
                    records.append({
                        **keys, **score, "catalog": "magfilo", "annotation_id": annotation_id,
                        "overlap_fraction": overlap / area,
                    })
            if not has_overlap:
                records.append({
                    **keys, "catalog": "magfilo", "annotation_id": None, "radius_arcsec": None,
                    "skeleton_length_arcsec": sum(len(points) * step for points, _, step, _ in branches),
                    "overlap_fraction": 0.0, "supported_spine_arcsec": 0.0,
                    "compared_spine_arcsec": 0.0, "angle_q80_deg": np.nan,
                    "longest_aligned_run_arcsec_at_20deg": 0.0,
                })
    return frame, records


def yearly_catalog_stats(frames, scores, years, catalogs):
    records = []
    frame_year = frames["frame_key"].str[:4].astype(int)
    score_year = scores["frame_key"].astype(str).str[:4].astype(int)
    for year in years:
        indexed = frames.loc[frame_year.eq(year)]
        ready = indexed.loc[indexed["input_status"].eq("ready")]
        for catalog in catalogs:
            covered = ready.loc[ready[f"{catalog}_status"].eq("covered")]
            candidates = scores.loc[score_year.eq(year) & scores["catalog"].eq(catalog)]
            denominator = int(covered["components"].sum())
            for radius in KISLOVODSK_RADII_ARCSEC if catalog == "kislovodsk" else (None,):
                subset = candidates.loc[
                    candidates["radius_arcsec"].isna() if radius is None
                    else candidates["radius_arcsec"].eq(radius)
                ]
                component_keys = ["frame_key", "component_id"]
                finite_angles = subset["angle_q80_deg"].notna()
                n_angles = len(subset.loc[finite_angles, component_keys].drop_duplicates())
                counts = {}
                for overlap, angle in product(OVERLAP_THRESHOLDS, ANGLE_THRESHOLDS_DEG):
                    matches = subset["overlap_fraction"].ge(overlap) & subset["angle_q80_deg"].le(angle)
                    counts[overlap, angle] = len(subset.loc[matches, component_keys].drop_duplicates())
                reference = counts[REFERENCE_OVERLAP, REFERENCE_ANGLE_DEG]
                for (overlap, angle), count in counts.items():
                    records.append({
                        "year": year, "catalog": catalog, "radius_arcsec": radius,
                        "min_overlap_fraction": overlap, "max_angle_q80_deg": angle,
                        "frames_indexed": len(indexed), "frames_with_inputs": len(ready),
                        "components_with_inputs": int(ready["components"].sum()),
                        "frames_covered": len(covered), "components_covered": denominator,
                        "components_with_angle": n_angles,
                        "matches": count if denominator else None,
                        "match_fraction": count / denominator if denominator else None,
                        "delta_matches_from_reference": count - reference if denominator else None,
                        "delta_percentage_points_from_reference": (
                            100 * (count - reference) / denominator if denominator else None
                        ),
                    })
    return pd.DataFrame(records).astype({
        "matches": "Int64",
        "delta_matches_from_reference": "Int64",
    })
