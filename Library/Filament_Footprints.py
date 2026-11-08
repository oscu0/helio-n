"""Project filament footprints and measure their overlap with CH components."""

import astropy.units as u
from astropy.coordinates import SkyCoord
import numpy as np
from scipy.cluster.hierarchy import DisjointSet
from skimage.draw import polygon as rasterize_polygon
from sunpy.coordinates import HeliographicCarrington


KISLOVODSK_FULL_WIDTHS_MM = (4., 8., 12., 16.)
FOOTPRINT_SCORE_COLUMNS = (
    "frame_key", "component_id", "catalog", "width_mm", "footprint_id",
    "component_pixels", "footprint_pixels", "intersection_pixels",
    "iou", "ch_covered_fraction", "footprint_inside_ch_fraction",
)


def visible_polygon(vertices, observer, horizon):
    """Clip a sampled surface outline to the near side before projecting it."""
    distance = vertices @ observer - horizon
    if not np.any(distance >= 0):
        return vertices[:0]
    if np.all(distance >= 0):
        return vertices

    outline = []
    for i, point in enumerate(vertices):
        previous = vertices[i - 1]
        before, after = distance[i - 1], distance[i]
        if (before >= 0) != (after >= 0):
            fraction = before / (before - after)
            crossing = previous + fraction * (point - previous)
            crossing /= np.linalg.norm(crossing)
            # Put the interpolated intersection on the apparent limb.
            radial = crossing - (crossing @ observer) * observer
            outline.append(horizon * observer + np.sqrt(1 - horizon**2) * radial / np.linalg.norm(radial))
        if after >= 0:
            outline.append(point)
    return np.asarray(outline)


def kislovodsk_footprint_labels(aia, filaments, widths_mm):
    """Project surface ellipses with segment endpoints as foci in the midpoint tangent plane."""
    frame = HeliographicCarrington(observer=aia.observer_coordinate, obstime=aia.date)
    radius = aia.rsun_meters.to_value(u.Mm)
    observer = aia.observer_coordinate.transform_to(frame).cartesian.xyz.to_value(u.Mm)
    distance_to_observer = np.linalg.norm(observer)
    horizon = radius / distance_to_observer
    observer = observer / distance_to_observer
    shape = aia.data.shape

    coords = filaments[["lon1", "lat1", "lon2", "lat2"]].to_numpy(dtype=float).reshape(-1, 2, 2)
    coords = coords[np.isfinite(coords).all(axis=(1, 2))]
    lon, lat = np.deg2rad(coords).transpose(2, 0, 1)
    directions = np.stack((np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)), axis=-1)
    start, end = directions.transpose(1, 0, 2)
    mid = start + end
    along = end - start
    mid_length = np.linalg.norm(mid, axis=1)
    along_length = np.linalg.norm(along, axis=1)
    valid = (mid_length > 1e-12) & (along_length > 1e-12)
    midpoint = mid[valid] / mid_length[valid, None]
    major_axis = along[valid] / along_length[valid, None]
    minor_axis = np.cross(midpoint, major_axis)
    focal_distance = radius * np.arctan2(along_length[valid], mid_length[valid])

    azimuth = np.arange(96) * (2 * np.pi / 96)
    owners = {width: np.zeros(shape, dtype=np.int32) for width in widths_mm}
    groups = {width: DisjointSet() for width in widths_mm}
    polygons = []
    for width in widths_mm:
        half_width = width / 2
        semi_major = np.hypot(focal_distance, half_width)
        offsets = (
            semi_major[:, None, None] * np.cos(azimuth)[None, :, None] * major_axis[:, None]
            + half_width * np.sin(azimuth)[None, :, None] * minor_axis[:, None]
        )
        distance = np.linalg.norm(offsets, axis=2, keepdims=True)
        vertices = (
            np.cos(distance / radius) * midpoint[:, None]
            + np.sin(distance / radius) * offsets / distance
        )
        for ellipse in vertices:
            outline = visible_polygon(ellipse, observer, horizon)
            if len(outline) >= 3:
                polygons.append((width, outline))

    if polygons:
        vertices = np.concatenate([outline for _, outline in polygons])
        world = SkyCoord(*(radius * vertices.T * u.Mm), representation_type="cartesian", frame=frame)
        x, y = aia.world_to_pixel(world)
        x = x.to_value(u.pix)
        y = shape[0] - 1 - y.to_value(u.pix)
        offset = 0
        for width, outline in polygons:
            stop = offset + len(outline)
            if np.isfinite(x[offset:stop]).all() and np.isfinite(y[offset:stop]).all():
                rows, cols = rasterize_polygon(y[offset:stop], x[offset:stop], shape)
                if rows.size:
                    # Keep one identity per ellipse even if the rasterized
                    # footprint separates into islands near the limb.
                    labels = owners[width]
                    connected = groups[width]
                    label = len(connected) + 1
                    connected.add(label)
                    for other in np.unique(labels[rows, cols]):
                        if other:
                            connected.merge(label, int(other))
                    labels[rows, cols] = label
            offset = stop

    footprints = {}
    for width, labels in owners.items():
        connected = groups[width].subsets()
        relabel = np.zeros(len(groups[width]) + 1, dtype=np.int32)
        for region, members in enumerate(connected, 1):
            relabel[list(members)] = region
        footprints[width] = relabel[labels], len(connected)
    return footprints


def footprint_scores(labels, component_areas, footprints, frame_key, catalog, *, width_mm=None, annotation_id=None):
    """Emit one row per intersecting CH and footprint, with all three area ratios."""
    overlap = (labels > 0) & (footprints > 0)
    if not overlap.any():
        return []
    pairs, intersections = np.unique(
        np.column_stack((labels[overlap], footprints[overlap])), axis=0, return_counts=True
    )
    footprint_areas = np.bincount(footprints.ravel())
    rows = []
    for (component, footprint), intersection in zip(pairs, intersections):
        ch_area = int(component_areas[component])
        filament_area = int(footprint_areas[footprint])
        intersection = int(intersection)
        rows.append({
            "frame_key": frame_key, "component_id": int(component), "catalog": catalog,
            "width_mm": width_mm, "footprint_id": str(footprint) if annotation_id is None else str(annotation_id),
            "component_pixels": ch_area, "footprint_pixels": filament_area, "intersection_pixels": intersection,
            "iou": intersection / (ch_area + filament_area - intersection),
            "ch_covered_fraction": intersection / ch_area,
            "footprint_inside_ch_fraction": intersection / filament_area,
        })
    return rows
