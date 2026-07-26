import json
from pathlib import Path

import astropy.units as u
import numpy as np
import pandas as pd
from skimage.draw import polygon as draw_polygon
from skimage.draw import line
from sunpy.physics.differential_rotation import solar_rotate_coordinate


MAGFILO_REQUIRED_KEYS = {
    "info",
    "licenses",
    "images",
    "annotations",
    "categories",
}


def load_magfilo(path):
    path = Path(path)
    with path.open() as catalog_file:
        catalog = json.load(catalog_file)

    assert MAGFILO_REQUIRED_KEYS <= catalog.keys(), (
        f"{path} is missing MAGFiLO/COCO collections: "
        f"{sorted(MAGFILO_REQUIRED_KEYS - catalog.keys())}"
    )

    image_ids = {image["id"] for image in catalog["images"]}
    assert len(image_ids) == len(catalog["images"]), "MAGFiLO image IDs are not unique"
    assert all(
        annotation["image_id"] in image_ids
        for annotation in catalog["annotations"]
    ), "MAGFiLO contains annotations with unknown image IDs"

    return catalog


def magfilo_image_record(catalog, image_id):
    records = [image for image in catalog["images"] if image["id"] == image_id]
    assert len(records) == 1, (
        f"Expected one MAGFiLO image record for {image_id}, found {len(records)}"
    )
    return records[0]


def magfilo_annotations(catalog, image_id):
    annotations = [
        annotation
        for annotation in catalog["annotations"]
        if annotation["image_id"] == image_id
    ]
    assert annotations, f"No MAGFiLO annotations for {image_id}"
    return annotations


def magfilo_observations(catalog):
    """Index MAGFiLO's physical GONG observations, not annotation instances."""
    images = pd.DataFrame(catalog["images"])
    images["date_captured"] = pd.to_datetime(images["date_captured"])
    annotations_per_image = pd.Series(
        [annotation["image_id"] for annotation in catalog["annotations"]]
    ).value_counts()
    rows = []
    for url, records in images.groupby("url", sort=False):
        observation_time = records["date_captured"].iloc[0]
        assert (records["date_captured"] == observation_time).all(), (
            f"MAGFiLO URL has inconsistent timestamps: {url}"
        )
        image_ids = records["id"].tolist()
        rows.append(
            {
                "url": url,
                "observation_dt": observation_time,
                "image_ids": image_ids,
                "annotation_instances": len(image_ids),
                "filament_annotations": int(
                    sum(annotations_per_image.get(image_id, 0) for image_id in image_ids)
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("observation_dt").reset_index(drop=True)


def select_magfilo_observation(observations, observation_dt, window_hours):
    assert window_hours > 0.0
    offset = (observations["observation_dt"] - observation_dt).abs()
    nearest_index = offset.idxmin()
    nearest = observations.loc[nearest_index].copy()
    nearest["time_offset_hours"] = (
        nearest["observation_dt"] - observation_dt
    ).total_seconds() / 3600.0
    return nearest if abs(nearest["time_offset_hours"]) <= window_hours else None


def magfilo_category_names(catalog):
    return {
        category["id"]: category["name"]
        for category in catalog["categories"]
    }


def magfilo_fits_url(image_record):
    url = image_record["url"]
    assert "/hag/" in url and url.endswith(".jpg"), (
        f"Unexpected GONG quicklook URL: {url}"
    )
    return url.replace("/hag/", "/haf/").removesuffix(".jpg") + ".fits.fz"


def coco_to_fits_pixels(points, image_height):
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    fits_pixels = points.copy()
    fits_pixels[:, 1] = image_height - 1 - points[:, 1]
    return fits_pixels


def project_magfilo_points(
    points,
    image_height,
    gong_map,
    target_map,
    display_coordinates=True,
    solar_rotate_to_target=True,
):
    source_pixels = coco_to_fits_pixels(points, image_height)
    world = gong_map.pixel_to_world(
        source_pixels[:, 0] * u.pixel,
        source_pixels[:, 1] * u.pixel,
    )
    if solar_rotate_to_target:
        world = solar_rotate_coordinate(
            world,
            observer=target_map.observer_coordinate,
        )
    target_pixels = target_map.world_to_pixel(world)
    projected = np.column_stack(
        [
            target_pixels.x.to_value(u.pixel),
            target_pixels.y.to_value(u.pixel),
        ]
    )
    if display_coordinates:
        projected[:, 1] = target_map.data.shape[0] - 1 - projected[:, 1]
    return projected


def project_magfilo_annotations(
    catalog,
    image_id,
    gong_map,
    target_map,
    display_coordinates=True,
    solar_rotate_to_target=True,
):
    image = magfilo_image_record(catalog, image_id)
    category_names = magfilo_category_names(catalog)
    projected = []

    for annotation in magfilo_annotations(catalog, image_id):
        polygons = [
            project_magfilo_points(
                polygon,
                image["height"],
                gong_map,
                target_map,
                display_coordinates=display_coordinates,
                solar_rotate_to_target=solar_rotate_to_target,
            )
            for polygon in annotation["segmentation"]
        ]
        spine = project_magfilo_points(
            annotation["spine"],
            image["height"],
            gong_map,
            target_map,
            display_coordinates=display_coordinates,
            solar_rotate_to_target=solar_rotate_to_target,
        )
        projected.append(
            {
                "annotation_id": annotation["id"],
                "category_id": annotation["category_id"],
                "category_name": category_names[annotation["category_id"]],
                "polygons": polygons,
                "spine": spine,
            }
        )

    return projected


def project_magfilo_observation(
    catalog,
    observation,
    gong_map,
    target_map,
    display_coordinates=True,
):
    """Project all independent annotations of one physical GONG observation."""
    projected = []
    for image_id in observation["image_ids"]:
        projected.extend(
            project_magfilo_annotations(
                catalog,
                image_id,
                gong_map,
                target_map,
                display_coordinates=display_coordinates,
            )
        )
    return projected


def rasterize_projected_annotations(projected_annotations, shape):
    mask = np.zeros(shape, dtype=bool)
    for annotation in projected_annotations:
        for polygon in annotation["polygons"]:
            finite = np.isfinite(polygon).all(axis=1)
            if finite.sum() < 3:
                continue
            rows, columns = draw_polygon(
                polygon[finite, 1],
                polygon[finite, 0],
                shape=shape,
            )
            mask[rows, columns] = True
    return mask


def projected_spine_segments(projected_annotations):
    segments = []
    for annotation in projected_annotations:
        spine = annotation["spine"]
        for start, end in zip(spine[:-1], spine[1:]):
            if np.isfinite(start).all() and np.isfinite(end).all():
                segments.append((*start, *end))
    return np.asarray(segments, dtype=np.float32).reshape(-1, 4)


def rasterize_projected_spines(projected_annotations, shape):
    mask = np.zeros(shape, dtype=bool)
    for x1, y1, x2, y2 in projected_spine_segments(projected_annotations):
        rows, columns = line(
            int(np.rint(y1)),
            int(np.rint(x1)),
            int(np.rint(y2)),
            int(np.rint(x2)),
        )
        in_bounds = (
            (rows >= 0)
            & (rows < shape[0])
            & (columns >= 0)
            & (columns < shape[1])
        )
        mask[rows[in_bounds], columns[in_bounds]] = True
    return mask
