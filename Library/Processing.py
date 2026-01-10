import json
import os

import numpy as np

import PIL

from skimage.morphology import (
    closing,
    disk,
    remove_small_objects,
    remove_small_holes,
)
from skimage.transform import resize

from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = str(MODULE_DIR.parent) + "/"

from Library.Config import *
from Library.IO import prepare_fits, pmap_path, prepare_pmap, resize_for_model

INFER_TRAINING = False


def fits_to_pmap(model, img2d):
    img_size = model.architecture["img_size"]
    img = np.asarray(img2d, dtype=np.float32)

    if img.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {img.shape}")

    img = resize_for_model(img, img_size)

    x = img[np.newaxis, ..., np.newaxis]  # (1, H, W, 1)

    if INFER_TRAINING:
        # Force training=True to diagnose BatchNorm moving-average issues.
        if hasattr(model, "model"):
            prob = model.model(x, training=True).numpy()[0, ..., 0]
        else:
            prob = model(x, training=True).numpy()[0, ..., 0]
    else:
        # Prefer the compiled inference path on the wrapped model, fall back as needed.
        if hasattr(model, "compiled_infer"):
            prob = model.compiled_infer(x)[0, ..., 0]
        elif hasattr(model, "predict"):
            prob = model.predict(x)[0, ..., 0]
        else:
            prob = model.model.predict(x, verbose=0)[0, ..., 0]

    return prob


def get_postprocessing_params(postprocessing):
    config = json.load(
        open(PROJECT_ROOT + "Config/Postprocessing/" + postprocessing + ".json")
    )
    return config


def pmap_to_mask(
    pmap, smoothing_params=get_postprocessing_params("P0"), save_path=None
):
    """
    Convert a probability map into a cleaned binary mask (0/1 float32),
    matching the semantics of prepare_mask(preserve_255=False).

    If save=True, also save a 0–255 uint8 PNG to save_path.
    """

    # --- threshold to get binary CH mask ---
    mask = (pmap > smoothing_params["threshold"])

    # --- morphological smoothing ---
    # closing supports bool; avoid float to keep skimage happy
    mask = closing(mask, disk(smoothing_params["closing_radius"]))

    if smoothing_params["min_size"] > 0:
        mask = remove_small_objects(mask, max_size=smoothing_params["min_size"])

    if smoothing_params["hole_size"] > 0:
        mask = remove_small_holes(mask, max_size=smoothing_params["hole_size"])

    mask_u8 = mask.astype(np.uint8) * 255
    img = PIL.Image.fromarray(mask_u8, mode="L")
    if img.size != (1024, 1024):
        img = img.resize((1024, 1024), resample=PIL.Image.NEAREST)

    arr = np.array(img)

    return (arr > 127).astype(np.float32)


def save_pmap(model, row, pmap=None):
    path = pmap_path(row, model.architecture_id, model.date_range_id)
    if pmap is None:
        _, data = prepare_fits(row.fits_path)
        pmap = fits_to_pmap(model, data)
    np.save(path, pmap)
    return path, pmap


def find_or_make_pmap(row, model):
    try:
        pmap = prepare_pmap(pmap_path(row, model.architecture_id, model.date_range_id))
    except FileNotFoundError:
        print("PMAP not found, generating...")
        _, data = prepare_fits(row.fits_path)
        pmap = fits_to_pmap(model, data)
        save_pmap(model, row, pmap)
    return pmap
