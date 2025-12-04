import json

import numpy as np

import PIL

from skimage.morphology import (
    binary_closing,
    disk,
    remove_small_objects,
    remove_small_holes,
)
from skimage.transform import resize


from Library.Config import *
from Library.IO import prepare_fits


def fits_to_pmap(model, img2d, img_size=model_params["img_size"]):
    img = np.asarray(img2d, dtype=np.float32)

    if img.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {img.shape}")

    if img.shape != (img_size, img_size):
        # Use float-preserving PIL mode "F", bilinear interpolation
        pil_img = PIL.Image.fromarray(img.astype(np.float32), mode="F")
        pil_img = pil_img.resize((img_size, img_size), resample=PIL.Image.BILINEAR)
        img = np.array(pil_img, dtype=np.float32)

    x = img[np.newaxis, ..., np.newaxis]  # (1, H, W, 1)

    prob = model.predict(x, verbose=0)[0, ..., 0]

    return prob


def pmap_to_mask(pmap, smoothing_params=smoothing_params, save_path=None):
    """
    Convert a probability map into a cleaned binary mask (0/1 float32),
    matching the semantics of prepare_mask(preserve_255=False).

    If save=True, also save a 0–255 uint8 PNG to save_path.
    """

    # --- threshold to get binary CH mask ---
    mask = (pmap > smoothing_params["threshold"]).astype(np.float32)

    # --- morphological smoothing ---
    # if smoothing_params["closing_radius"] > 0:
    mask = binary_closing(mask, disk(smoothing_params["closing_radius"]))

    if smoothing_params["min_size"] > 0:
        mask = remove_small_objects(mask, min_size=smoothing_params["min_size"])

    if smoothing_params["hole_size"] > 0:
        mask = remove_small_holes(mask, area_threshold=smoothing_params["hole_size"])


    mask_u8 = (mask > 0.5).astype(np.uint8) * 255
    img = PIL.Image.fromarray(mask_u8, mode="L")
    if img.size != (1024, 1024):
        img = img.resize((1024, 1024), resample=PIL.Image.NEAREST)

    arr = np.array(img)

    return (arr > 127).astype(np.float32)



def save_pmap(model, row, pmap=None):
    path = row.mask_path.replace("CH_MASK_FINAL.png", "UNET_PMAP.npy")
    if pmap is None:
        pmap = fits_to_pmap(model, prepare_fits(row.fits_path))
    np.save(path, pmap)
    return path
