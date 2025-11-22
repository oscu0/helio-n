import json

import numpy as np

import PIL

from skimage.morphology import (
    binary_closing,
    disk,
    remove_small_objects,
    remove_small_holes,
)


with open("./Config/Smoothing Params.json", "r") as f:
    smoothing_params = json.load(f)


with open("./Config/Training Params.json", "r") as f:
    model_params = json.load(f)


def fits_to_pmap(model, img2d, resize=False, img_size=model_params["img_size"]):
    img = np.asarray(img2d, dtype=np.float32)

    if img.ndim != 2:
        raise ValueError(f"Expected 2D array, got shape {img.shape}")

    if resize and img.shape != (img_size, img_size):
        # Use float-preserving PIL mode "F", bilinear interpolation
        pil_img = PIL.Image.fromarray(img.astype(np.float32), mode="F")
        pil_img = pil_img.resize((img_size, img_size), resample=PIL.Image.BILINEAR)
        img = np.array(pil_img, dtype=np.float32)

    x = img[np.newaxis, ..., np.newaxis]  # (1, H, W, 1)

    prob = model.predict(x, verbose=0)[0, ..., 0]

    return prob


def pmap_to_mask(pmap, smoothing_params=smoothing_params, save=False):
    mask = pmap > smoothing_params["threshold"]  # binary mask

    if smoothing_params["closing_radius"] > 0:
        mask = binary_closing(mask, disk(smoothing_params["closing_radius"]))

    if smoothing_params["min_size"] > 0:
        mask = remove_small_objects(mask, min_size=smoothing_params["min_size"])

    if smoothing_params["hole_size"] > 0:
        mask = remove_small_holes(mask, area_threshold=smoothing_params["hole_size"])

    # mask_uint8 = (mask * 255).clip(0, 255).astype(np.uint8)
    img = PIL.Image.fromarray(mask)

    if img.size != (1024, 1024):
        img = img.resize((1024, 1024), resample=PIL.Image.NEAREST)

    return np.array(img)


def save_pmap(row, pmap=None):
    path = row.mask_path.replace("CH_MASK_FINAL.png", "UNET_PMAP.npy")
    if pmap is None:
        pmap = fits_to_pmap(prepare_fits(row.fits_path))
    np.save(path, pmap)
    return path
