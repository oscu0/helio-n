import os
import glob

import numpy as np
import pandas as pd

import PIL

from sunpy.map.maputils import all_coordinates_from_map, coordinate_is_on_solar_disk
import sunpy.map


def mask_fits(f):
    hpc_coords = all_coordinates_from_map(f)
    mask = coordinate_is_on_solar_disk(hpc_coords)
    palette = f.cmap.copy()
    palette.set_bad("black")
    scaled_map = sunpy.map.Map(f.data, f.meta, mask=~mask)
    ff = scaled_map.data
    return ff


def prepare_fits(path, mask_disk=True, clip_low=1, clip_high=99):
    f = sunpy.map.Map(path)
    if mask_disk:
        ff = mask_fits(f).data
    else:
        ff = f.data
    ff = np.flipud(ff)

    low = np.percentile(ff, clip_low)
    high = np.percentile(ff, clip_high)
    ff = np.clip(ff, low, high)
    ff = (ff - low) / (high - low + 1e-6)
    return ff


def prepare_hmi_jpg(jpg_path, target_size=1024):
    """
    Load 512px HMI JPEG (grayscale), upscale to target_size.
    Return a float32 array.
    """
    img = PIL.Image.open(jpg_path).convert("F")  # 32-bit float
    img = img.resize((target_size, target_size), resample=PIL.Image.BILINEAR)
    arr = np.array(img, dtype=np.float32)
    return arr


def prepare_mask(path, preserve_255=False):
    from PIL import Image

    im = Image.open(path).convert("L")
    if not preserve_255:
        arr = np.array(im, dtype=np.float32)
        # Consider >127 as CH
        arr = (arr > 127).astype(np.float32)
    else:
        arr = np.array(im)
    return arr


def prepare_hmi_jpg(
    path="/Users/aosh/Library/Containers/net.langui.FTPMounter/Data/.FTPVolumes/dec1/mnt/sun/sdo/hmi/L0/2017/08/31/20170831_171500_M_512.jpg",
    target_size=(1024, 1024),
):
    """
    Load an HMI JPG (512×512 or similar) and upscale to AIA grid size.
    Returns float32 magnetogram-like values in range [-1,1] based on brightness.
    """
    im = Image.open(path).convert("L")  # grayscale JPG
    im = im.resize(target_size, Image.BILINEAR)
    arr = np.array(im, dtype=np.float32)

    # Convert brightness → rough polarity proxy:
    # 0 = black → strong negative
    # 255 = white → strong positive
    # midpoint 128 → zero-ish
    arr = (arr - 128.0) / 128.0  # now in approx [-1, +1]

    return arr


def prepare_dataset(fits_root, masks_root):
    def index(p):
        return p.split("/")[-1][3:16]

    # Collect all FITS and masks
    fits_files = glob.glob(os.path.join(fits_root, "**", "*.fits"), recursive=True)
    mask_files = glob.glob(
        os.path.join(masks_root, "**", "*_CH_MASK_FINAL.png"), recursive=True
    )

    df_fits = pd.DataFrame(
        {
            "key": [index(p) for p in fits_files],
            "fits_path": fits_files,
        }
    )

    df_masks = pd.DataFrame(
        {
            "key": [index(p) for p in mask_files],
            "mask_path": mask_files,
        }
    )

    # Optional: detect duplicate keys (same timestamp, multiple files)
    dup_fits = df_fits[df_fits.duplicated("key", keep=False)]
    dup_masks = df_masks[df_masks.duplicated("key", keep=False)]

    if not dup_fits.empty:
        print("⚠ Duplicate keys in FITS:")
        print(dup_fits.sort_values("key"))

    if not dup_masks.empty:
        print("⚠ Duplicate keys in masks:")
        print(dup_masks.sort_values("key"))

    df_fits = df_fits.drop_duplicates("key", keep="first")
    df_masks = df_masks.drop_duplicates("key", keep="first")

    # Outer join to see everything in one table
    merged = df_fits.merge(df_masks, on="key", how="outer", indicator=True)

    matches = merged[merged["_merge"] == "both"].copy()
    fits_only = merged[merged["_merge"] == "left_only"].copy()
    masks_only = merged[merged["_merge"] == "right_only"].copy()

    for df in (matches, fits_only, masks_only):
        df.drop(columns=["_merge"], inplace=True)

    matches.set_index(matches.key, inplace=True, drop=True)
    matches.drop(["key"], axis=1, inplace=True)
    matches["pmap_path"] = ""

    matches.to_parquet("./Data/df.parquet")
    fits_only.to_csv("./Data/fits_only.csv")
    masks_only.to_csv("./Data/masks_only.csv")

    return matches, fits_only, masks_only
