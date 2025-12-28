import os
import glob

import numpy as np
import pandas as pd

import PIL

from sunpy.map.maputils import all_coordinates_from_map, coordinate_is_on_solar_disk
import sunpy.map

from Library.Config import paths


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

    im = PIL.Image.open(path).convert("L")
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
    im = PIL.Image.open(path).convert("L")  # grayscale JPG
    im = im.resize(target_size, PIL.Image.BILINEAR)
    arr = np.array(im, dtype=np.float32)

    # Convert brightness → rough polarity proxy:
    # 0 = black → strong negative
    # 255 = white → strong positive
    # midpoint 128 → zero-ish
    arr = (arr - 128.0) / 128.0  # now in approx [-1, +1]

    return arr


def prepare_pmap(path):
    # very complex and useful function
    return np.load(path)


def prepare_dataset(
    fits_root,
    masks_root,
    pmaps_root=-1,
    architecture="A1",
    date_range="D1",
    hmi_root=None,
    max_time_delta="29min",
    out_parquet=paths["artifact_root"] + "Paths.parquet",
):
    """
    Scan FITS / mask / (optional) pmap roots and return matched & unmatched DataFrames.
    Inexact matches are allowed within max_time_delta when aligning to FITS keys.

    Returns
    -------
    matches : DataFrame
        Rows where FITS + mask + pmap (pmap_path may be empty if pmaps_root is None or missing).
        Index is the key (timestamp string).
        Columns: fits_path, mask_path, pmap_path
    fits_only : DataFrame
        Rows with FITS present but neither mask nor pmap.
    masks_only : DataFrame
        Rows with mask present but neither fits nor pmap.
    pmaps_only : DataFrame
        Rows with pmap present but neither fits nor mask (empty if pmaps_root is None).
    partials : DataFrame
        Rows where exactly two of the three are present (fits+mask, fits+pmap, mask+pmap).
    """

    if pmaps_root == -1:
        pmaps_root = masks_root

    def index(p):
        return os.path.basename(p)[3:16]

    def index_hmi(p):
        return os.path.basename(p)[15:28]

    # Collect files
    fits_files = glob.glob(os.path.join(fits_root, "**", "*.fits"), recursive=True)
    mask_files = glob.glob(
        os.path.join(masks_root, "**", "*_CH_MASK.png"), recursive=True
    )

    df_fits = pd.DataFrame(
        {"key": [index(p) for p in fits_files], "fits_path": fits_files}
    )
    df_masks = pd.DataFrame(
        {"key": [index(p) for p in mask_files], "mask_path": mask_files}
    )

    # Optional
    if pmaps_root is not None:
        pmap_files = glob.glob(
            os.path.join(
                pmaps_root, "**", "*_CH_" + architecture + date_range + "PX" + ".npy"
            ),
            recursive=True,
        )
    else:
        pmap_files = []

    df_pmaps = pd.DataFrame(
        {"key": [index(p) for p in pmap_files], "pmap_path": pmap_files}
    )

    if hmi_root is not None:
        hmi_files = glob.glob(os.path.join(hmi_root, "**", "hmi*.fits"), recursive=True)
    else:
        hmi_files = []

    df_hmi = pd.DataFrame(
        {"key": [index_hmi(p) for p in hmi_files], "hmi_path": hmi_files}
    )

    # Report duplicates
    dup_fits = df_fits[df_fits.duplicated("key", keep=False)]
    dup_masks = df_masks[df_masks.duplicated("key", keep=False)]
    dup_pmaps = df_pmaps[df_pmaps.duplicated("key", keep=False)]
    dup_hmi = df_hmi[df_hmi.duplicated("key", keep=False)]

    if not dup_fits.empty:
        print("⚠ Duplicate keys in FITS:")
        print(dup_fits.sort_values("key"))
    if not dup_masks.empty:
        print("⚠ Duplicate keys in masks:")
        print(dup_masks.sort_values("key"))
    if not dup_pmaps.empty:
        print("⚠ Duplicate keys in PMAPs:")
        print(dup_pmaps.sort_values("key"))
    if not dup_hmi.empty:
        print("⚠ Duplicate keys in HMI:")
        print(dup_hmi.sort_values("key"))

    # Keep first occurrence for simplicity
    df_fits = df_fits.drop_duplicates("key", keep="first")
    df_masks = df_masks.drop_duplicates("key", keep="first")
    df_pmaps = df_pmaps.drop_duplicates("key", keep="first")
    df_hmi = df_hmi.drop_duplicates("key", keep="first")

    def to_dt(series):
        cleaned = series.astype(str).str.replace(r"\D", "", regex=True)
        with_seconds = cleaned.where(cleaned.str.len() == 14)
        with_minutes = cleaned.where(cleaned.str.len() == 12)
        dt = pd.to_datetime(with_seconds, format="%Y%m%d%H%M%S", errors="coerce")
        dt_minutes = pd.to_datetime(with_minutes, format="%Y%m%d%H%M", errors="coerce")
        return dt.fillna(dt_minutes)

    # Full outer join of three tables
    merged = df_fits.merge(df_masks, on="key", how="outer")
    merged = merged.merge(df_pmaps, on="key", how="outer")
    merged = merged.merge(df_hmi, on="key", how="outer")

    if max_time_delta is not None:
        tolerance = pd.Timedelta(max_time_delta)
        fits_align = df_fits[["key"]].copy()
        fits_align["dt"] = to_dt(fits_align["key"])
        fits_align = fits_align.dropna(subset=["dt"]).sort_values("dt")

        if not fits_align.empty:

            def fill_from_nearest(df_other, path_col):
                nonlocal merged
                if df_other.empty:
                    return
                other_align = df_other[["key", path_col]].copy()
                other_align["dt"] = to_dt(other_align["key"])
                other_align = other_align.rename(columns={"key": "other_key"})
                other_align = other_align.dropna(subset=["dt"]).sort_values("dt")
                if other_align.empty:
                    return
                aligned = pd.merge_asof(
                    fits_align,
                    other_align,
                    on="dt",
                    direction="nearest",
                    tolerance=tolerance,
                )
                aligned = aligned.dropna(subset=[path_col])
                if aligned.empty:
                    return
                aligned = aligned[["key", path_col]].rename(
                    columns={path_col: f"{path_col}_fuzzy"}
                )
                merged = merged.merge(aligned, on="key", how="left")
                merged[path_col] = merged[path_col].fillna(merged[f"{path_col}_fuzzy"])
                merged.drop(columns=[f"{path_col}_fuzzy"], inplace=True)

            fill_from_nearest(df_masks, "mask_path")
            fill_from_nearest(df_pmaps, "pmap_path")
            fill_from_nearest(df_hmi, "hmi_path")

    # Flags
    has_fits = merged["fits_path"].notna()
    has_mask = merged["mask_path"].notna()
    has_pmap = merged["pmap_path"].notna()
    has_hmi = merged["hmi_path"].notna()

    # Categories
    fits_only = merged[has_fits & ~has_mask & ~has_pmap].copy()
    masks_only = merged[~has_fits & has_mask & ~has_pmap].copy()
    pmaps_only = merged[~has_fits & ~has_mask & has_pmap].copy()
    hmi_only = merged[~has_fits & ~has_mask & has_hmi].copy()

    # Prepare "matches" DataFrame similar to previous API: use all_three primarily,
    # but include rows where pmap may be empty if pmaps_root not provided.
    # We'll create a canonical matches DF that contains any row that has at least fits and mask,
    # and include pmap_path if present (empty string otherwise).
    matches = merged[has_fits & has_mask].copy()
    # Evil pandas moment
    matches["pmap_path"] = matches["pmap_path"].fillna(np.nan).replace([np.nan], [None])
    matches["hmi_path"] = matches["hmi_path"].fillna(np.nan).replace([np.nan], [None])

    # Convert index to the key (timestamp string) for matches
    matches.set_index(matches["key"], inplace=True, drop=True)
    matches.drop(columns=["key"], inplace=True)

    # For the convenience CSVs/parquet, drop the helper merges index column if present
    for df, path, name in [
        (fits_only, paths["artifact_root"] + "FITS Only.csv", "fits_only"),
        (masks_only, paths["artifact_root"] + "Masks Only.csv", "masks_only"),
        (pmaps_only, paths["artifact_root"] + "PMAPs Only.csv", "pmaps_only"),
        (hmi_only, paths["artifact_root"] + "HMI Only.csv", "hmi_only"),
    ]:
        # ensure we don't fail when a category is empty
        try:
            df.drop(columns=["key"], inplace=True)
        except Exception:
            pass
        try:
            df.to_csv(path)
        except Exception:
            # if writing fails (e.g. empty), ignore
            pass

    # Save matches to parquet (if desired)
    try:
        matches.to_parquet(out_parquet)
    except Exception:
        pass

    # Return core results
    return matches, {
        "fits_only": fits_only,
        "masks_only": masks_only,
        "pmaps_only": pmaps_only,
        "hmi_only": hmi_only,
    }
