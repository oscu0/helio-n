import os
import glob
import re

import numpy as np
import pandas as pd

import PIL

from sunpy.map.maputils import all_coordinates_from_map, coordinate_is_on_solar_disk
import sunpy.map

from Library.Config import paths


def mask_fits(f):
    try:
        hpc_coords = all_coordinates_from_map(f)
        mask = coordinate_is_on_solar_disk(hpc_coords)
        palette = f.cmap.copy()
        palette.set_bad("black")
        scaled_map = sunpy.map.Map(f.data, f.meta, mask=~mask)
        ff = scaled_map.data
        return ff
    except Exception as e:
        # If WCS is missing/invalid, skip disk masking to avoid crashing the pipeline
        print(f"Warning: failed to mask FITS ({e}); returning raw data.")
        return f.data


def prepare_fits(path, mask_disk=True, clip_low=1, clip_high=99):
    """
    Load FITS as sunpy map and normalized data array.
    Returns (map_obj, map_data) where map_data is flipped, clipped, and normalized.
    """
    map_obj = sunpy.map.Map(path)
    if mask_disk:
        map_data = mask_fits(map_obj)
        if np.ma.isMaskedArray(map_data):
            map_data = map_data.filled(np.nan)
    else:
        map_data = map_obj.data
    map_data = np.flipud(map_data)

    if np.isnan(map_data).any():
        low = np.nanpercentile(map_data, clip_low)
        high = np.nanpercentile(map_data, clip_high)
        map_data = np.clip(map_data, low, high)
        map_data = (map_data - low) / (high - low + 1e-6)
        map_data = np.nan_to_num(map_data, nan=0.0)
    else:
        low = np.percentile(map_data, clip_low)
        high = np.percentile(map_data, clip_high)
        map_data = np.clip(map_data, low, high)
        map_data = (map_data - low) / (high - low + 1e-6)
    return map_obj, map_data


def resize_for_model(img2d: np.ndarray, target_size: int) -> np.ndarray:
    """
    Resize a 2-D float image to the model input size using PIL bilinear in float mode.
    Keeps values in float32; no normalization changes beyond interpolation.
    """
    if img2d.shape == (target_size, target_size):
        return img2d.astype(np.float32, copy=False)

    pil_img = PIL.Image.fromarray(img2d.astype(np.float32), mode="F")
    pil_img = pil_img.resize((target_size, target_size), resample=PIL.Image.BILINEAR)
    return np.array(pil_img, dtype=np.float32)


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
    hmi_root=None,
    max_time_delta="29min",
    out_parquet=paths["artifact_root"] + "Paths.parquet",
    hourly=True,
):

    def index(p):
        return os.path.basename(p)[3:16]

    def index_hmi(p):
        name = os.path.basename(p)
        match = re.search(r"(\\d{8}_\\d{6})", name)
        if match:
            return match.group(1)
        return name[15:28]

    # Collect files
    fits_files = glob.glob(os.path.join(fits_root, "**", "*.fits"), recursive=True)
    mask_files = glob.glob(
        os.path.join(masks_root, "**", "*_CH_MASK_FINAL*.png"), recursive=True
    )

    df_fits = pd.DataFrame(
        {"key": [index(p) for p in fits_files], "fits_path": fits_files}
    )
    df_masks = pd.DataFrame(
        {"key": [index(p) for p in mask_files], "mask_path": mask_files}
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
    dup_hmi = df_hmi[df_hmi.duplicated("key", keep=False)]

    if not dup_fits.empty:
        print("⚠ Duplicate keys in FITS:")
        print(dup_fits.sort_values("key"))
    if not dup_masks.empty:
        print("⚠ Duplicate keys in masks:")
        print(dup_masks.sort_values("key"))
    if not dup_hmi.empty:
        print("⚠ Duplicate keys in HMI:")
        print(dup_hmi.sort_values("key"))

    # Keep first occurrence for simplicity
    df_fits = df_fits.drop_duplicates("key", keep="first")
    df_masks = df_masks.drop_duplicates("key", keep="first")
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
            fill_from_nearest(df_hmi, "hmi_path")

    # Flags
    has_fits = merged["fits_path"].notna()
    has_mask = merged["mask_path"].notna()
    has_hmi = merged["hmi_path"].notna()

    # Categories
    fits_only = merged[has_fits & ~has_mask].copy()
    masks_only = merged[~has_fits & has_mask].copy()
    no_hmi = merged[has_fits & has_mask & ~has_hmi].copy()

    # Prepare "matches" DataFrame similar to previous API: use all_three primarily,
    # but include rows where pmap may be empty if pmaps_root not provided.
    # We'll create a canonical matches DF that contains any row that has at least fits and mask,
    # and include pmap_path if present (empty string otherwise).
    matches = merged[has_fits & has_mask].copy()

    # Evil pandas moment
    matches["hmi_path"] = matches["hmi_path"].fillna(np.nan).replace([np.nan], [None])

    # Convert index to the key (timestamp string) for matches
    matches.set_index(matches["key"], inplace=True, drop=True)
    matches.drop(columns=["key"], inplace=True)

    # For the convenience CSVs/parquet, drop the helper merges index column if present
    for df, path, name in [
        (fits_only, paths["artifact_root"] + "FITS Only.csv", "fits_only"),
        (masks_only, paths["artifact_root"] + "Masks Only.csv", "masks_only"),
        (no_hmi, paths["artifact_root"] + "No HMI.csv", "hmi_only"),
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

    # skip 120sec fits that are occasionally present
    if hourly:
        # extract hour key: YYYYMMDD_HH
        matches["hour"] = matches.index.str.slice(0, 11)

        matches = matches.drop_duplicates(subset="hour", keep="first").drop(columns="hour")

    # Save matches to parquet (if desired)
    matches.to_parquet(out_parquet)

    # Return core results
    return matches, {
        "fits_only": fits_only,
        "masks_only": masks_only,
        "no_hmi": no_hmi,
    }


def pmap_path(row, architecture_id, date_range_id):
    base = base_output_stem(row.mask_path)
    return f"{base}CH_MASK_{architecture_id}{date_range_id}PX.npy"


def plot_path(
    base_stem: str, architecture_id: str, date_range_id: str, postprocessing: str
) -> str:
    """Build CH plot path from base stem (no CH_MASK suffix or extension)."""
    return f"{base_stem}CH_{architecture_id}{date_range_id}{postprocessing}.png"


def unet_mask_path(
    base_stem: str, architecture_id: str, date_range_id: str, postprocessing: str
) -> str:
    """Build UNet CH mask path from base stem (no CH_MASK suffix or extension)."""
    return f"{base_stem}CH_MASK_{architecture_id}{date_range_id}{postprocessing}.png"


def base_output_stem(mask_path: str) -> str:
    """
    Given a mask path (endswith CH_MASK.png or CH_MASK_FINAL.png), return the stem to which
    architecture/date/postprocessing suffixes can be appended. Ensures no double extensions/suffixes.
    Example: /.../AIA20160801_010005_0193_CH_MASK_FINAL.png -> /.../AIA20160801_010005_0193_
    """
    base = mask_path
    if base.endswith(".png"):
        base = base[: -len(".png")]

    # Trim trailing CH_MASK and any appended suffix (e.g., CH_MASK_FINAL, CH_MASK_A1D1P1)
    if "CH_MASK" in base:
        base = base[: base.rfind("CH_MASK")]

    # Expect _FINAL in the original filename; leave a trailing underscore as before
    if not mask_path.endswith("_FINAL.png"):
        raise ValueError(f"Mask filename missing _FINAL: {mask_path}")

    return base


def synoptic_dataset(df):
    def hrmin(x):
        return int(x[-4:])

    def round_time(x):
        stub = hrmin(x)
        if stub >= 300 and stub < 900:
            x = "0600"
        elif stub >= 900 and stub < 1500:
            x = "1200"
        elif stub >= 1500 and stub < 2100:
            x = "1800"
        else:
            x = "0000"
        return x

    def round_index(x):
        return x[:8] + "_" + round_time(x)

        # df index is like "YYYYMMDD_HHMM"

    idx = df.index.astype(str)

    day = idx.str.slice(0, 8)  # YYYYMMDD
    hh = idx.str.slice(9, 11).astype(int)  # HH
    mm = idx.str.slice(11, 13).astype(int)  # MM

    tmin = hh * 60 + mm  # minutes since midnight

    targets = np.array([0, 6 * 60, 12 * 60, 18 * 60])  # 00,06,12,18 in minutes

    # distance on a circle (day wraps): min(|t-target|, 1440-|t-target|)
    diff = np.abs(tmin.to_numpy()[:, None] - targets[None, :])
    circ = np.minimum(diff, 1440 - diff)

    tmp = df.copy()
    tmp["_day"] = day.to_numpy()
    tmp["_tmin"] = tmin.to_numpy()

    keep_idx = []
    max_offset = 180  # minutes (+/- 3 hours)
    for k, target in enumerate(targets):
        # pick the closest row per day to this target (wrap-aware)
        d = circ[:, k]
        sub = tmp.assign(_d=d)
        sub = sub[sub["_d"] <= max_offset]
        if sub.empty:
            continue
        best = (
            sub.sort_values(["_day", "_d", "_tmin"]).groupby("_day", sort=False).head(1)
        )
        keep_idx.append(best.index)

    keep = keep_idx[0].union(keep_idx[1]).union(keep_idx[2]).union(keep_idx[3])

    df_mini = df.loc[keep].sort_index().copy()

    df_mini.index = df_mini.index.map(round_index)

    df_mini.to_parquet(paths["artifact_root"] + "Paths (Synoptic).parquet")

    return df_mini
