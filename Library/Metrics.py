import astropy.units as u
import mahotas
import numpy as np
import sunpy.map
from skimage import measure
from sunpy.coordinates import frames

from Config import *
from Library import Processing
from Library.Config import *
from Library.IO import prepare_fits, prepare_mask, prepare_pmap


def _ensure_binary_mask(mask):
    """Convert mask to boolean 2D array."""
    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D, got shape {mask.shape}")
    if mask.dtype == bool:
        return mask
    return mask > 0.5


def generate_omask(row, lon_half=20, lat_half=40):
    m = sunpy.map.Map(row.fits_path)
    ny, nx = m.data.shape

    # Pixel grid (row = y, col = x)
    yy, xx = np.mgrid[0:ny, 0:nx]

    # Pixel -> world (Helioprojective → Helio. Stonyhurst)
    coords = m.pixel_to_world(xx * u.pix, yy * u.pix)
    hgs = coords.transform_to(frames.HeliographicStonyhurst)

    # Angles in degrees
    lon_deg = hgs.lon.to(u.deg).value  # [-180, 180]
    lat_deg = hgs.lat.to(u.deg).value  # [-90, 90]

    omask = (lon_deg / lon_half) ** 2 + (lat_deg / lat_half) ** 2 <= 1.0
    return omask


def compute_zernike_descriptor(mask, omask=None, degree=8):
    """
    Compute a Zernike-moment-based shape descriptor for a binary mask
    representing a union of coronal-hole regions.

    This version properly handles *multiple disconnected regions*
    by treating their union as a single composite shape.

    Parameters
    ----------
    mask : 2D binary array (bool or 0/1)
        Union of all CH pixels.
    degree : int
        Maximum Zernike polynomial degree (n_max).
        Typical useful range: 6–12.

    Returns
    -------
    desc : np.ndarray
        Real-valued rotation-invariant Zernike descriptor.
        L2-normalized; zero-vector if mask empty.
    """
    # --- ensure strictly binary mask ---
    mask = np.asarray(mask > 0, dtype=np.uint8)

    if omask is not None:
        region = np.asarray(omask, dtype=bool)
        mask = np.logical_and(mask, region)

    # --- extract union bounding box ---
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return np.zeros(1, dtype=float)

    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()
    cropped = mask[y_min : y_max + 1, x_min : x_max + 1]

    # --- pad to square (centered) ---
    h, w = cropped.shape
    size = max(h, w)
    pad_y = (size - h) // 2
    pad_x = (size - w) // 2

    square = np.zeros((size, size), dtype=float)
    square[pad_y : pad_y + h, pad_x : pad_x + w] = cropped

    # --- Zernike moments require radius = center of square ---
    radius = size // 2

    # --- compute Zernike descriptor (rotation-invariant magnitudes) ---
    zm = mahotas.features.zernike_moments(square, radius, degree)

    desc = np.asarray(zm, dtype=float)

    # --- L2-normalize for numerical stability ---
    norm = np.linalg.norm(desc)
    if norm > 0:
        desc = desc / norm

    return desc


def compute_fourier_descriptor(mask, omask=None, num_descriptors=20, n_samples=256):
    """
    Compute Fourier shape descriptor from the boundaries of a binary mask.

    This version:
      - works on the *union* of all regions,
      - computes a descriptor for each contour,
      - averages descriptors across all contours,
      - keeps your original call/return semantics.

    Parameters
    ----------
    mask : 2D array-like (bool or 0/1)
        Binary mask of a region (or union of CHs).
    num_descriptors : int
        Number of low-frequency coefficients to keep (excluding DC).
    n_samples : int
        Number of boundary points to resample to (uniform along contour).

    Returns
    -------
    desc : np.ndarray, shape (num_descriptors,)
        Rotation/translation/starting-point invariant boundary descriptor,
        based on magnitudes of low-frequency Fourier coefficients,
        averaged over all contours. If the mask has no valid contours,
        returns a zero vector of length num_descriptors.
    """
    mask = _ensure_binary_mask(mask)  # keep your helper

    if omask is not None:
        region = np.asarray(omask, dtype=bool)
        mask = np.logical_and(mask, region)

    # --- find *all* contours at level 0.5 ---
    contours = measure.find_contours(mask.astype(float), level=0.5)

    if len(contours) == 0:
        return np.zeros(num_descriptors, dtype=float)

    desc_list = []

    for contour in contours:
        # contour: array of shape (N, 2) with (row, col) = (y, x)
        if contour.shape[0] < 10:
            # ignore tiny/noisy contours
            continue

        ys, xs = contour[:, 0], contour[:, 1]

        # --- resample to fixed number of points along the contour length ---
        dy = np.diff(ys)
        dx = np.diff(xs)
        dists = np.sqrt(dx**2 + dy**2)
        cumlen = np.concatenate([[0.0], np.cumsum(dists)])
        total_len = cumlen[-1]

        if total_len == 0:
            continue

        # new parameterization from 0 to total_len
        target = np.linspace(0.0, total_len, n_samples, endpoint=False)
        xs_resampled = np.interp(target, cumlen, xs)
        ys_resampled = np.interp(target, cumlen, ys)

        # --- build complex sequence and normalize ---
        z = xs_resampled + 1j * ys_resampled

        # translation invariance: subtract centroid
        z = z - z.mean()

        # scale invariance: normalize by RMS radius
        scale = np.sqrt(np.mean(np.abs(z) ** 2))
        if scale == 0:
            continue
        z = z / scale

        # --- Fourier transform along contour index ---
        Z = np.fft.fft(z)

        # ignore DC term Z[0]; keep first num_descriptors low-freq terms
        max_k = min(num_descriptors, len(Z) // 2)
        coeffs = Z[1 : max_k + 1]

        # magnitudes → rotation & starting-point invariance
        desc = np.abs(coeffs)

        # zero-pad if needed
        if len(desc) < num_descriptors:
            pad = np.zeros(num_descriptors - len(desc), dtype=float)
            desc = np.concatenate([desc, pad])

        # per-contour normalization (optional but stabilizing)
        norm = np.linalg.norm(desc)
        if norm > 0:
            desc = desc / norm

        desc_list.append(desc)

    if not desc_list:
        return np.zeros(num_descriptors, dtype=float)

    # --- average descriptors over all contours ---
    desc_stack = np.stack(desc_list, axis=0)  # (n_contours, num_descriptors)
    desc_mean = desc_stack.mean(axis=0)

    # final normalization for comparability across images
    norm = np.linalg.norm(desc_mean)
    if norm > 0:
        desc_mean = desc_mean / norm

    return desc_mean


def iou(mask1, mask2, omask=None):
    """
    Compute Intersection-over-Union (IoU) for two binary masks,
    optionally restricted to a supplied region mask (e.g. equatorial oval).

    Parameters
    ----------
    mask1, mask2 : 2D arrays
        Binary masks (0/1, 0/255, bool, float).
    omask : 2D bool array or None
        If provided, IoU is computed only inside this region.

    Returns
    -------
    float
        IoU in [0,1].
    """

    # convert to boolean masks
    m1 = np.asarray(mask1) > 0.5
    m2 = np.asarray(mask2) > 0.5

    # apply oval if supplied
    if omask is not None:
        region = np.asarray(omask, dtype=bool)
        m1 = np.logical_and(m1, region)
        m2 = np.logical_and(m2, region)

    intersection = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    return intersection / union


def dice(mask1, mask2, omask=None):
    m1 = np.asarray(mask1) > 0.5
    m2 = np.asarray(mask2) > 0.5

    # apply oval if supplied
    if omask is not None:
        region = np.asarray(omask, dtype=bool)
        m1 = np.logical_and(m1, region)
        m2 = np.logical_and(m2, region)

    intersection = np.logical_and(m1, m2).sum()
    a1 = m1.sum()
    a2 = m2.sum()

    denom = a1 + a2
    if denom == 0:
        return 1.0

    return 2.0 * intersection / denom


def shape_distance(desc_a, desc_b, metric="l2"):
    """
    Compute distance between two descriptor vectors.

    Parameters
    ----------
    desc_a, desc_b : array-like
        Descriptor vectors (e.g., Zernike or Fourier descriptors).
    metric : {"l2", "l1", "cosine"}
        Distance / dissimilarity measure.

    Returns
    -------
    d : float
        Distance (larger = more dissimilar).
    """
    a = np.asarray(desc_a, dtype=float)
    b = np.asarray(desc_b, dtype=float)

    if metric == "l2":
        return np.linalg.norm(a - b)
    elif metric == "l1":
        return np.sum(np.abs(a - b))
    elif metric == "cosine":
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na == 0 or nb == 0:
            return 1.0  # maximal dissimilarity
        cos_sim = np.dot(a, b) / (na * nb)
        return 1.0 - cos_sim  # cosine distance
    else:
        raise ValueError(f"Unknown metric: {metric!r}")


def abs_area(mask, omask):
    """
    Count the number of CH pixels (mask==1) inside a supplied oval mask.

    Parameters
    ----------
    mask : 2D array-like
        Binary mask (0/1, 0/255, bool, float).
    omask : 2D bool array
        Region mask produced by generate_omask().

    Returns
    -------
    int
        Number of pixels where mask == 1 inside the oval.
    """

    # convert to boolean CH mask
    m = np.asarray(mask) > 0.5

    # restrict to oval
    region = np.asarray(omask, dtype=bool)

    # count CH pixels inside oval
    return np.logical_and(m, region).sum()


def rel_area(mask, omask):
    return abs_area(mask, omask) / omask.sum()


def stats(
    row,
    smoothing_params=Processing.get_postprocessing_params("P0"),
    m2=None,
    model=None,
    oval=None,
):
    if oval is None:
        oval = generate_omask(row)

    m1 = prepare_mask(row.mask_path)
    if m2 is None:
        if row.pmap_path is not None:
            pmap = prepare_pmap(row.pmap_path)
        else:
            pmap = Processing.fits_to_pmap(model, prepare_fits(row.fits_path))

        m2 = Processing.pmap_to_mask(
            pmap,
            smoothing_params,
        )

    stats = {}

    stats["fourier_distance"] = shape_distance(
        compute_fourier_descriptor(m1, omask=oval),
        compute_fourier_descriptor(m2, omask=oval),
    )

    stats["zernike_distance"] = shape_distance(
        compute_zernike_descriptor(m1, omask=oval),
        compute_zernike_descriptor(m2, omask=oval),
    )

    c1 = abs_area(m1, oval)
    c2 = abs_area(m2, oval)

    if c1 + c2 == 0:
        # both are zero; perfect match
        stats["rel_area"] = 0.0
    elif c1 * c2 == 0:
        # only one is zero, 100% mismatch
        stats["rel_area"] = 1.0
    else:
        # same as IoU, conceptually
        stats["rel_area"] = 1 - (min(c1, c2) / max(c1, c2))

    stats["iou"] = iou(m1, m2, oval)
    stats["dice"] = dice(m1, m2, oval)

    return stats


def print_distance(
    row, model, smoothing_params=Processing.get_postprocessing_params("P0")
):
    s = stats(row, smoothing_params, model=model)

    print("Fourier Distance: ", s["fourier_distance"])
    print("Zernike Distance: ", s["zernike_distance"])
    print("Center CH Area Difference (non-projective): ", s["rel_area"])
    print("I over U: ", s["iou"])
    print("Dice: ", s["dice"])
