from Config import *

import numpy as np


from skimage import measure
import mahotas

from .Config import *
from . import Processing
from .IO import prepare_fits

import ipywidgets as widgets
from IPython.display import display, clear_output


def _ensure_binary_mask(mask):
    """Convert mask to boolean 2D array."""
    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise ValueError(f"mask must be 2D, got shape {mask.shape}")
    if mask.dtype == bool:
        return mask
    return mask > 0.5

def compute_zernike_descriptor(mask, degree=8):
    """
    Compute Zernike-moment-based shape descriptor for a binary mask,
    using mahotas.features.zernike_moments.

    Parameters
    ----------
    mask : 2D array-like (bool or 0/1)
        Binary mask of a single region (or union of CHs).
    degree : int
        Maximum Zernike polynomial degree (typical: 6–12).
        This 'degree' in mahotas is 'n_max' in the literature.

    Returns
    -------
    desc : np.ndarray, shape (M,)
        Rotation-invariant Zernike descriptor (magnitudes), L2-normalized.
        If the mask is empty, returns a zero vector of length 1.
    """
    mask = _ensure_binary_mask(mask)

    # --- crop to bounding box of the region ---
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        # empty mask
        return np.zeros(1, dtype=float)

    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()
    cropped = mask[y_min : y_max + 1, x_min : x_max + 1]

    h, w = cropped.shape
    # make it square by padding (centered)
    size = max(h, w)
    pad_y = (size - h) // 2
    pad_x = (size - w) // 2

    square = np.zeros((size, size), dtype=float)
    square[pad_y : pad_y + h, pad_x : pad_x + w] = cropped.astype(float)

    # mahotas assumes the Zernike circle is centered in the image
    radius = size // 2

    # mahotas.features.zernike_moments returns a 1D array of (real) magnitudes,
    # already rotation-invariant
    zm = mahotas.features.zernike_moments(square, radius, degree)

    desc = np.asarray(zm, dtype=float)

    # Optional: L2 normalize to make scale of descriptor comparable across regions
    norm = np.linalg.norm(desc)
    if norm > 0:
        desc = desc / norm

    return desc

def compute_fourier_descriptor(mask, num_descriptors=20, n_samples=256):
    """
    Compute Fourier shape descriptor from the boundary of a binary mask.

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
        based on magnitudes of low-frequency Fourier coefficients.
    """
    mask = _ensure_binary_mask(mask)

    # --- find contours at 0.5 ---
    contours = measure.find_contours(mask.astype(float), level=0.5)
    if len(contours) == 0:
        # empty mask
        return np.zeros(num_descriptors, dtype=float)

    # choose the longest contour (largest region)
    contour = max(contours, key=lambda c: c.shape[0])

    # contour: array of shape (N, 2) with (row, col) = (y, x)
    ys, xs = contour[:, 0], contour[:, 1]

    # --- resample to fixed number of points along the contour length ---
    # compute cumulative distance along contour
    dy = np.diff(ys)
    dx = np.diff(xs)
    dists = np.sqrt(dx**2 + dy**2)
    cumlen = np.concatenate([[0], np.cumsum(dists)])
    total_len = cumlen[-1]

    if total_len == 0:
        return np.zeros(num_descriptors, dtype=float)

    # new parameterization from 0 to total_len
    target = np.linspace(0, total_len, n_samples, endpoint=False)
    # interpolate x(t), y(t)
    xs_resampled = np.interp(target, cumlen, xs)
    ys_resampled = np.interp(target, cumlen, ys)

    # --- build complex sequence and normalize ---
    z = xs_resampled + 1j * ys_resampled

    # translation invariance: subtract centroid
    z = z - z.mean()

    # scale invariance: normalize by RMS radius
    scale = np.sqrt(np.mean(np.abs(z) ** 2))
    if scale > 0:
        z = z / scale

    # --- Fourier transform along contour index ---
    Z = np.fft.fft(z)
    # We ignore the DC term Z[0] (translation)
    # and use the first num_descriptors low-frequency terms.
    # For invariance to starting point & rotation, use magnitudes.
    # freq indices: 1..num_descriptors
    max_k = min(num_descriptors, len(Z) // 2)
    coeffs = Z[1 : max_k + 1]
    desc = np.abs(coeffs)

    # zero-pad if needed
    if len(desc) < num_descriptors:
        pad = np.zeros(num_descriptors - len(desc), dtype=float)
        desc = np.concatenate([desc, pad])

    # optional second normalization
    norm = np.linalg.norm(desc)
    if norm > 0:
        desc = desc / norm

    return desc

def iou(mask1, mask2):
    """
    Compute Intersection-over-Union (IoU) for two binary masks.

    Parameters
    ----------
    mask1, mask2 : array-like
        2D numpy arrays. Values can be {0,1}, {0,255}, float, or bool.

    Returns
    -------
    float
        IoU value in [0,1].
    """

    # Convert to boolean
    m1 = np.asarray(mask1) > 0.5
    m2 = np.asarray(mask2) > 0.5

    intersection = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    return intersection / union

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
    
def dice(mask1, mask2):
    m1 = np.asarray(mask1) > 0.5
    m2 = np.asarray(mask2) > 0.5

    intersection = np.logical_and(m1, m2).sum()
    a1 = m1.sum()
    a2 = m2.sum()

    denom = a1 + a2
    if denom == 0:
        return 1.0

    return 2.0 * intersection / denom

def rect_area(mask):
    range_x = [384, 640]
    range_y = [256, 768]
    return mask[range_y[0] : range_y[1], range_x[0] : range_x[1]].flatten().sum()


def stats(row, smoothing_params=smoothing_params, m2=None):
    m1 = Processing.prepare_mask(row.mask_path)
    if m2 is None:
        m2 = Processing.pmap_to_mask(Processing.fits_to_pmap(IO.prepare_fits(row.fits_path)), smoothing_params)

    stats = {}

    stats["fourier_distance"] = shape_distance(
        compute_fourier_descriptor(m1),
        compute_fourier_descriptor(m2),
    )

    stats["zernike_distance"] = shape_distance(
        compute_zernike_descriptor(m1),
        compute_zernike_descriptor(m2),
    )

    stats["rel_area"] = 1 - (rect_area(m2) / rect_area(m1))

    stats["iou"] = iou(m1, m2)
    stats["dice"] = dice(m1, m2)

    return stats

def print_distance(row, smoothing_params=smoothing_params):
    s = stats(row, smoothing_params)

    print("Fourier Distance: ", s["fourier_distance"])
    print("Zernike Distance: ", s["zernike_distance"])
    print("Center CH Area Difference (non-projective): ", s["rel_area"])
    print("I over U: ", s["iou"])
    print("Dice: ", s["dice"])