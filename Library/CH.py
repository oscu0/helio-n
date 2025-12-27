import astropy.units as u
import sunpy.map
import numpy as np
from Library import Processing
from Library.Config import *
from Library.IO import prepare_fits, prepare_mask, prepare_pmap
from Library.Metrics import generate_omask
from sunpy.coordinates import frames
from sunpy.map.maputils import all_coordinates_from_map, coordinate_is_on_solar_disk


def project(map, mask):
    mask = np.asarray(mask, dtype=float)
    H, W = mask.shape

    # pixel grid
    yy, xx = np.indices((H, W))  # yy=row, xx=col

    x_pix = (xx * u.pix).ravel()
    y_pix = (yy * u.pix).ravel()

    # pixel -> heliographic Stonyhurst
    hpc = map.pixel_to_world(x_pix, y_pix)
    hgs = hpc.transform_to(frames.HeliographicStonyhurst)

    lat = hgs.lat.to(u.rad).value.reshape(H, W)
    lon = hgs.lon.to(u.rad).value.reshape(H, W)

    # disk-centre latitude B0
    B0 = map.observer_coordinate.heliographic_stonyhurst.lat.to(u.rad).value

    # mu = cos(rho)
    mu = np.sin(lat) * np.sin(B0) + np.cos(lat) * np.cos(B0) * np.cos(lon)

    # avoid limb/pole problems
    proj_mask = np.zeros_like(mask, dtype=float)
    good = (mask != 0) & (mu > 1e-3)  # ignore pixels where mu ~ 0 (near limb)
    proj_mask[good] = mask[good] / mu[good]

    return proj_mask


def ch_abs_area(row, reference_mode=False, oval=True):
    m = sunpy.map.Map(row.fits_path)

    if reference_mode:
        ch_mask_map = prepare_mask(row.mask_path)
    else:
        ch_mask_map = Processing.pmap_to_mask(
            prepare_pmap(row.pmap_path), smoothing_params=Processing.get_postprocessing_params("P0")
        )

    if oval:
        ch_mask_map *= generate_omask(row)

    mask_proj = project(m, ch_mask_map)

    return np.nan_to_num(mask_proj, 0).sum()
    # return mask_proj


def omask_area(row):
    omask = generate_omask(row)
    return project(sunpy.map.Map(row.fits_path), omask).sum()


def sun_area(row):
    map = sunpy.map.Map(row.fits_path)
    hpc_coords = all_coordinates_from_map(map)
    mask = coordinate_is_on_solar_disk(hpc_coords)
    return project(map, mask).sum()


def ch_rel_area(row, reference_mode=False):
    ch_area = ch_abs_area(row, reference_mode, oval=True)

    # print("OMASK AREA:", omask_area(row, generate_omask(row)))
    # print("CH AREA:", ch_area)

    return ch_area / sun_area(row)
    # return ch_area / omask_area(row, generate_omask(row))
