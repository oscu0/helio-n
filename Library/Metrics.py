import astropy.units as u
import numpy as np
import sunpy.map
from sunpy.coordinates import frames


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
