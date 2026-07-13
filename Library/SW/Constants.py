import astropy.units as u
from sunpy.sun import constants as sun_constants


SW_MODEL_V2_HANDOFF = "2019-01-01 00:00:00"
CARRINGTON_ROTATION_DAYS = float(sun_constants.mean_synodic_period.to_value(u.day))
SOLAR_RADIUS_KM = float(sun_constants.radius.to_value(u.km))
