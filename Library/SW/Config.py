import json
from dataclasses import dataclass
from pathlib import Path

from Library.Config import machine_config
from Library.SW.CH_SW_Model import load_ch_sw_model
from Models.CH_SW_Correspondence.Shugay_Slow_SW import (
    load as load_slow_sw_patch_model,
)

MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parent.parent
SW_CONFIG_DIR = PROJECT_ROOT / "Config" / "SW"

SW_RUNTIME_DEFAULTS = {
    "animation_dpi": 100,
}


@dataclass(frozen=True)
class SatelliteConfig:
    """Small registry entry for a satellite or packaged measurement source."""

    sat_id: str
    label: str
    loader: str | None
    coord_frame: str = "HEE"
    icme_catalog: str | None = None


SATELLITE_CONFIGS = {
    "ace": SatelliteConfig("ace", "ACE", None),
    "ace_earth": SatelliteConfig(
        "ace_earth", "ACE @ Earth", "ace_earth", icme_catalog="unified_earth"
    ),
    "earth": SatelliteConfig("earth", "Earth", None),
    "psp": SatelliteConfig("psp", "PSP", None),
    "solo": SatelliteConfig("solo", "Solar Orbiter", None),
    "stereo_a": SatelliteConfig(
        "stereo_a", "STEREO-A", "stereo_a", coord_frame="HGS", icme_catalog="icmecat_v2.3"
    ),
    "stereo_b": SatelliteConfig("stereo_b", "STEREO-B", None),
}
DEFAULT_ENABLED_SATELLITES = ("ace_earth", "stereo_a")


def get_satellite_config(sat_id):
    sat_id = str(sat_id)
    assert sat_id in SATELLITE_CONFIGS, (
        f"Unknown satellite ID: {sat_id!r}; "
        f"available IDs: {sorted(SATELLITE_CONFIGS)}"
    )
    return SATELLITE_CONFIGS[sat_id]


def parse_satellite_ids(value=None):
    if value is None:
        return list(DEFAULT_ENABLED_SATELLITES)
    if value == "none":
        return []
    satellite_ids = [item.strip() for item in str(value).split(",")]
    assert all(satellite_ids), "Satellite selection contains an empty ID"
    for sat_id in satellite_ids:
        get_satellite_config(sat_id)
    assert len(satellite_ids) == len(set(satellite_ids)), (
        "Satellite selection contains duplicates"
    )
    return satellite_ids


def load_empirical_spec():
    return load_ch_sw_model()


def load_slow_sw_patch_spec():
    return load_slow_sw_patch_model()


def load_ballistic_spec():
    with (SW_CONFIG_DIR / "Ballistic.json").open("r") as handle:
        raw = json.load(handle)
    raw["json_path"] = SW_CONFIG_DIR / "Ballistic.json"
    return raw


def load_sw_runtime_spec():
    runtime = dict(SW_RUNTIME_DEFAULTS)
    runtime.update(machine_config.get("sw", {}))
    return runtime
