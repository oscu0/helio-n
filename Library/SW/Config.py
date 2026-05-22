import json
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
    "dense_memory_budget_gb": 10.0,
    "max_seed_batch": 256,
    "post_chunk_t": 128,
    "animation_dpi": 100,
    "animation_1h_mult": 2.0,
}


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
