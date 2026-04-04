import json
from pathlib import Path

from Library.Config import hostname, machine_config, machines, override_machine
from Library.SW.CH_SW_Model import CHSWModel

MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parent.parent
SW_CONFIG_DIR = PROJECT_ROOT / "Config" / "SW"
SW_MODEL_DIR = PROJECT_ROOT / "Models" / "CH_SW_Correspondence"
MACHINE_CONFIG_PATH = PROJECT_ROOT / "Config" / "Machine.json"

SW_RUNTIME_DEFAULTS = {
    "dense_memory_budget_gb": 10.0,
    "max_seed_batch": 256,
    "post_chunk_t": 128,
    "animation_dpi": 100,
}


def load_json_config(path):
    config_path = Path(path)
    with config_path.open("r") as handle:
        return json.load(handle)


def _resolved_machine_name():
    candidate = override_machine if override_machine else hostname
    if candidate in machines:
        return candidate
    return next(iter(machines))


def load_empirical_spec(path=None):
    config_path = Path(path) if path is not None else SW_MODEL_DIR / "Shugay.py"
    return CHSWModel.load(config_path)


def load_ballistic_spec(path=None):
    config_path = Path(path) if path is not None else SW_CONFIG_DIR / "Ballistic.json"
    raw = load_json_config(config_path)
    return {
        "json_path": config_path,
        "superresolution_enabled": bool(raw["superresolution_enabled"]),
        "superresolution_step_minutes": int(raw["superresolution_step_minutes"]),
        "base_time_step_minutes": int(raw["base_time_step_minutes"]),
        "phi_step_minutes": int(raw["phi_step_minutes"]),
        "cr_days": float(raw["cr_days"]),
        "r0": int(raw["r0"]),
        "r_solar_km": float(raw["r_solar_km"]),
        "r_max": int(raw["r_max"]),
        "horizon_hours": float(raw["horizon_hours"]),
        "use_swept_cell": bool(raw["use_swept_cell"]),
        "field_half_width_h": float(raw["field_half_width_h"]),
        "use_cr_reset": bool(raw["use_cr_reset"]),
        "memory_guard_enabled": bool(raw["memory_guard_enabled"]),
        "simulation_pad_days": float(raw["simulation_pad_days"]),
        "plot_phi_target": float(raw["plot_phi_target"]),
        "earth_phi_target": float(raw["earth_phi_target"]),
        "earth_r_target": float(raw["earth_r_target"]),
    }


def load_sw_runtime_spec(current_machine_config=None, machine_name=None):
    runtime = dict(SW_RUNTIME_DEFAULTS)
    machine = (
        machine_config if current_machine_config is None else current_machine_config
    )
    runtime.update(machine.get("sw", {}))
    resolved_machine_name = (
        _resolved_machine_name()
        if current_machine_config is None and machine_name is None
        else (machine_name if machine_name is not None else "<override>")
    )
    return {
        "machine_json_path": MACHINE_CONFIG_PATH,
        "machine_name": resolved_machine_name,
        "dense_memory_budget_gb": float(runtime["dense_memory_budget_gb"]),
        "max_seed_batch": int(runtime["max_seed_batch"]),
        "post_chunk_t": int(runtime["post_chunk_t"]),
        "animation_dpi": int(runtime["animation_dpi"]),
    }
