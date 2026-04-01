import json
from dataclasses import dataclass
from pathlib import Path

from Library.Config import hostname, machine_config, machines, override_machine

MODULE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = MODULE_DIR.parent.parent
SW_CONFIG_DIR = PROJECT_ROOT / "Config" / "SW"
MACHINE_CONFIG_PATH = PROJECT_ROOT / "Config" / "Machine.json"

SW_RUNTIME_DEFAULTS = {
    "dense_memory_budget_gb": 10.0,
    "max_seed_batch": 256,
    "input_chunk_rows": 200000,
    "post_chunk_t": 128,
    "sparse_time_chunk": 48,
    "animation_dpi": 100,
}


@dataclass(frozen=True)
class EmpiricalSpec:
    """Typed view of Config/SW/Empirical.json."""

    json_path: Path
    slow_sw_speed: float
    a: float
    alpha: float


@dataclass(frozen=True)
class BallisticSpec:
    """Typed view of Config/SW/Ballistic.json."""

    json_path: Path
    superresolution_enabled: bool
    superresolution_step_minutes: int
    base_time_step_minutes: int
    phi_step_minutes: int
    cr_days: float
    r0: int
    r_solar_km: float
    r_max: int
    horizon_hours: float
    use_swept_cell: bool
    field_half_width_h: float
    use_cr_reset: bool
    prop_stats_mode: str
    memory_guard_enabled: bool
    simulation_pad_days: float
    plot_phi_target: float
    earth_phi_target: float
    earth_r_target: float


@dataclass(frozen=True)
class RuntimeSpec:
    """Typed view of the active machine's sw block in Config/Machine.json."""

    machine_json_path: Path
    machine_name: str
    dense_memory_budget_gb: float
    max_seed_batch: int
    input_chunk_rows: int
    post_chunk_t: int
    sparse_time_chunk: int
    animation_dpi: int


@dataclass(frozen=True)
class TimeControls:
    """Resolved cadence controls derived from Ballistic.json."""

    source_json_path: Path
    superresolution_enabled: bool
    superresolution_step_minutes: int
    base_time_step_minutes: int
    time_step_minutes: int
    time_step_hours: float
    time_freq: str


def load_json_config(path):
    config_path = Path(path)
    with config_path.open("r") as handle:
        return json.load(handle)


def load_empirical_config(path=None):
    config_path = Path(path) if path is not None else SW_CONFIG_DIR / "Empirical.json"
    return load_json_config(config_path)


def load_ballistic_config(path=None):
    config_path = Path(path) if path is not None else SW_CONFIG_DIR / "Ballistic.json"
    return load_json_config(config_path)


def load_sw_runtime_config(current_machine_config=None):
    runtime = dict(SW_RUNTIME_DEFAULTS)
    machine = (
        machine_config if current_machine_config is None else current_machine_config
    )
    runtime.update(machine.get("sw", {}))
    return runtime


def _resolved_machine_name():
    candidate = override_machine if override_machine else hostname
    if candidate in machines:
        return candidate
    return next(iter(machines))


def load_empirical_spec(path=None):
    config_path = Path(path) if path is not None else SW_CONFIG_DIR / "Empirical.json"
    raw = load_empirical_config(config_path)
    return EmpiricalSpec(
        json_path=config_path,
        slow_sw_speed=float(raw["slow_sw_speed"]),
        a=float(raw["a"]),
        alpha=float(raw["alpha"]),
    )


def load_ballistic_spec(path=None):
    config_path = Path(path) if path is not None else SW_CONFIG_DIR / "Ballistic.json"
    raw = load_ballistic_config(config_path)
    return BallisticSpec(
        json_path=config_path,
        superresolution_enabled=bool(raw["superresolution_enabled"]),
        superresolution_step_minutes=int(raw["superresolution_step_minutes"]),
        base_time_step_minutes=int(raw["base_time_step_minutes"]),
        phi_step_minutes=int(raw["phi_step_minutes"]),
        cr_days=float(raw["cr_days"]),
        r0=int(raw["r0"]),
        r_solar_km=float(raw["r_solar_km"]),
        r_max=int(raw["r_max"]),
        horizon_hours=float(raw["horizon_hours"]),
        use_swept_cell=bool(raw["use_swept_cell"]),
        field_half_width_h=float(raw["field_half_width_h"]),
        use_cr_reset=bool(raw["use_cr_reset"]),
        prop_stats_mode=str(raw["prop_stats_mode"]),
        memory_guard_enabled=bool(raw["memory_guard_enabled"]),
        simulation_pad_days=float(raw["simulation_pad_days"]),
        plot_phi_target=float(raw["plot_phi_target"]),
        earth_phi_target=float(raw["earth_phi_target"]),
        earth_r_target=float(raw["earth_r_target"]),
    )


def load_sw_runtime_spec(current_machine_config=None, machine_name=None):
    runtime = load_sw_runtime_config(current_machine_config=current_machine_config)
    resolved_machine_name = (
        _resolved_machine_name()
        if current_machine_config is None and machine_name is None
        else (machine_name if machine_name is not None else "<override>")
    )
    return RuntimeSpec(
        machine_json_path=MACHINE_CONFIG_PATH,
        machine_name=resolved_machine_name,
        dense_memory_budget_gb=float(runtime["dense_memory_budget_gb"]),
        max_seed_batch=int(runtime["max_seed_batch"]),
        input_chunk_rows=int(runtime["input_chunk_rows"]),
        post_chunk_t=int(runtime["post_chunk_t"]),
        sparse_time_chunk=int(runtime["sparse_time_chunk"]),
        animation_dpi=int(runtime["animation_dpi"]),
    )


def resolve_time_controls(ballistic, superresolution_enabled_override=None):
    superresolution_enabled = (
        ballistic.superresolution_enabled
        if superresolution_enabled_override is None
        else bool(superresolution_enabled_override)
    )
    superresolution_step_minutes = int(ballistic.superresolution_step_minutes)
    base_time_step_minutes = int(ballistic.base_time_step_minutes)
    time_step_minutes = (
        superresolution_step_minutes
        if superresolution_enabled
        else base_time_step_minutes
    )
    time_step_hours = float(time_step_minutes) / 60.0
    return TimeControls(
        source_json_path=ballistic.json_path,
        superresolution_enabled=superresolution_enabled,
        superresolution_step_minutes=superresolution_step_minutes,
        base_time_step_minutes=base_time_step_minutes,
        time_step_minutes=int(time_step_minutes),
        time_step_hours=time_step_hours,
        time_freq=f"{int(time_step_minutes)}min",
    )
