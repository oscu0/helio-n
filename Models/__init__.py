from __future__ import annotations

from importlib import import_module

from Models._base import DateRangeSpec, ModelSpec

__all__ = ["DateRangeSpec", "ModelSpec", "load_architecture", "load_date_range"]


def _load_module(architecture_id: str):
    try:
        return import_module(f"Models.{architecture_id}")
    except ModuleNotFoundError as exc:
        raise ValueError(f"Unknown architecture_id '{architecture_id}'.") from exc


def load_architecture(architecture_id: str) -> dict:
    module = _load_module(architecture_id)

    if hasattr(module, "get_architecture"):
        architecture = module.get_architecture()
    elif hasattr(module, "MODEL"):
        architecture = dict(module.MODEL.params)
    else:
        raise ValueError(
            f"Models.{architecture_id} must define get_architecture() or MODEL."
        )

    return architecture


def load_date_range(architecture_id: str, date_range_id: str) -> DateRangeSpec:
    module = _load_module(architecture_id)

    if hasattr(module, "get_date_range"):
        return module.get_date_range(date_range_id)

    date_ranges = getattr(module, "DATE_RANGES", None)
    if not date_ranges or date_range_id not in date_ranges:
        raise ValueError(
            f"Models.{architecture_id} has no date range '{date_range_id}'."
        )

    return date_ranges[date_range_id]
