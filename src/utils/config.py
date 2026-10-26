from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import torch
import yaml


def load_yaml_config(path: str | Path) -> dict:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _resolve_config_path(config_ref: str | Path, base_dir: Path) -> Path:
    config_path = Path(config_ref)
    if not config_path.is_absolute():
        config_path = (base_dir / config_path).resolve()
    return config_path


def _resolve_dataset_entry(entry: str | dict, experiment_dir: Path) -> dict:
    if isinstance(entry, str):
        raw_entry = {"config": entry}
    else:
        raw_entry = deepcopy(entry)

    dataset_config_ref = raw_entry.get("config") or raw_entry.get("dataset_config")
    if not dataset_config_ref:
        raise ValueError("Each dataset entry must define 'config' or 'dataset_config'.")

    dataset_config_path = _resolve_config_path(dataset_config_ref, experiment_dir)
    raw_entry["dataset"] = load_yaml_config(dataset_config_path)
    raw_entry["dataset_config_path"] = str(dataset_config_path)
    raw_entry["training"] = dict(raw_entry.get("training", {}))
    raw_entry["cross_validation"] = dict(raw_entry.get("cross_validation", {}))
    raw_entry["output"] = dict(raw_entry.get("output", {}))
    return raw_entry


def load_experiment_config(path: str | Path) -> dict:
    experiment_path = Path(path)
    config = load_yaml_config(experiment_path)

    dataset_entries = config.get("datasets", [])
    if dataset_entries:
        config["dataset_entries"] = [
            _resolve_dataset_entry(entry, experiment_path.parent) for entry in dataset_entries
        ]

    return config


def resolve_device(device_name: str) -> str:
    normalized = str(device_name).lower()
    if normalized == "mps" and not torch.backends.mps.is_available():
        return "cpu"
    return normalized
