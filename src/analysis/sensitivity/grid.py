from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import re

import yaml

from src.utils.config import load_yaml_config


def _resolve_base_config(study_config: dict) -> dict:
    base_experiment_config = study_config["base_experiment_config"]
    if isinstance(base_experiment_config, dict):
        return deepcopy(base_experiment_config)

    base_experiment_config_path = Path(base_experiment_config)
    if base_experiment_config_path.is_absolute():
        return load_yaml_config(base_experiment_config_path)

    config_path = study_config.get("_config_path")
    if config_path is None:
        return load_yaml_config(base_experiment_config_path)

    resolved_base_path = (Path(config_path).resolve().parent / base_experiment_config_path).resolve()
    return load_yaml_config(resolved_base_path)


def _resolve_base_config_path(study_config: dict) -> Path | None:
    base_experiment_config = study_config["base_experiment_config"]
    if isinstance(base_experiment_config, dict):
        return None

    base_experiment_config_path = Path(base_experiment_config)
    if base_experiment_config_path.is_absolute():
        return base_experiment_config_path.resolve()

    config_path = study_config.get("_config_path")
    if config_path is None:
        return base_experiment_config_path.resolve()

    return (Path(config_path).resolve().parent / base_experiment_config_path).resolve()


def _set_param_path(data: dict, path: str, value) -> None:
    current = data
    parts = path.split(".")
    for part in parts[:-1]:
        if not isinstance(current, dict) or part not in current:
            raise ValueError(f"Unknown parameter path: {path}")
        current = current[part]

    leaf_key = parts[-1]
    if not isinstance(current, dict) or leaf_key not in current:
        raise ValueError(f"Unknown parameter path: {path}")
    current[leaf_key] = value


def _slugify_component(value, *, field_name: str) -> str:
    if isinstance(value, str):
        raw_value = value.strip()
        if not raw_value:
            raise ValueError(f"Each sweep must define a non-empty {field_name}.")
    else:
        raw_value = str(value)

    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", raw_value)
    if not slug:
        raise ValueError(f"Unable to generate a safe slug for {field_name}: {value!r}")
    return slug


def expand_sensitivity_runs(study_config: dict) -> list[dict]:
    base_config = _resolve_base_config(study_config)
    base_config_path = _resolve_base_config_path(study_config)
    runs = []
    seen_run_ids: set[str] = set()

    for sweep in study_config["sweeps"]:
        sweep_name = sweep["name"]
        sweep_slug = _slugify_component(sweep_name, field_name="sweep name")
        for value in sweep["values"]:
            resolved_config = deepcopy(base_config)
            _set_param_path(resolved_config, sweep["param"], value)
            value_slug = _slugify_component(value, field_name=f"value for sweep '{sweep_name}'")
            run_id = f"{sweep_slug}__{value_slug}"
            if run_id in seen_run_ids:
                raise ValueError(f"Duplicate generated run_id '{run_id}' detected during sweep expansion.")
            seen_run_ids.add(run_id)
            runs.append(
                {
                    "run_id": run_id,
                    "sweep_name": sweep_name,
                    "param_path": sweep["param"],
                    "param_value": value,
                    "resolved_config": resolved_config,
                    "_base_experiment_config_path": str(base_config_path) if base_config_path else None,
                }
            )

    return runs


def write_resolved_config(run_dir: Path, config: dict) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    output_path = run_dir / "experiment_config_resolved.yaml"
    output_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return output_path
