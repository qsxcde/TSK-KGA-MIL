from __future__ import annotations

from pathlib import Path

from src.utils.config import load_yaml_config

SUPPORTED_REPORT_METRICS = ("acc", "f1", "auc")
SUPPORTED_SWEEP_FIELDS = frozenset({"name", "param", "values", "label"})


def _resolve_param_path(data: dict, path: str):
    current = data
    for part in path.split("."):
        if not isinstance(current, dict) or part not in current:
            raise ValueError(f"Unknown parameter path: {path}")
        current = current[part]
    return current


def _validate_base_experiment_shape(base_config: dict) -> None:
    datasets = list(base_config.get("datasets", []))
    if len(datasets) != 1:
        raise ValueError(
            "Sensitivity analysis v1 requires base_experiment_config to define exactly one dataset entry."
        )


def _resolve_base_experiment_config_path(config: dict, config_path: str | Path | None = None) -> Path:
    base_ref = config.get("base_experiment_config")
    if not base_ref:
        raise ValueError("Missing required config section: base_experiment_config")

    base_ref_path = Path(base_ref)
    if base_ref_path.is_absolute():
        return base_ref_path

    if config_path is not None:
        return (Path(config_path).resolve().parent / base_ref_path).resolve()

    raise ValueError(
        "Relative base_experiment_config requires config_path in validate_sensitivity_config()."
    )


def validate_report_metrics(report_config: dict) -> None:
    primary_metric = report_config.get("primary_metric") or "auc"
    metrics = list(report_config.get("metrics", [primary_metric]))

    if primary_metric not in SUPPORTED_REPORT_METRICS:
        raise ValueError(
            f"Unsupported primary report metric '{primary_metric}'. Supported metrics: {', '.join(SUPPORTED_REPORT_METRICS)}."
        )

    for metric in metrics:
        if metric not in SUPPORTED_REPORT_METRICS:
            raise ValueError(
                f"Unsupported report metric '{metric}'. Supported metrics: {', '.join(SUPPORTED_REPORT_METRICS)}."
            )


def _validate_study_config(study_config: dict) -> None:
    if not isinstance(study_config, dict):
        raise ValueError("Config section 'study' must be a mapping.")

    for field_name in ("name", "output_root_dir"):
        value = study_config.get(field_name)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Config section 'study' must define non-empty '{field_name}'.")


def validate_sensitivity_config(config: dict, *, config_path: str | Path | None = None) -> None:
    required = ["base_experiment_config", "study", "sweeps", "report"]
    for key in required:
        if key not in config:
            raise ValueError(f"Missing required config section: {key}")

    sweeps = config["sweeps"]
    if not sweeps:
        raise ValueError("Sensitivity config must define at least one sweep.")

    _validate_study_config(config["study"])

    for sweep in sweeps:
        sweep_name = sweep.get("name")
        if not isinstance(sweep_name, str) or not sweep_name.strip():
            raise ValueError("Each sweep must define a non-empty 'name'.")
        extra_fields = set(sweep) - SUPPORTED_SWEEP_FIELDS
        if "params" in sweep or isinstance(sweep.get("param"), (list, tuple, dict)):
            raise ValueError(
                f"Sweep '{sweep_name}' attempts to declare multiple parameters, which is unsupported in sensitivity analysis v1."
            )
        if extra_fields:
            raise ValueError(
                f"Sweep '{sweep_name}' defines unsupported fields: {', '.join(sorted(extra_fields))}."
            )
        if "param" not in sweep or "values" not in sweep:
            raise ValueError("Each sweep must define 'param' and 'values'.")
        if not isinstance(sweep["param"], str) or not sweep["param"].strip():
            raise ValueError(f"Sweep '{sweep_name}' must define a non-empty 'param'.")
        if not isinstance(sweep["values"], list) or not sweep["values"]:
            raise ValueError(f"Sweep '{sweep_name}' must define non-empty values.")

    validate_report_metrics(config["report"])

    base_config_path = _resolve_base_experiment_config_path(config, config_path=config_path)
    base_config = load_yaml_config(base_config_path)
    _validate_base_experiment_shape(base_config)

    for sweep in sweeps:
        _resolve_param_path(base_config, sweep["param"])


def load_sensitivity_config(path: str | Path) -> dict:
    config_path = Path(path).resolve()
    config = load_yaml_config(config_path)
    validate_sensitivity_config(config, config_path=config_path)
    config["_config_path"] = config_path
    return config
