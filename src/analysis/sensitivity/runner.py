from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import tempfile

from main import run_kfold_experiment
from src.analysis.sensitivity.html import build_sensitivity_html
from src.analysis.sensitivity.plots import build_sensitivity_plots
from src.analysis.sensitivity.report import build_sensitivity_tables
from src.utils import load_experiment_config, resolve_device
import yaml

from .config import load_sensitivity_config
from .grid import expand_sensitivity_runs, write_resolved_config


def _resolve_base_experiment_config_path(study_config: dict) -> Path:
    base_ref_path = Path(study_config["base_experiment_config"])
    if base_ref_path.is_absolute():
        return base_ref_path
    return (Path(study_config["_config_path"]).resolve().parent / base_ref_path).resolve()


def _load_resolved_experiment_config(resolved_config: dict, *, base_experiment_path: Path) -> dict:
    materialized_config = _materialize_resolved_config({"resolved_config": resolved_config})
    with tempfile.NamedTemporaryFile(
        "w",
        suffix=".yaml",
        prefix=f"{base_experiment_path.stem}_sensitivity_",
        dir=base_experiment_path.parent,
        encoding="utf-8",
        delete=False,
    ) as handle:
        yaml.safe_dump(materialized_config, handle, sort_keys=False)
        temp_config_path = Path(handle.name)

    try:
        return load_experiment_config(temp_config_path)
    finally:
        temp_config_path.unlink(missing_ok=True)


def _build_runtime_config(experiment_config: dict, *, run_dir: Path) -> dict:
    dataset_entries = list(experiment_config.get("dataset_entries", []))
    if len(dataset_entries) != 1:
        raise ValueError("Sensitivity runner currently requires exactly one dataset entry.")

    dataset_entry = dataset_entries[0]
    dataset_config = dataset_entry["dataset"]

    runtime_config = {
        "data_path": str(dataset_config["path"]),
        "dataset_name": str(dataset_config["name"]),
        "num_rules": int(experiment_config["model"]["num_rules"]),
        "rule_output_dim": int(experiment_config["model"]["rule_output_dim"]),
        "aggregation_method": str(experiment_config["model"]["aggregation_method"]),
        "batch_size": int(experiment_config["training"].get("batch_size", 1)),
        "lr": float(experiment_config["training"]["lr"]),
        "max_epoch": int(experiment_config["training"]["max_epoch"]),
        "fcm_m": float(experiment_config["model"]["fcm_m"]),
        "fcm_max_iter": int(experiment_config["model"]["fcm_max_iter"]),
        "patience": int(experiment_config["training"]["patience"]),
        "device": resolve_device(str(experiment_config["training"]["device"])),
        "hidden_dim": int(experiment_config["model"]["hidden_dim"]),
        "rule_temperature": float(experiment_config["model"]["rule_temperature"]),
        "use_competitive_scoring": bool(experiment_config["model"]["use_competitive_scoring"]),
        "explanation_samples_per_fold": int(
            experiment_config["output"].get("explanation_samples_per_fold", 3)
        ),
        "output_root_dir": str(run_dir),
        "run_name": None,
        "n_splits": int(experiment_config["cross_validation"]["n_splits"]),
        "random_state": int(experiment_config["cross_validation"]["random_state"]),
    }

    runtime_config.update(dataset_entry.get("training", {}))
    runtime_config.update(dataset_entry.get("cross_validation", {}))
    runtime_config.update(dataset_entry.get("output", {}))
    runtime_config["output_root_dir"] = str(run_dir)
    runtime_config["run_name"] = None
    runtime_config["device"] = resolve_device(str(runtime_config["device"]))
    return runtime_config


def _build_runtime_config_from_resolved_config(
    resolved_config: dict,
    *,
    config_source_dir: Path,
    run_dir: Path,
) -> dict:
    dataset_entries = list(resolved_config.get("datasets", []))
    if len(dataset_entries) != 1:
        raise ValueError("Sensitivity runner currently requires exactly one dataset entry.")

    dataset_entry = dataset_entries[0]
    dataset_ref = dataset_entry.get("config") or dataset_entry.get("dataset_config")
    if not dataset_ref:
        raise ValueError("Each dataset entry must define 'config' or 'dataset_config'.")

    dataset_config_path = Path(dataset_ref)
    if not dataset_config_path.is_absolute():
        dataset_config_path = (Path(config_source_dir) / dataset_config_path).resolve()

    loaded_experiment_config = deepcopy(resolved_config)
    loaded_experiment_config["dataset_entries"] = [
        {
            "dataset": yaml.safe_load(dataset_config_path.read_text(encoding="utf-8")) or {},
            "training": dict(dataset_entry.get("training", {})),
            "cross_validation": dict(dataset_entry.get("cross_validation", {})),
            "output": dict(dataset_entry.get("output", {})),
        }
    ]
    return _build_runtime_config(loaded_experiment_config, run_dir=run_dir)


def _materialize_resolved_config(run_spec: dict) -> dict:
    resolved_config = deepcopy(run_spec["resolved_config"])
    resolved_config.pop("_runner_runtime_config", None)
    return resolved_config


def _extract_cv_metadata(run_spec: dict) -> dict:
    cross_validation = run_spec.get("resolved_config", {}).get("cross_validation", {})
    metadata = {}
    if "n_splits" in cross_validation:
        metadata["n_splits"] = cross_validation["n_splits"]
    if "random_state" in cross_validation:
        metadata["random_state"] = cross_validation["random_state"]
    return metadata


def _result_path(study_dir: Path, run_id: str) -> Path:
    return study_dir / "runs" / run_id / "run_result.json"


def _resolve_report_config(study_config: dict) -> dict:
    report_config = dict(study_config.get("report", {}))
    primary_metric = report_config.get("primary_metric") or "auc"
    metrics = list(report_config.get("metrics", [primary_metric]))
    if primary_metric not in metrics:
        metrics.insert(0, primary_metric)
    return {
        "primary_metric": primary_metric,
        "metrics": metrics,
        "generate_plots": bool(report_config.get("generate_plots", True)),
        "generate_html": bool(report_config.get("generate_html", True)),
    }


def _resolve_runtime_config(run_spec: dict, study_meta: dict, run_dir: Path) -> dict:
    runtime_config = run_spec["resolved_config"].get("_runner_runtime_config")
    if runtime_config is not None:
        return deepcopy(runtime_config)

    base_experiment_path = study_meta.get("base_experiment_path")
    if base_experiment_path is not None:
        loaded_experiment_config = _load_resolved_experiment_config(
            run_spec["resolved_config"],
            base_experiment_path=Path(base_experiment_path),
        )
        return _build_runtime_config(loaded_experiment_config, run_dir=run_dir)

    base_experiment_path = run_spec.get("_base_experiment_config_path")
    if base_experiment_path:
        config_source_dir = Path(base_experiment_path).resolve().parent
    else:
        config_source_dir = Path(study_meta["study_dir"])

    return _build_runtime_config_from_resolved_config(
        run_spec["resolved_config"],
        config_source_dir=config_source_dir,
        run_dir=run_dir,
    )


def run_single_sensitivity_item(run_spec: dict, study_meta: dict) -> dict:
    run_dir = Path(study_meta["study_dir"]) / "runs" / run_spec["run_id"]
    run_dir.mkdir(parents=True, exist_ok=True)
    write_resolved_config(run_dir, _materialize_resolved_config(run_spec))

    runtime_config = None
    try:
        runtime_config = _resolve_runtime_config(run_spec, study_meta, run_dir)
        result = run_kfold_experiment(runtime_config)
        best_epochs = list(result.get("best_epochs", []))
        payload = {
            "study_name": study_meta["study_name"],
            "sweep_name": run_spec["sweep_name"],
            "param_path": run_spec["param_path"],
            "param_value": run_spec["param_value"],
            "run_id": run_spec["run_id"],
            "dataset_name": result["dataset_name"],
            "status": "success",
            "mean_acc": result["mean_metrics"]["acc"],
            "mean_f1": result["mean_metrics"]["f1"],
            "mean_auc": result["mean_metrics"]["auc"],
            "std_acc": result["std_metrics"]["acc"],
            "std_f1": result["std_metrics"]["f1"],
            "std_auc": result["std_metrics"]["auc"],
            "best_epoch_mean": sum(best_epochs) / len(best_epochs) if best_epochs else 0.0,
            "training_time_sec": result["training_time_sec"],
            "artifacts_path": result["artifacts_path"],
            "n_splits": runtime_config["n_splits"],
            "random_state": runtime_config["random_state"],
        }
    except Exception as exc:
        payload = {
            "study_name": study_meta["study_name"],
            "sweep_name": run_spec["sweep_name"],
            "param_path": run_spec["param_path"],
            "param_value": run_spec["param_value"],
            "run_id": run_spec["run_id"],
            "dataset_name": runtime_config.get("dataset_name") if runtime_config else None,
            "status": "failed",
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }
        if runtime_config is not None:
            payload["n_splits"] = runtime_config.get("n_splits")
            payload["random_state"] = runtime_config.get("random_state")
        else:
            payload.update(_extract_cv_metadata(run_spec))

    (run_dir / "run_result.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return payload


def run_sensitivity_study(config_path: str | Path) -> dict:
    study_config = load_sensitivity_config(config_path)
    study_settings = study_config["study"]
    study_name = study_settings["name"]
    study_dir = Path(study_settings["output_root_dir"]) / study_name
    study_dir.mkdir(parents=True, exist_ok=True)

    base_experiment_path = _resolve_base_experiment_config_path(study_config)
    continue_on_error = bool(study_settings.get("continue_on_error", False))
    resume = bool(study_settings.get("resume", False))
    report_config = _resolve_report_config(study_config)

    results = []
    success_count = 0
    failure_count = 0
    skipped_count = 0

    for run_spec in expand_sensitivity_runs(study_config):
        result_path = _result_path(study_dir, run_spec["run_id"])
        if resume and result_path.exists():
            existing_payload = json.loads(result_path.read_text(encoding="utf-8"))
            if existing_payload.get("status") == "success":
                results.append(existing_payload)
                success_count += 1
                skipped_count += 1
                continue

        run_payload = run_single_sensitivity_item(
            run_spec,
            {
                "study_name": study_name,
                "study_dir": study_dir,
                "base_experiment_path": base_experiment_path,
            },
        )
        results.append(run_payload)

        if run_payload["status"] == "success":
            success_count += 1
            continue

        failure_count += 1
        if not continue_on_error:
            raise RuntimeError(
                f"Sensitivity run '{run_spec['run_id']}' failed: {run_payload['error_message']}"
            )

    status = "completed_with_failures" if failure_count else "completed"
    manifest_payload = {
        "study_name": study_name,
        "study_dir": str(study_dir),
        "config_path": str(Path(config_path).resolve()),
        "base_experiment_config": str(base_experiment_path),
        "status": status,
        "total_runs": len(results),
        "success_count": success_count,
        "failure_count": failure_count,
        "skipped_count": skipped_count,
        "continue_on_error": continue_on_error,
        "resume": resume,
        "run_results": results,
        "table_paths": {},
        "plot_paths": [],
        "report_path": None,
    }
    manifest_path = study_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    table_paths = build_sensitivity_tables(
        study_dir.resolve(),
        primary_metric=report_config["primary_metric"],
    )
    plot_paths: list[str] = []
    if report_config["generate_plots"]:
        plot_paths = build_sensitivity_plots(
            study_dir.resolve(),
            report_config["primary_metric"],
            metrics=report_config["metrics"],
        )

    report_path = None
    if report_config["generate_html"]:
        report_path = build_sensitivity_html(study_dir.resolve(), table_paths, plot_paths)

    manifest_payload["table_paths"] = table_paths
    manifest_payload["plot_paths"] = plot_paths
    manifest_payload["report_path"] = str(report_path) if report_path is not None else None
    manifest_path.write_text(
        json.dumps(manifest_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return {**manifest_payload, "manifest_path": str(manifest_path)}
