from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from main import run_kfold_experiment
from src.utils import load_experiment_config, resolve_device


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run DS-TSK-MIL experiments from a YAML config.")
    parser.add_argument("--config", required=True, help="Path to experiment config YAML.")
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    dataset_entries = list(config.get("dataset_entries", []))
    if not dataset_entries:
        raise ValueError("Experiment config must define a non-empty 'datasets' list.")

    base_config = {
        "num_rules": int(config["model"]["num_rules"]),
        "rule_output_dim": int(config["model"]["rule_output_dim"]),
        "aggregation_method": str(config["model"]["aggregation_method"]),
        "batch_size": int(config["training"].get("batch_size", 1)),
        "lr": float(config["training"]["lr"]),
        "max_epoch": int(config["training"]["max_epoch"]),
        "fcm_m": float(config["model"]["fcm_m"]),
        "fcm_max_iter": int(config["model"]["fcm_max_iter"]),
        "patience": int(config["training"]["patience"]),
        "device": resolve_device(str(config["training"]["device"])),
        "hidden_dim": int(config["model"]["hidden_dim"]),
        "rule_temperature": float(config["model"]["rule_temperature"]),
        "use_competitive_scoring": bool(config["model"]["use_competitive_scoring"]),
        "explanation_samples_per_fold": int(config["output"].get("explanation_samples_per_fold", 3)),
        "output_root_dir": str(config["output"].get("root_dir", ".")),
        "run_name": config["output"].get("run_name"),
        "n_splits": int(config["cross_validation"]["n_splits"]),
        "random_state": int(config["cross_validation"]["random_state"]),
    }

    report = []
    for entry in dataset_entries:
        entry_config = dict(base_config)
        dataset_config = entry["dataset"]
        entry_config["data_path"] = str(dataset_config["path"])
        entry_config["dataset_name"] = str(dataset_config["name"])
        entry_config.update(entry.get("training", {}))
        entry_config.update(entry.get("cross_validation", {}))
        entry_config.update(entry.get("output", {}))
        entry_config["device"] = resolve_device(str(entry_config["device"]))
        report.append(run_kfold_experiment(entry_config))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
