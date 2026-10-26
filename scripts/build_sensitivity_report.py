from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.sensitivity.html import build_sensitivity_html
from src.analysis.sensitivity.config import validate_report_metrics
from src.analysis.sensitivity.plots import build_sensitivity_plots
from src.analysis.sensitivity.report import build_sensitivity_tables
from src.utils.config import load_yaml_config


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build static plots and HTML report for an existing sensitivity study."
    )
    parser.add_argument("--study-dir", required=True, help="Path to an existing sensitivity study dir.")
    parser.add_argument("--primary-metric", help="Primary metric name, for example 'auc'.")
    return parser


def _load_report_config(study_dir: Path) -> dict:
    manifest_path = study_dir / "manifest.json"
    if not manifest_path.exists():
        return {}

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config_path = manifest.get("config_path")
    if not config_path:
        return {}

    config_file = Path(config_path)
    if not config_file.exists():
        return {}

    study_config = load_yaml_config(config_file)
    return dict(study_config.get("report", {}))


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    study_dir = Path(args.study_dir).resolve()
    report_config = _load_report_config(study_dir)
    primary_metric = args.primary_metric or report_config.get("primary_metric") or "auc"
    metrics = list(report_config.get("metrics", [primary_metric]))
    effective_report_config = dict(report_config)
    effective_report_config["primary_metric"] = primary_metric
    effective_report_config["metrics"] = metrics
    validate_report_metrics(effective_report_config)
    if primary_metric not in metrics:
        metrics.insert(0, primary_metric)
    table_paths = build_sensitivity_tables(study_dir, primary_metric=primary_metric)
    plot_paths = build_sensitivity_plots(study_dir, primary_metric, metrics=metrics)
    report_path = build_sensitivity_html(study_dir, table_paths, plot_paths)
    print(
        json.dumps(
            {
                "study_dir": str(study_dir),
                "table_paths": table_paths,
                "plot_paths": plot_paths,
                "report_path": str(report_path),
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
