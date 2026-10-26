from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path


def _load_run_results(study_dir: Path) -> list[dict]:
    results = []
    for path in sorted((study_dir / "runs").glob("*/run_result.json")):
        results.append(json.loads(path.read_text(encoding="utf-8")))
    return results


def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _project_rows(rows: list[dict], fieldnames: list[str]) -> list[dict]:
    return [{field: row.get(field) for field in fieldnames} for row in rows]


def build_sensitivity_tables(study_dir: Path, primary_metric: str = "auc") -> dict:
    study_dir = Path(study_dir)
    table_dir = study_dir / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    results = _load_run_results(study_dir)

    raw_path = table_dir / "raw_runs.csv"
    if results:
        fieldnames = sorted({key for row in results for key in row.keys()})
        _write_csv(raw_path, fieldnames, results)
    else:
        raw_path.write_text("", encoding="utf-8")

    success_rows = [row for row in results if row.get("status") == "success"]
    summary_by_value = table_dir / "summary_by_value.csv"
    summary_value_fields = [
        "sweep_name",
        "param_path",
        "param_value",
        "dataset_name",
        "mean_acc",
        "mean_f1",
        "mean_auc",
        "std_acc",
        "std_f1",
        "std_auc",
        "best_epoch_mean",
        "training_time_sec",
    ]
    _write_csv(
        summary_by_value,
        summary_value_fields,
        _project_rows(success_rows, summary_value_fields),
    )

    summary_by_parameter = table_dir / "summary_by_parameter.csv"
    parameter_groups: dict[str, list[dict]] = defaultdict(list)
    for row in success_rows:
        parameter_groups[str(row["sweep_name"])].append(row)

    primary_metric_field = f"mean_{primary_metric}"
    summary_metric_field = f"best_mean_{primary_metric}"
    summary_parameter_rows = []
    for sweep_name in sorted(parameter_groups):
        rows = parameter_groups[sweep_name]
        summary_parameter_rows.append(
            {
                "sweep_name": sweep_name,
                "num_runs": len(rows),
                summary_metric_field: max(float(row[primary_metric_field]) for row in rows),
            }
        )
    _write_csv(
        summary_by_parameter,
        ["sweep_name", "num_runs", summary_metric_field],
        summary_parameter_rows,
    )

    failure_path = table_dir / "failure_report.csv"
    failure_rows = [row for row in results if row.get("status") == "failed"]
    _write_csv(
        failure_path,
        ["run_id", "sweep_name", "param_value", "error_type", "error_message"],
        _project_rows(
            failure_rows,
            ["run_id", "sweep_name", "param_value", "error_type", "error_message"],
        ),
    )

    return {
        "raw_runs": str(raw_path),
        "summary_by_value": str(summary_by_value),
        "summary_by_parameter": str(summary_by_parameter),
        "failure_report": str(failure_path),
    }
