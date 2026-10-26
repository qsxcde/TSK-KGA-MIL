import csv
import json
import tempfile
import unittest
from pathlib import Path

from src.analysis.sensitivity.report import build_sensitivity_tables


class SensitivityReportTests(unittest.TestCase):
    def test_build_tables_creates_summary_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "runs" / "num_rules__4"
            run_dir.mkdir(parents=True)
            (run_dir / "run_result.json").write_text(
                json.dumps(
                    {
                        "study_name": "demo",
                        "sweep_name": "num_rules",
                        "param_path": "model.num_rules",
                        "param_value": 4,
                        "run_id": "num_rules__4",
                        "dataset_name": "musk1+",
                        "status": "success",
                        "mean_acc": 0.8,
                        "mean_f1": 0.7,
                        "mean_auc": 0.9,
                        "std_acc": 0.1,
                        "std_f1": 0.2,
                        "std_auc": 0.05,
                        "best_epoch_mean": 8,
                        "training_time_sec": 5.0,
                        "artifacts_path": "artifacts/demo",
                    }
                ),
                encoding="utf-8",
            )
            outputs = build_sensitivity_tables(Path(tmpdir))
            self.assertTrue(Path(outputs["summary_by_value"]).exists())
            with Path(outputs["summary_by_value"]).open("r", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual("num_rules", rows[0]["sweep_name"])

    def test_build_tables_uses_requested_primary_metric_for_parameter_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            study_dir = Path(tmpdir)
            run_dir = study_dir / "runs" / "num_rules__4"
            run_dir.mkdir(parents=True)
            (run_dir / "run_result.json").write_text(
                json.dumps(
                    {
                        "study_name": "demo",
                        "sweep_name": "num_rules",
                        "param_path": "model.num_rules",
                        "param_value": 4,
                        "run_id": "num_rules__4",
                        "dataset_name": "musk1+",
                        "status": "success",
                        "mean_acc": 0.8,
                        "mean_f1": 0.7,
                        "mean_auc": 0.9,
                        "std_acc": 0.1,
                        "std_f1": 0.2,
                        "std_auc": 0.05,
                        "best_epoch_mean": 8,
                        "training_time_sec": 5.0,
                        "artifacts_path": "artifacts/demo",
                    }
                ),
                encoding="utf-8",
            )

            outputs = build_sensitivity_tables(study_dir, primary_metric="f1")

            with Path(outputs["summary_by_parameter"]).open("r", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

            self.assertEqual(["sweep_name", "num_runs", "best_mean_f1"], list(rows[0].keys()))
            self.assertEqual("0.7", rows[0]["best_mean_f1"])


if __name__ == "__main__":
    unittest.main()
