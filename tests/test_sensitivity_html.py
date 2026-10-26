import csv
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src.analysis.sensitivity.html import build_sensitivity_html
from src.analysis.sensitivity.plots import build_sensitivity_plots


def _write_run_result(study_dir: Path, run_id: str, payload: dict) -> None:
    run_dir = study_dir / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_result.json").write_text(json.dumps(payload), encoding="utf-8")


class SensitivityHtmlTests(unittest.TestCase):
    def test_requirements_declares_pillow_dependency(self):
        requirements = Path("requirements.txt").read_text(encoding="utf-8").splitlines()
        self.assertIn("Pillow", requirements)

    def test_build_html_creates_index_page(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            study_dir = Path(tmpdir)
            report_path = build_sensitivity_html(
                study_dir,
                {
                    "summary_by_value": str(study_dir / "tables" / "summary_by_value.csv"),
                    "summary_by_parameter": str(study_dir / "tables" / "summary_by_parameter.csv"),
                    "failure_report": str(study_dir / "tables" / "failure_report.csv"),
                },
                [str(study_dir / "plots" / "num_rules_auc.png")],
            )
            self.assertTrue(report_path.exists())
            self.assertIn("Sensitivity Analysis Report", report_path.read_text(encoding="utf-8"))

    def test_build_html_includes_required_sections_failure_details_and_reproducibility(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            study_dir = Path(tmpdir)
            tables_dir = study_dir / "tables"
            plots_dir = study_dir / "plots"
            runs_dir = study_dir / "runs"
            tables_dir.mkdir(parents=True)
            plots_dir.mkdir(parents=True)
            runs_dir.mkdir(parents=True)

            (study_dir / "manifest.json").write_text(
                json.dumps(
                    {
                        "study_name": "demo-study",
                        "config_path": str(study_dir / "configs" / "study.yaml"),
                        "base_experiment_config": str(study_dir / "configs" / "base.yaml"),
                        "total_runs": 3,
                        "success_count": 2,
                        "failure_count": 1,
                    }
                ),
                encoding="utf-8",
            )
            (runs_dir / "num_rules__2").mkdir()
            (runs_dir / "num_rules__2" / "experiment_config_resolved.yaml").write_text(
                "model:\n  num_rules: 2\n",
                encoding="utf-8",
            )
            (runs_dir / "num_rules__4").mkdir()
            (runs_dir / "num_rules__4" / "experiment_config_resolved.yaml").write_text(
                "model:\n  num_rules: 4\n",
                encoding="utf-8",
            )

            (tables_dir / "raw_runs.csv").write_text(
                "\n".join(
                    [
                        "study_name,sweep_name,param_path,param_value,run_id,dataset_name,status,mean_acc,mean_f1,mean_auc,std_acc,std_f1,std_auc,best_epoch_mean,training_time_sec",
                        "demo-study,num_rules,model.num_rules,2,num_rules__2,demo-dataset,success,0.80,0.70,0.88,0.01,0.02,0.03,2.0,1.0",
                        "demo-study,num_rules,model.num_rules,4,num_rules__4,demo-dataset,success,0.82,0.72,0.91,0.02,0.03,0.01,3.0,1.2",
                        "demo-study,num_rules,model.num_rules,8,num_rules__8,demo-dataset,failed,,,,,,,,",
                    ]
                ),
                encoding="utf-8",
            )
            (tables_dir / "summary_by_value.csv").write_text(
                "\n".join(
                    [
                        "sweep_name,param_path,param_value,dataset_name,mean_acc,mean_f1,mean_auc,std_acc,std_f1,std_auc,best_epoch_mean,training_time_sec",
                        "num_rules,model.num_rules,2,demo-dataset,0.80,0.70,0.88,0.01,0.02,0.03,2.0,1.0",
                        "num_rules,model.num_rules,4,demo-dataset,0.82,0.72,0.91,0.02,0.03,0.01,3.0,1.2",
                    ]
                ),
                encoding="utf-8",
            )
            (tables_dir / "summary_by_parameter.csv").write_text(
                "\n".join(
                    [
                        "sweep_name,num_runs,best_mean_auc",
                        "num_rules,2,0.91",
                    ]
                ),
                encoding="utf-8",
            )
            (tables_dir / "failure_report.csv").write_text(
                "\n".join(
                    [
                        "run_id,sweep_name,param_value,error_type,error_message",
                        "num_rules__8,num_rules,8,RuntimeError,boom",
                    ]
                ),
                encoding="utf-8",
            )
            for plot_name in (
                "num_rules_auc.png",
                "num_rules_f1.png",
                "num_rules_acc.png",
                "num_rules_training_time_sec.png",
                "num_rules_best_epoch_mean.png",
                "overview_heatmap.png",
            ):
                (plots_dir / plot_name).write_bytes(b"\x89PNG\r\n\x1a\nplaceholder")

            report_path = build_sensitivity_html(
                study_dir,
                {
                    "raw_runs": str(tables_dir / "raw_runs.csv"),
                    "summary_by_value": str(tables_dir / "summary_by_value.csv"),
                    "summary_by_parameter": str(tables_dir / "summary_by_parameter.csv"),
                    "failure_report": str(tables_dir / "failure_report.csv"),
                },
                [
                    str(plots_dir / "num_rules_auc.png"),
                    str(plots_dir / "num_rules_f1.png"),
                    str(plots_dir / "num_rules_acc.png"),
                    str(plots_dir / "num_rules_training_time_sec.png"),
                    str(plots_dir / "num_rules_best_epoch_mean.png"),
                    str(plots_dir / "overview_heatmap.png"),
                ],
            )

            html = report_path.read_text(encoding="utf-8")
            for section_name in (
                "Study Overview",
                "Key Findings",
                "Parameter Sections",
                "Failures and Warnings",
                "Reproducibility",
            ):
                self.assertIn(section_name, html)
            self.assertIn("demo-study", html)
            self.assertIn("demo-dataset", html)
            self.assertIn("RuntimeError", html)
            self.assertIn("boom", html)
            self.assertIn("python scripts/build_sensitivity_report.py --study-dir", html)
            self.assertIn("experiment_config_resolved.yaml", html)
            self.assertIn("overview_heatmap.png", html)

    def test_build_plots_creates_pngs_for_requested_metrics_and_cost_views(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            study_dir = Path(tmpdir)
            _write_run_result(
                study_dir,
                "num_rules__2",
                {
                    "study_name": "demo-study",
                    "sweep_name": "num_rules",
                    "param_path": "model.num_rules",
                    "param_value": 2,
                    "run_id": "num_rules__2",
                    "dataset_name": "demo-dataset",
                    "status": "success",
                    "mean_acc": 0.80,
                    "mean_f1": 0.70,
                    "mean_auc": 0.88,
                    "std_acc": 0.01,
                    "std_f1": 0.02,
                    "std_auc": 0.03,
                    "best_epoch_mean": 2.0,
                    "training_time_sec": 1.0,
                    "artifacts_path": "artifacts/demo-2",
                },
            )
            _write_run_result(
                study_dir,
                "num_rules__4",
                {
                    "study_name": "demo-study",
                    "sweep_name": "num_rules",
                    "param_path": "model.num_rules",
                    "param_value": 4,
                    "run_id": "num_rules__4",
                    "dataset_name": "demo-dataset",
                    "status": "success",
                    "mean_acc": 0.82,
                    "mean_f1": 0.72,
                    "mean_auc": 0.91,
                    "std_acc": 0.02,
                    "std_f1": 0.03,
                    "std_auc": 0.01,
                    "best_epoch_mean": 3.0,
                    "training_time_sec": 1.2,
                    "artifacts_path": "artifacts/demo-4",
                },
            )
            _write_run_result(
                study_dir,
                "num_rules__8",
                {
                    "study_name": "demo-study",
                    "sweep_name": "num_rules",
                    "param_path": "model.num_rules",
                    "param_value": 8,
                    "run_id": "num_rules__8",
                    "dataset_name": "demo-dataset",
                    "status": "failed",
                    "error_type": "RuntimeError",
                    "error_message": "boom",
                },
            )

            plot_paths = build_sensitivity_plots(study_dir, "auc", metrics=["auc", "f1", "acc"])

            self.assertEqual(6, len(plot_paths))
            plot_names = {Path(path).name for path in plot_paths}
            self.assertEqual(
                {
                    "num_rules_auc.png",
                    "num_rules_f1.png",
                    "num_rules_acc.png",
                    "num_rules_training_time_sec.png",
                    "num_rules_best_epoch_mean.png",
                    "overview_heatmap.png",
                },
                plot_names,
            )
            for path in plot_paths:
                plot_path = Path(path)
                self.assertTrue(plot_path.exists())
                self.assertEqual(b"\x89PNG\r\n\x1a\n", plot_path.read_bytes()[:8])

    def test_build_plots_passes_std_metrics_as_error_bars_for_performance_plots(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            study_dir = Path(tmpdir)
            _write_run_result(
                study_dir,
                "num_rules__2",
                {
                    "study_name": "demo-study",
                    "sweep_name": "num_rules",
                    "param_path": "model.num_rules",
                    "param_value": 2,
                    "run_id": "num_rules__2",
                    "dataset_name": "demo-dataset",
                    "status": "success",
                    "mean_acc": 0.80,
                    "mean_f1": 0.70,
                    "mean_auc": 0.88,
                    "std_acc": 0.01,
                    "std_f1": 0.02,
                    "std_auc": 0.03,
                    "best_epoch_mean": 2.0,
                    "training_time_sec": 1.0,
                    "artifacts_path": "artifacts/demo-2",
                },
            )
            _write_run_result(
                study_dir,
                "num_rules__4",
                {
                    "study_name": "demo-study",
                    "sweep_name": "num_rules",
                    "param_path": "model.num_rules",
                    "param_value": 4,
                    "run_id": "num_rules__4",
                    "dataset_name": "demo-dataset",
                    "status": "success",
                    "mean_acc": 0.82,
                    "mean_f1": 0.72,
                    "mean_auc": 0.91,
                    "std_acc": 0.02,
                    "std_f1": 0.03,
                    "std_auc": 0.01,
                    "best_epoch_mean": 3.0,
                    "training_time_sec": 1.2,
                    "artifacts_path": "artifacts/demo-4",
                },
            )

            with patch("src.analysis.sensitivity.plots._draw_plot") as mock_draw:
                with patch("src.analysis.sensitivity.plots._draw_overview_heatmap"):
                    build_sensitivity_plots(study_dir, "auc", metrics=["auc", "f1", "acc"])

            observed_error_values = {
                Path(call.args[0]).name: call.kwargs["error_values"] for call in mock_draw.call_args_list
            }
            self.assertEqual([0.03, 0.01], observed_error_values["num_rules_auc.png"])
            self.assertEqual([0.02, 0.03], observed_error_values["num_rules_f1.png"])
            self.assertEqual([0.01, 0.02], observed_error_values["num_rules_acc.png"])
            self.assertIsNone(observed_error_values["num_rules_training_time_sec.png"])
            self.assertIsNone(observed_error_values["num_rules_best_epoch_mean.png"])

    def test_report_builder_cli_generates_tables_plot_and_html_for_existing_study_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            study_dir = Path(tmpdir)
            _write_run_result(
                study_dir,
                "num_rules__2",
                {
                    "study_name": "demo-study",
                    "sweep_name": "num_rules",
                    "param_path": "model.num_rules",
                    "param_value": 2,
                    "run_id": "num_rules__2",
                    "dataset_name": "demo-dataset",
                    "status": "success",
                    "mean_acc": 0.80,
                    "mean_f1": 0.70,
                    "mean_auc": 0.88,
                    "std_acc": 0.01,
                    "std_f1": 0.02,
                    "std_auc": 0.03,
                    "best_epoch_mean": 2.0,
                    "training_time_sec": 1.0,
                    "artifacts_path": "artifacts/demo-2",
                },
            )
            _write_run_result(
                study_dir,
                "num_rules__4",
                {
                    "study_name": "demo-study",
                    "sweep_name": "num_rules",
                    "param_path": "model.num_rules",
                    "param_value": 4,
                    "run_id": "num_rules__4",
                    "dataset_name": "demo-dataset",
                    "status": "success",
                    "mean_acc": 0.82,
                    "mean_f1": 0.72,
                    "mean_auc": 0.91,
                    "std_acc": 0.02,
                    "std_f1": 0.03,
                    "std_auc": 0.01,
                    "best_epoch_mean": 3.0,
                    "training_time_sec": 1.2,
                    "artifacts_path": "artifacts/demo-4",
                },
            )
            _write_run_result(
                study_dir,
                "num_rules__8",
                {
                    "study_name": "demo-study",
                    "sweep_name": "num_rules",
                    "param_path": "model.num_rules",
                    "param_value": 8,
                    "run_id": "num_rules__8",
                    "dataset_name": "demo-dataset",
                    "status": "failed",
                    "error_type": "RuntimeError",
                    "error_message": "boom",
                },
            )

            completed = subprocess.run(
                [
                    "python",
                    "scripts/build_sensitivity_report.py",
                    "--study-dir",
                    str(study_dir),
                    "--primary-metric",
                    "auc",
                ],
                check=True,
                capture_output=True,
                text=True,
            )

            payload = json.loads(completed.stdout)
            report_path = Path(payload["report_path"])
            self.assertTrue(report_path.exists())
            self.assertEqual(4, len(payload["plot_paths"]))
            for plot_path in payload["plot_paths"]:
                self.assertTrue(Path(plot_path).exists())
            html = report_path.read_text(encoding="utf-8")
            self.assertIn("num_rules_auc.png", html)
            self.assertIn("num_rules_training_time_sec.png", html)
            self.assertIn("num_rules_best_epoch_mean.png", html)
            self.assertIn("overview_heatmap.png", html)
            self.assertIn("failure_report.csv", html)

            with Path(payload["table_paths"]["failure_report"]).open("r", encoding="utf-8") as handle:
                failure_rows = list(csv.DictReader(handle))
            self.assertEqual(1, len(failure_rows))
            self.assertEqual("num_rules__8", failure_rows[0]["run_id"])


if __name__ == "__main__":
    unittest.main()
